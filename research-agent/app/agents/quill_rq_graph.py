"""实现Quill RQ 阶段（研究问题生成）的 LangGraph。

复用Quill公共检索管线（概念解析、双语检索、去重与文献整理），替换后半段
为研究问题综合与校验：问题表述是创作，提出依据是硬证据——每个 rationale
必须原位使用真实 [n] Citation，理论/方法提示只是待查证的创作性线索。
"""

from __future__ import annotations

import json
from collections import Counter
from typing import Any

from langgraph.graph import END, START, StateGraph

from app.agents.quill_graph import (
    QuillGraph,
    _format_reference,
    _normalize_theory_text,
    _theory_citation_ids,
)
from app.agents.quill_rq_state import QuillRQState
from app.agents.quill_schemas import (
    QuillRQResult,
    RQDraft,
)

# 理论综合每段最多生成两轮；网关内部的传输或 Schema 重试另行计数。
MAX_ANALYSIS_ATTEMPTS = 2
# 单次生成的候选研究问题上限。
MAX_QUESTIONS = 5


class QuillRQGraph(QuillGraph):
    """执行研究主题输入到 RQ 结果的固定工作流。"""

    _stage_intent = "RQ"
    _retrieval_purpose = "研究问题生成"

    def _build(self):
        """编译固定 Evidence、两轮有界回退的Quill RQ 图。

        Returns:
            可异步调用并以稳定节点名记录步骤的已编译 LangGraph。
        """
        builder = StateGraph(QuillRQState)
        # 前半段复用 THEORY 图的受控检索管线节点（概念、扩展、检索、去重、整理）。
        builder.add_node("ConceptExtraction", self._concept_extraction)
        builder.add_node("QueryExpansion", self._query_expansion)
        builder.add_node("LiteratureRetrieval", self._literature_retrieval)
        builder.add_node("EvidenceDedup", self._evidence_dedup)
        builder.add_node("MetadataNormalization", self._metadata_normalization)
        # 后半段为 RQ 专属节点。
        builder.add_node("RQSynthesis", self._rq_synthesis)
        builder.add_node("RQValidation", self._rq_validation)
        builder.add_node("CitationValidator", self._citation_validator)

        builder.add_edge(START, "ConceptExtraction")
        builder.add_edge("ConceptExtraction", "QueryExpansion")
        builder.add_edge("QueryExpansion", "LiteratureRetrieval")
        builder.add_edge("LiteratureRetrieval", "EvidenceDedup")
        builder.add_edge("EvidenceDedup", "MetadataNormalization")
        builder.add_edge("MetadataNormalization", "RQSynthesis")
        builder.add_edge("RQSynthesis", "RQValidation")
        builder.add_conditional_edges(
            "RQValidation",
            self._route_rq_validation,
            {"PASS": "CitationValidator", "RETRY": "RQSynthesis"},
        )
        builder.add_edge("CitationValidator", END)
        return builder.compile()

    def _route_rq_validation(self, state: QuillRQState) -> str:
        """根据问题校验结果选择组装结果或重做问题综合。

        Args:
            state: 包含本轮校验状态的图状态。

        Returns:
            PASS 或 RETRY 路由标识。
        """
        return state["analysis_validation_status"]

    def _synthesis_instruction(self, state: QuillRQState) -> str:
        """构造只包含本次真实 Evidence 的研究问题综合指令。

        Args:
            state: 包含研究主题、输出语言和最终 Evidence 的 RQ 状态。

        Returns:
            限制事实来源和 Citation 规则的问题生成指令。
        """
        payload = [item.model_dump(mode="json") for item in state["evidence"]]
        language = {"ZH": "中文", "EN": "English", "BILINGUAL": "中英双语"}[
            state["output_language"]
        ]
        return (
            f"只根据随后提供的已确认 Evidence，为“{state['research_topic']}”生成"
            f" 3 至 {MAX_QUESTIONS} 个候选研究问题，使用{language}。"
            "每项同时给出候选论文题目proposed_title、研究对象research_object和"
            "研究范围research_scope。三项属于创作建议，不写事实结论或Citation，"
            "用户未限定的地区、时间、样本只能标为建议或待确定，不得冒充已确认条件。"
            "question 用一句完整的问句表述，不写 Citation；rationale 中每个事实"
            "结论必须在对应位置使用半角 [n]，citation_ids 必须列出 rationale 实际"
            "使用的全部引用编号；rationale 要说明从现有研究的范围、时间或方法分布"
            "如何推导出该问题的价值，不得把未检索到直接说成全球不存在。"
            "theory_hints 与 method_hints 是供作者进一步查证的创作性线索，每条只写"
            "简短名称，不写 Citation，也不得声称其已被证据验证。"
            "所有 [n] 和 citation_ids 只能使用随后 Evidence 的 citation_id，二者"
            "编号集合必须一致；不得输出 citation_id 字段名，不得用常识补全 Evidence "
            "未支持的结论。不得使用[n,m]，要[n][m]。"
            + self._writing_requirements_instruction(state)
            + f"\n已确认 Evidence：{json.dumps(payload, ensure_ascii=False)}"
        )

    async def _rq_synthesis(self, state: QuillRQState) -> dict[str, Any]:
        """基于一次完整 Evidence 输入生成候选研究问题草稿。

        Args:
            state: 包含研究主题、输出语言和最终 Evidence 的 RQ 状态。

        Returns:
            等待后续节点校验引用的问题草稿。
        """

        async def operation() -> dict[str, Any]:
            """从受控 Evidence 生成问题草稿。

            Returns:
                等待校验的分析草稿及本轮业务尝试次数。
            """
            attempt = state.get("analysis_attempts", 0) + 1
            if attempt > MAX_ANALYSIS_ATTEMPTS:
                raise ValueError("研究问题生成次数超过上限")
            correction = state.get("analysis_validation_error", "")
            instruction = self._synthesis_instruction(state)
            if correction:
                # 只反馈确定性校验原因；重用冻结 Evidence，不把上轮草稿重复上传。
                instruction += f"\n上轮校验失败：{correction}。请重新生成并修正引用。"
            output = await self.gateway.structured(
                instruction=instruction,
                schema=RQDraft,
                system_prompt=self.system_prompt,
                thinking=bool(state.get("thinking")),
            )
            return {
                "analysis_draft": _normalize_draft(output),
                "analysis_attempts": attempt,
            }

        return await self._tracked("RQSynthesis", state, operation)

    async def _rq_validation(self, state: QuillRQState) -> dict[str, Any]:
        """在组装结果前统一校验研究问题的 Citation 边界。

        Args:
            state: 包含 RQSynthesis 输出和本次固定 Evidence 的 RQ 状态。

        Returns:
            数量统计及 PASS 或 RETRY 决策。

        Raises:
            ValueError: 第二轮仍有缺失、错位或越界引用。
        """

        async def operation() -> dict[str, Any]:
            """检查问题草稿的全部引用边界与线索约束。

            Returns:
                已校验数量或触发重新生成的失败原因。

            Raises:
                ValueError: 第二轮结构化引用仍不满足本次 Evidence 约束。
            """
            allowed = {item.citation_index for item in state["evidence"]}
            draft = state["analysis_draft"]
            failure = ""
            for item in draft.research_questions:
                citations = set(item.citation_ids)
                if not citations <= allowed or _theory_citation_ids(item.rationale) != citations:
                    failure = failure or "研究问题包含缺失、错位或越界 Citation"
            counts = {"question_count": len(draft.research_questions)}
            if failure:
                if state.get("analysis_attempts", 0) >= MAX_ANALYSIS_ATTEMPTS:
                    raise ValueError(failure)
                return {
                    "analysis_validation_status": "RETRY",
                    "analysis_validation_error": failure,
                }
            return {
                **counts,
                "analysis_validation_status": "PASS",
                "analysis_validation_error": "",
            }

        return await self._tracked("RQValidation", state, operation)

    async def _citation_validator(self, state: QuillRQState) -> dict[str, Any]:
        """映射最终所有引用到真实 Document/Chunk/Page 并组装结果。

        Args:
            state: 已通过问题校验的 RQ 状态。

        Returns:
            真实 Citation、参考文献、Evidence 摘要和可发布 QuillRQResult。

        Raises:
            ValueError: 任一使用中的 Citation 无法映射到当前 Run 的真实文献分块。
        """

        async def operation() -> dict[str, Any]:
            """完成真实来源映射、参考文献格式化和最终结果组装。

            Returns:
                通过最终发布闸门的状态增量。

            Raises:
                ValueError: 使用中的 Citation 与真实来源集合不一致。
            """
            draft = state["analysis_draft"]

            # 1. 汇总各问题实际使用的 Citation 编号
            used: set[int] = set()
            for item in draft.research_questions:
                used.update(item.citation_ids)

            # 2. 将编号严格投影回 EvidenceDedup 固定的真实 Document/Chunk/Page 引用
            citations = [item for item in state["citations"] if item.citation_index in used]
            if {item.citation_index for item in citations} != used:
                raise ValueError("Citation 无法映射到真实文献分块")

            # 3. 只为实际使用的文献生成指定格式参考文献，并计算可解释覆盖摘要
            references = [
                _format_reference(item, state["citation_style"])
                for item in state["documents"]
                if any(citation_id in used for citation_id in item["citation_ids"])
            ]
            years = sorted({item["year"] for item in state["documents"] if item["year"]})
            languages = Counter(item["language"] for item in state["documents"])
            summary = {
                "evidence_count": len(state["evidence"]),
                "document_count": len(state["documents"]),
                "year_range": [years[0], years[-1]] if years else [],
                "language_coverage": dict(languages),
                "question_count": len(draft.research_questions),
            }

            # 4. 组装唯一专业结果；最终回答由已校验结构化事实确定性组织。
            result = QuillRQResult(
                status="SUCCESS",
                stage="RQ",
                research_topic=state["research_topic"],
                language=state["output_language"],
                citation_style=state["citation_style"],
                research_questions=draft.research_questions,
                references=references,
                evidence_summary=summary,
                evidence_level="HIGH" if len(state["documents"]) >= 5 else "MEDIUM",
            )
            answer = _build_rq_answer(result)
            return {
                "citation_validation_status": "PASS",
                "validated_citations": citations,
                "used_citation_indices": sorted(used),
                "references": references,
                "evidence_summary": summary,
                "evidence_level": result.evidence_level,
                "rq_result": result,
                "final_answer": answer,
            }

        return await self._tracked("CitationValidator", state, operation)


def _normalize_draft(draft: RQDraft) -> RQDraft:
    """把模型草稿各 rationale 中的全角 Citation 原位规范为半角 [n]。

    Args:
        draft: 模型返回的研究问题草稿。

    Returns:
        正文 Citation 标记已规范化的同结构草稿。
    """
    return draft.model_copy(
        update={
            "research_questions": [
                item.model_copy(update={"rationale": _normalize_theory_text(item.rationale)})
                for item in draft.research_questions
            ]
        }
    )


def _build_rq_answer(result: QuillRQResult) -> str:
    """把已通过引用校验的结构化结果组织为最终 Markdown 回答。

    Args:
        result: 已完成 Citation 投影的Quill RQ 结果。

    Returns:
        每条依据都带真实 Citation 编号的 Markdown 文档。
    """
    headings = {
        "ZH": ("候选研究问题", "参考文献"),
        "EN": ("Candidate research questions", "References"),
        "BILINGUAL": (
            "候选研究问题 / Candidate research questions",
            "参考文献 / References",
        ),
    }[result.language]
    sections: list[str] = [
        f"# {result.research_topic}",
        f"## {headings[0]}",
    ]
    for index, item in enumerate(result.research_questions, start=1):
        hints = ""
        if item.theory_hints or item.method_hints:
            theory = "、".join(item.theory_hints) or "—"
            method = "、".join(item.method_hints) or "—"
            hints = f"\n\n理论线索：{theory}\n\n方法线索：{method}"
        sections.append(
            f"### 问题{index}：{item.question}\n\n"
            + (f"候选题目：{item.proposed_title}\n\n" if item.proposed_title else "")
            + (f"研究对象（建议）：{item.research_object}\n\n" if item.research_object else "")
            + (f"研究范围（建议）：{item.research_scope}\n\n" if item.research_scope else "")
            + f"{item.rationale}{hints}"
        )
    if result.references:
        sections.append(f"## {headings[1]}")
        sections.extend(f"- {reference}" for reference in result.references)
    if result.limitations:
        sections.extend(f"> {note}" for note in result.limitations)
    sections.append(f"> {result.academic_notice}")
    return "\n\n".join(sections)
