"""实现Quill PLAN 阶段（写作方案）的 LangGraph。

复用Quill公共检索管线，替换后半段为大纲综合与校验：章节结构与写作要点是
纯创作（不得夹带 [n] Citation 或具体事实），建议参考的文献编号是硬证据，
必须命中本次真实 Evidence。
"""

from __future__ import annotations

import json
import re
from collections import Counter
from typing import Any

from langgraph.graph import END, START, StateGraph

from app.agents.quill_graph import (
    QuillGraph,
    _format_reference,
)
from app.agents.quill_plan_state import QuillPlanState
from app.agents.quill_schemas import (
    PlanDraft,
    QuillPlanResult,
)

# 大纲综合每段最多生成两轮；网关内部的传输或 Schema 重试另行计数。
MAX_ANALYSIS_ATTEMPTS = 2


class QuillPlanGraph(QuillGraph):
    """执行研究主题（可含前序 RQ 产物）输入到 PLAN 结果的固定工作流。"""

    _stage_intent = "PLAN"
    _retrieval_purpose = "写作方案与大纲设计"

    def _build(self):
        """编译固定 Evidence、两轮有界回退的Quill PLAN 图。

        Returns:
            可异步调用并以稳定节点名记录步骤的已编译 LangGraph。
        """
        builder = StateGraph(QuillPlanState)
        # 前半段复用 THEORY 图的受控检索管线节点。
        builder.add_node("ConceptExtraction", self._concept_extraction)
        builder.add_node("QueryExpansion", self._query_expansion)
        builder.add_node("LiteratureRetrieval", self._literature_retrieval)
        builder.add_node("EvidenceDedup", self._evidence_dedup)
        builder.add_node("MetadataNormalization", self._metadata_normalization)
        # 后半段为 PLAN 专属节点。
        builder.add_node("PlanSynthesis", self._plan_synthesis)
        builder.add_node("PlanValidation", self._plan_validation)
        builder.add_node("CitationValidator", self._citation_validator)

        builder.add_edge(START, "ConceptExtraction")
        builder.add_edge("ConceptExtraction", "QueryExpansion")
        builder.add_edge("QueryExpansion", "LiteratureRetrieval")
        builder.add_edge("LiteratureRetrieval", "EvidenceDedup")
        builder.add_edge("EvidenceDedup", "MetadataNormalization")
        builder.add_edge("MetadataNormalization", "PlanSynthesis")
        builder.add_edge("PlanSynthesis", "PlanValidation")
        builder.add_conditional_edges(
            "PlanValidation",
            self._route_plan_validation,
            {"PASS": "CitationValidator", "RETRY": "PlanSynthesis"},
        )
        builder.add_edge("CitationValidator", END)
        return builder.compile()

    def _route_plan_validation(self, state: QuillPlanState) -> str:
        """根据大纲校验结果选择组装结果或重做大纲综合。

        Args:
            state: 包含本轮校验状态的图状态。

        Returns:
            PASS 或 RETRY 路由标识。
        """
        return state["analysis_validation_status"]

    def _synthesis_instruction(self, state: QuillPlanState) -> str:
        """构造只包含本次真实 Evidence 与前序产物的写作方案指令。

        Args:
            state: 包含研究主题、可选 RQ 产物和最终 Evidence 的 PLAN 状态。

        Returns:
            限定结构创作与建议引用边界的模型指令。
        """
        payload = [item.model_dump(mode="json") for item in state["evidence"]]
        language = {"ZH": "中文", "EN": "English", "BILINGUAL": "中英双语"}[
            state["output_language"]
        ]
        parts = [
            f"只根据随后提供的已确认 Evidence，为“{state['research_topic']}”设计"
            f"一份学术论文写作大纲，使用{language}。",
            "thesis_statement 只写论文中心论点的简练结构表述；sections 为 3 至 12 个"
            "章节，section_id 用 1、2.1 这类体现顺序层级的唯一编号。",
            "这是结构性创作辅助：title 与 writing_points 只描述该节应覆盖的内容与"
            "论证安排，不得写入具体研究事实、数据，也不得使用 [n] Citation 标记。",
            "suggested_citations 是建议该节参考的真实 Evidence 编号，只能从随后 "
            "Evidence 的 citation_id 中选择；没有合适的就留空数组，不得编造或凑数。",
        ]
        artifact = state.get("rq_artifact")
        if artifact:
            questions = artifact.get("research_questions") or []
            if questions:
                parts.append(
                    "大纲需覆盖以下已确认的研究问题："
                    + json.dumps(
                        [item.get("question") for item in questions],
                        ensure_ascii=False,
                    )
                )
        parts.append(self._writing_requirements_instruction(state))
        parts.append("已确认 Evidence：" + json.dumps(payload, ensure_ascii=False))
        return "\n".join(parts)

    async def _plan_synthesis(self, state: QuillPlanState) -> dict[str, Any]:
        """基于受控 Evidence 与可选 RQ 产物生成大纲草稿。

        Args:
            state: 包含研究主题和最终 Evidence 的 PLAN 状态。

        Returns:
            等待后续节点校验的大纲草稿。
        """

        async def operation() -> dict[str, Any]:
            """从受控输入生成大纲草稿。

            Returns:
                等待校验的分析草稿及本轮业务尝试次数。
            """
            attempt = state.get("analysis_attempts", 0) + 1
            if attempt > MAX_ANALYSIS_ATTEMPTS:
                raise ValueError("写作方案生成次数超过上限")
            correction = state.get("analysis_validation_error", "")
            instruction = self._synthesis_instruction(state)
            if correction:
                # 只反馈确定性校验原因；重用冻结 Evidence，不把上轮草稿重复上传。
                instruction += f"\n上轮校验失败：{correction}。请重新生成并修正建议引用。"
            output = await self.gateway.structured(
                instruction=instruction,
                schema=PlanDraft,
                system_prompt=self.system_prompt,
                thinking=bool(state.get("thinking")),
            )
            return {
                "analysis_draft": output,
                "analysis_attempts": attempt,
            }

        return await self._tracked("PlanSynthesis", state, operation)

    async def _plan_validation(self, state: QuillPlanState) -> dict[str, Any]:
        """在组装结果前统一校验大纲的纯创作边界与建议引用范围。

        Args:
            state: 包含 PlanSynthesis 输出和本次固定 Evidence 的 PLAN 状态。

        Returns:
            数量统计及 PASS 或 RETRY 决策。

        Raises:
            ValueError: 第二轮仍违反结构创作约束或建议引用越界。
        """

        async def operation() -> dict[str, Any]:
            """检查大纲的确定性契约。

            Returns:
                已校验数量或触发重新生成的失败原因。

            Raises:
                ValueError: 第二轮大纲仍不满足本次 Evidence 约束。
            """
            allowed = {item.citation_index for item in state["evidence"]}
            draft = state["analysis_draft"]
            failure = ""
            seen_ids: set[str] = set()
            for section in draft.sections:
                if section.section_id in seen_ids:
                    failure = failure or "大纲章节编号重复"
                seen_ids.add(section.section_id)
                # 结构性字段属于纯创作：不得夹带事实引用标记。
                structured_texts = [section.title, *section.writing_points]
                if any("[n]" in text or _has_citation_marker(text) for text in structured_texts):
                    failure = failure or "大纲章节夹带 Citation 或具体事实"
                if not set(section.suggested_citations) <= allowed:
                    failure = failure or "大纲建议引用包含越界 Citation"
            counts = {"section_count": len(draft.sections)}
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

        return await self._tracked("PlanValidation", state, operation)

    async def _citation_validator(self, state: QuillPlanState) -> dict[str, Any]:
        """映射建议引用到真实 Document/Chunk/Page 并组装结果。

        Args:
            state: 已通过大纲校验的 PLAN 状态。

        Returns:
            真实 Citation、参考文献、Evidence 摘要和可发布 QuillPlanResult。

        Raises:
            ValueError: 任一建议引用无法映射到当前 Run 的真实文献分块。
        """

        async def operation() -> dict[str, Any]:
            """完成真实来源映射、参考文献格式化和最终结果组装。

            Returns:
                通过最终发布闸门的状态增量。

            Raises:
                ValueError: 建议引用与真实来源集合不一致。
            """
            draft = state["analysis_draft"]

            # 1. 汇总各章节建议引用的 Citation 编号
            used: set[int] = set()
            for section in draft.sections:
                used.update(section.suggested_citations)

            # 2. 将编号严格投影回 EvidenceDedup 固定的真实 Document/Chunk/Page 引用
            citations = [item for item in state["citations"] if item.citation_index in used]
            if {item.citation_index for item in citations} != used:
                raise ValueError("Citation 无法映射到真实文献分块")

            # 3. 只为被建议的文献生成参考文献，并计算可解释覆盖摘要
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
                "section_count": len(draft.sections),
            }

            # 4. 组装唯一专业结果；最终回答由已校验结构确定性组织。
            result = QuillPlanResult(
                status="SUCCESS",
                stage="PLAN",
                research_topic=state["research_topic"],
                language=state["output_language"],
                citation_style=state["citation_style"],
                based_on_artifact_id=state.get("based_on_artifact_id"),
                thesis_statement=draft.thesis_statement,
                sections=draft.sections,
                references=references,
                evidence_summary=summary,
                # 建议引用为空时没有可溯源 Citation，必须按 INSUFFICIENT 发布。
                evidence_level="MEDIUM" if citations else "INSUFFICIENT",
            )
            answer = _build_plan_answer(result)
            return {
                "citation_validation_status": "PASS",
                "validated_citations": citations,
                "used_citation_indices": sorted(used),
                "references": references,
                "evidence_summary": summary,
                "evidence_level": result.evidence_level,
                "plan_result": result,
                "final_answer": answer,
            }

        return await self._tracked("CitationValidator", state, operation)


def _has_citation_marker(text: str) -> bool:
    """检测结构性文本是否夹带全角或半角 Citation 标记。

    Args:
        text: 章节标题或写作要点文本。

    Returns:
        存在任何形式的 Citation 标记时为真。
    """
    return bool(re.search(r"\[\s*\d+\s*\]|【\s*\d+\s*】|［\s*\d+\s*］", text))


def _build_plan_answer(result: QuillPlanResult) -> str:
    """把已通过校验的大纲结果组织为最终 Markdown 回答。

    Args:
        result: 已完成 Citation 投影的Quill PLAN 结果。

    Returns:
        章节树与建议参考文献组成的 Markdown 文档。
    """
    headings = {
        "ZH": ("中心论点", "论文大纲", "建议参考文献"),
        "EN": ("Thesis statement", "Outline", "Suggested references"),
        "BILINGUAL": (
            "中心论点 / Thesis statement",
            "论文大纲 / Outline",
            "建议参考文献 / Suggested references",
        ),
    }[result.language]
    sections: list[str] = [
        f"# {result.research_topic}",
        f"## {headings[0]}",
        result.thesis_statement,
        f"## {headings[1]}",
    ]
    for section in result.sections:
        points = "\n".join(f"- {point}" for point in section.writing_points)
        citations = ""
        if section.suggested_citations:
            markers = "".join(f"[{cid}]" for cid in section.suggested_citations)
            citations = f"\n\n建议参考：{markers}"
        sections.append(
            f"### {section.section_id} {section.title}\n\n{points}{citations}"
        )
    if result.references:
        sections.append(f"## {headings[2]}")
        sections.extend(f"- {reference}" for reference in result.references)
    if result.limitations:
        sections.extend(f"> {note}" for note in result.limitations)
    sections.append(f"> {result.academic_notice}")
    return "\n\n".join(sections)
