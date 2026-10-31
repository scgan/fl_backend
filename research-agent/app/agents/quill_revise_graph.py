"""实现雕龙 REVISE 阶段（修改完善）的 LangGraph。

复用雕龙公共检索管线，替换后半段为定向修改综合与校验：保留作者原意与
结构，按用户修改要求改写；新增或调整的事实性论断必须原位使用本次真实
[n] Citation，原文中过期的引用编号一律剥离，校验器拒绝越界引用。
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
from app.agents.quill_revise_state import QuillReviseState
from app.agents.quill_schemas import (
    QuillReviseResult,
    RevisedText,
)

# 修改综合每段最多生成两轮；网关内部的传输或 Schema 重试另行计数。
MAX_ANALYSIS_ATTEMPTS = 2


class QuillReviseGraph(QuillGraph):
    """执行待改文本输入到修改结果的固定工作流。"""

    _stage_intent = "REVISE"
    _retrieval_purpose = "修改论证的论据支撑"

    def _build(self):
        """编译固定 Evidence、两轮有界回退的雕龙 REVISE 图。

        Returns:
            可异步调用并以稳定节点名记录步骤的已编译 LangGraph。
        """
        builder = StateGraph(QuillReviseState)
        builder.add_node("ConceptExtraction", self._concept_extraction)
        builder.add_node("QueryExpansion", self._query_expansion)
        builder.add_node("LiteratureRetrieval", self._literature_retrieval)
        builder.add_node("EvidenceDedup", self._evidence_dedup)
        builder.add_node("MetadataNormalization", self._metadata_normalization)
        builder.add_node("ReviseSynthesis", self._revise_synthesis)
        builder.add_node("ReviseValidation", self._revise_validation)
        builder.add_node("CitationValidator", self._citation_validator)

        builder.add_edge(START, "ConceptExtraction")
        builder.add_edge("ConceptExtraction", "QueryExpansion")
        builder.add_edge("QueryExpansion", "LiteratureRetrieval")
        builder.add_edge("LiteratureRetrieval", "EvidenceDedup")
        builder.add_edge("EvidenceDedup", "MetadataNormalization")
        builder.add_edge("MetadataNormalization", "ReviseSynthesis")
        builder.add_edge("ReviseSynthesis", "ReviseValidation")
        builder.add_conditional_edges(
            "ReviseValidation",
            self._route_revise_validation,
            {"PASS": "CitationValidator", "RETRY": "ReviseSynthesis"},
        )
        builder.add_edge("CitationValidator", END)
        return builder.compile()

    def _route_revise_validation(self, state: QuillReviseState) -> str:
        """根据修改校验结果选择组装结果或重做修改。

        Args:
            state: 包含本轮校验状态的图状态。

        Returns:
            PASS 或 RETRY 路由标识。
        """
        return state["analysis_validation_status"]

    def _synthesis_instruction(self, state: QuillReviseState) -> str:
        """构造只包含原文、修改要求与真实 Evidence 的修改指令。

        Args:
            state: 包含待改文本、修改要求和最终 Evidence 的 REVISE 状态。

        Returns:
            限定改写边界与 Citation 规则的修改指令。
        """
        payload = [item.model_dump(mode="json") for item in state["evidence"]]
        language = {"ZH": "中文", "EN": "English", "BILINGUAL": "中英双语"}[
            state["output_language"]
        ]
        return (
            f"请按修改要求改写下面的论文文本，使用{language}。\n"
            f"修改要求：{state['revise_instruction']}\n"
            "改写规则：保留作者的论点、结构与原意，只按修改要求调整；"
            "原文中已有的 [n] 编号属于上一次运行，一律删除不得照抄；"
            "新增或调整的事实性论断必须在对应位置使用半角 [n]，n 只能来自随后 "
            "Evidence 的 citation_id；原文已有而 Evidence 无法支撑的论断保留文字但"
            "不得添加任何 Citation，也不得用常识补全或编造引用。"
            "正文只使用 [n]，不得输出 citation_id 字段名；不得使用[n,m]，要[n][m]。"
            f"\n待改文本：{state['query']}"
            + self._writing_requirements_instruction(state)
            + f"\n已确认 Evidence：{json.dumps(payload, ensure_ascii=False)}"
        )

    async def _revise_synthesis(self, state: QuillReviseState) -> dict[str, Any]:
        """基于受控 Evidence 与修改要求生成修改后文本。

        Args:
            state: 包含待改文本和最终 Evidence 的 REVISE 状态。

        Returns:
            等待后续节点校验的修改文本。
        """

        async def operation() -> dict[str, Any]:
            """从受控输入生成修改文本。

            Returns:
                等待校验的分析草稿及本轮业务尝试次数。
            """
            attempt = state.get("analysis_attempts", 0) + 1
            if attempt > MAX_ANALYSIS_ATTEMPTS:
                raise ValueError("学术修改次数超过上限")
            correction = state.get("analysis_validation_error", "")
            instruction = self._synthesis_instruction(state)
            if correction:
                # 只反馈确定性校验原因；重用冻结 Evidence，不把上轮草稿重复上传。
                instruction += f"\n上轮校验失败：{correction}。请重新修改并修正引用。"
            output = await self.gateway.structured(
                instruction=instruction,
                schema=RevisedText,
                system_prompt=self.system_prompt,
                thinking=bool(state.get("thinking")),
            )
            return {
                "analysis_draft": RevisedText(
                    revised_text=_normalize_theory_text(output.revised_text)
                ),
                "analysis_attempts": attempt,
            }

        return await self._tracked("ReviseSynthesis", state, operation)

    async def _revise_validation(self, state: QuillReviseState) -> dict[str, Any]:
        """校验修改文本的 Citation 边界。

        Args:
            state: 包含 ReviseSynthesis 输出和本次固定 Evidence 的 REVISE 状态。

        Returns:
            数量统计及 PASS 或 RETRY 决策。

        Raises:
            ValueError: 第二轮仍有越界引用或非法标记。
        """

        async def operation() -> dict[str, Any]:
            """检查修改文本的确定性契约。

            Returns:
                已校验数量或触发重新修改的失败原因。

            Raises:
                ValueError: 第二轮修改仍不满足本次 Evidence 约束。
            """
            allowed = {item.citation_index for item in state["evidence"]}
            draft = state["analysis_draft"]
            failure = ""
            if "citation_id" in draft.revised_text:
                failure = "修改文本包含未规范化的内部 Citation 标记"
            inline = _theory_citation_ids(draft.revised_text)
            if not inline <= allowed:
                failure = failure or "修改文本包含越界 Citation"
            counts = {"revised_chars": len(draft.revised_text)}
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

        return await self._tracked("ReviseValidation", state, operation)

    async def _citation_validator(self, state: QuillReviseState) -> dict[str, Any]:
        """映射修改文本引用到真实 Document/Chunk/Page 并组装结果。

        Args:
            state: 已通过修改校验的 REVISE 状态。

        Returns:
            真实 Citation、参考文献、Evidence 摘要和可发布 QuillReviseResult。

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
            used = _theory_citation_ids(draft.revised_text)

            citations = [item for item in state["citations"] if item.citation_index in used]
            if {item.citation_index for item in citations} != used:
                raise ValueError("Citation 无法映射到真实文献分块")

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
            }
            result = QuillReviseResult(
                status="SUCCESS",
                stage="REVISE",
                research_topic=state["query"][:200],
                language=state["output_language"],
                citation_style=state["citation_style"],
                revise_instruction=state["revise_instruction"],
                revised_text=draft.revised_text,
                references=references,
                evidence_summary=summary,
                # 新增论断带真实引用即 MEDIUM；纯表达性修改按 INSUFFICIENT 发布。
                evidence_level="MEDIUM" if citations else "INSUFFICIENT",
            )
            answer = _build_revise_answer(result)
            return {
                "citation_validation_status": "PASS",
                "validated_citations": citations,
                "used_citation_indices": sorted(used),
                "references": references,
                "evidence_summary": summary,
                "evidence_level": result.evidence_level,
                "revise_result": result,
                "final_answer": answer,
            }

        return await self._tracked("CitationValidator", state, operation)


def _build_revise_answer(result: QuillReviseResult) -> str:
    """把已通过引用校验的修改结果组织为最终 Markdown 回答。

    Args:
        result: 已完成 Citation 投影的雕龙 REVISE 结果。

    Returns:
        修改要求、修改后文本与参考文献组成的 Markdown 文档。
    """
    headings = {
        "ZH": ("修改要求", "建议参考文献"),
        "EN": ("Revision instruction", "Suggested references"),
        "BILINGUAL": (
            "修改要求 / Revision instruction",
            "建议参考文献 / Suggested references",
        ),
    }[result.language]
    sections: list[str] = [
        f"## {headings[0]}",
        result.revise_instruction,
        result.revised_text,
    ]
    if result.references:
        sections.append(f"## {headings[1]}")
        sections.extend(f"- {reference}" for reference in result.references)
    if result.limitations:
        sections.extend(f"> {note}" for note in result.limitations)
    sections.append(f"> {result.academic_notice}")
    return "\n\n".join(sections)
