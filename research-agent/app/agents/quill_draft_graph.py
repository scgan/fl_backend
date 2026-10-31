"""实现雕龙 DRAFT 阶段（逐节起草）的 LangGraph。

复用雕龙公共检索管线，替换后半段为单节正文综合与校验：草稿中的事实性
论断必须原位使用真实 [n] Citation，证据不足处用 [待补充文献：主题线索]
占位；校验器拒绝任何越界引用或编造的 citation_id 标记。
"""

from __future__ import annotations

import json
from collections import Counter
from typing import Any

from langgraph.graph import END, START, StateGraph

from app.agents.quill_draft_state import QuillDraftState
from app.agents.quill_graph import (
    QuillGraph,
    _format_reference,
    _normalize_theory_text,
    _theory_citation_ids,
)
from app.agents.quill_schemas import (
    QuillDraftResult,
    SectionDraft,
)

# 草稿综合每段最多生成两轮；网关内部的传输或 Schema 重试另行计数。
MAX_ANALYSIS_ATTEMPTS = 2


class QuillDraftGraph(QuillGraph):
    """执行大纲章节输入到本节草稿的固定工作流。"""

    _stage_intent = "DRAFT"
    _retrieval_purpose = "章节起草的论据支撑"

    def _build(self):
        """编译固定 Evidence、两轮有界回退的雕龙 DRAFT 图。

        Returns:
            可异步调用并以稳定节点名记录步骤的已编译 LangGraph。
        """
        builder = StateGraph(QuillDraftState)
        builder.add_node("ConceptExtraction", self._concept_extraction)
        builder.add_node("QueryExpansion", self._query_expansion)
        builder.add_node("LiteratureRetrieval", self._literature_retrieval)
        builder.add_node("EvidenceDedup", self._evidence_dedup)
        builder.add_node("MetadataNormalization", self._metadata_normalization)
        builder.add_node("DraftSynthesis", self._draft_synthesis)
        builder.add_node("DraftValidation", self._draft_validation)
        builder.add_node("CitationValidator", self._citation_validator)

        builder.add_edge(START, "ConceptExtraction")
        builder.add_edge("ConceptExtraction", "QueryExpansion")
        builder.add_edge("QueryExpansion", "LiteratureRetrieval")
        builder.add_edge("LiteratureRetrieval", "EvidenceDedup")
        builder.add_edge("EvidenceDedup", "MetadataNormalization")
        builder.add_edge("MetadataNormalization", "DraftSynthesis")
        builder.add_edge("DraftSynthesis", "DraftValidation")
        builder.add_conditional_edges(
            "DraftValidation",
            self._route_draft_validation,
            {"PASS": "CitationValidator", "RETRY": "DraftSynthesis"},
        )
        builder.add_edge("CitationValidator", END)
        return builder.compile()

    def _route_draft_validation(self, state: QuillDraftState) -> str:
        """根据草稿校验结果选择组装结果或重做起草。

        Args:
            state: 包含本轮校验状态的图状态。

        Returns:
            PASS 或 RETRY 路由标识。
        """
        return state["analysis_validation_status"]

    def _synthesis_instruction(self, state: QuillDraftState) -> str:
        """构造只包含本次真实 Evidence 与大纲章节信息的起草指令。

        Args:
            state: 包含研究主题、目标章节和最终 Evidence 的 DRAFT 状态。

        Returns:
            限制事实来源、引用边界与占位符规则的起草指令。
        """
        payload = [item.model_dump(mode="json") for item in state["evidence"]]
        language = {"ZH": "中文", "EN": "English", "BILINGUAL": "中英双语"}[
            state["output_language"]
        ]
        section = state.get("section") or {}
        artifact = state.get("plan_artifact") or {}
        parts = [
            f"只根据随后提供的已确认 Evidence，为论文“{state['research_topic']}”起草"
            f"章节 {section.get('section_id', state.get('section_ref', ''))} "
            f"「{section.get('title', '')}」的正文，使用{language}，约 400 至 800 字。",
            "该节应覆盖的写作要点："
            + json.dumps(section.get("writing_points") or [], ensure_ascii=False),
        ]
        if artifact.get("thesis_statement"):
            parts.append(f"论文中心论点：{artifact['thesis_statement']}")
        if artifact.get("sections"):
            parts.append(
                "全文大纲（仅供理解上下文，不要起草其他章节）："
                + json.dumps(
                    [
                        {"section_id": item.get("section_id"), "title": item.get("title")}
                        for item in artifact["sections"]
                    ],
                    ensure_ascii=False,
                )
            )
        parts.append(
            "写作规则：每个事实性论断必须在对应位置使用半角 [n]，n 只能来自随后 "
            "Evidence 的 citation_id；Evidence 不足以支撑的论断用 "
            "[待补充文献：主题线索] 占位，严禁编造引用或使用常识补全。"
            "正文只使用 [n]，不得输出 citation_id 字段名；不得使用[n,m]，要[n][m]。"
            "分段书写，段落之间留空行。"
        )
        parts.append(self._writing_requirements_instruction(state))
        parts.append("已确认 Evidence：" + json.dumps(payload, ensure_ascii=False))
        return "\n".join(parts)

    async def _draft_synthesis(self, state: QuillDraftState) -> dict[str, Any]:
        """基于受控 Evidence 与大纲章节生成单节草稿。

        Args:
            state: 包含研究主题和最终 Evidence 的 DRAFT 状态。

        Returns:
            等待后续节点校验的章节草稿。
        """

        async def operation() -> dict[str, Any]:
            """从受控输入生成章节草稿。

            Returns:
                等待校验的分析草稿及本轮业务尝试次数。
            """
            attempt = state.get("analysis_attempts", 0) + 1
            if attempt > MAX_ANALYSIS_ATTEMPTS:
                raise ValueError("章节起草次数超过上限")
            correction = state.get("analysis_validation_error", "")
            instruction = self._synthesis_instruction(state)
            if correction:
                # 只反馈确定性校验原因；重用冻结 Evidence，不把上轮草稿重复上传。
                instruction += f"\n上轮校验失败：{correction}。请重新起草并修正引用。"
            output = await self.gateway.structured(
                instruction=instruction,
                schema=SectionDraft,
                system_prompt=self.system_prompt,
                thinking=bool(state.get("thinking")),
            )
            return {
                "analysis_draft": SectionDraft(
                    section_text=_normalize_theory_text(output.section_text)
                ),
                "analysis_attempts": attempt,
            }

        return await self._tracked("DraftSynthesis", state, operation)

    async def _draft_validation(self, state: QuillDraftState) -> dict[str, Any]:
        """校验章节草稿的 Citation 边界与占位符规范。

        Args:
            state: 包含 DraftSynthesis 输出和本次固定 Evidence 的 DRAFT 状态。

        Returns:
            数量统计及 PASS 或 RETRY 决策。

        Raises:
            ValueError: 第二轮仍有越界引用或非法标记。
        """

        async def operation() -> dict[str, Any]:
            """检查草稿正文的确定性契约。

            Returns:
                已校验数量或触发重新起草的失败原因。

            Raises:
                ValueError: 第二轮草稿仍不满足本次 Evidence 约束。
            """
            allowed = {item.citation_index for item in state["evidence"]}
            draft = state["analysis_draft"]
            failure = ""
            if "citation_id" in draft.section_text:
                failure = "草稿包含未规范化的内部 Citation 标记"
            inline = _theory_citation_ids(draft.section_text)
            if not inline <= allowed:
                failure = failure or "草稿包含越界 Citation"
            counts = {"draft_chars": len(draft.section_text)}
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

        return await self._tracked("DraftValidation", state, operation)

    async def _citation_validator(self, state: QuillDraftState) -> dict[str, Any]:
        """映射草稿引用到真实 Document/Chunk/Page 并组装结果。

        Args:
            state: 已通过草稿校验的 DRAFT 状态。

        Returns:
            真实 Citation、参考文献、Evidence 摘要和可发布 QuillDraftResult。

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
            section = state.get("section") or {}
            used = _theory_citation_ids(draft.section_text)

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
            placeholders = draft.section_text.count("[待补充文献")
            summary = {
                "evidence_count": len(state["evidence"]),
                "document_count": len(state["documents"]),
                "year_range": [years[0], years[-1]] if years else [],
                "language_coverage": dict(languages),
                "placeholder_count": placeholders,
            }

            limitations: list[str] = []
            if placeholders:
                limitations.append(
                    f"草稿含 {placeholders} 处 [待补充文献] 占位，请补充对应文献后完善论断。"
                )
            result = QuillDraftResult(
                status="SUCCESS",
                stage="DRAFT",
                research_topic=state["research_topic"],
                language=state["output_language"],
                citation_style=state["citation_style"],
                based_on_artifact_id=state.get("based_on_artifact_id"),
                section_ref=str(state.get("section_ref", "")),
                section_title=str(section.get("title", "")),
                draft_text=draft.section_text,
                references=references,
                evidence_summary=summary,
                limitations=limitations,
                # 有真实引用即 MEDIUM；纯占位草稿按 INSUFFICIENT 发布。
                evidence_level="MEDIUM" if citations else "INSUFFICIENT",
            )
            answer = _build_draft_answer(result)
            return {
                "citation_validation_status": "PASS",
                "validated_citations": citations,
                "used_citation_indices": sorted(used),
                "references": references,
                "evidence_summary": summary,
                "evidence_level": result.evidence_level,
                "draft_result": result,
                "final_answer": answer,
            }

        return await self._tracked("CitationValidator", state, operation)


def _build_draft_answer(result: QuillDraftResult) -> str:
    """把已通过引用校验的章节草稿组织为最终 Markdown 回答。

    Args:
        result: 已完成 Citation 投影的雕龙 DRAFT 结果。

    Returns:
        带真实 Citation 编号与参考文献的 Markdown 章节。
    """
    headings = {
        "ZH": ("建议参考文献",),
        "EN": ("Suggested references",),
        "BILINGUAL": ("建议参考文献 / Suggested references",),
    }[result.language]
    sections: list[str] = [
        f"# {result.section_ref} {result.section_title}".strip(),
        result.draft_text,
    ]
    if result.references:
        sections.append(f"## {headings[0]}")
        sections.extend(f"- {reference}" for reference in result.references)
    if result.limitations:
        sections.extend(f"> {note}" for note in result.limitations)
    sections.append(f"> {result.academic_notice}")
    return "\n\n".join(sections)
