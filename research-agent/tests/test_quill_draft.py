"""覆盖雕龙逐节起草（DRAFT）阶段的大纲驱动起草、占位符契约与 options 白名单。"""

from __future__ import annotations

import asyncio
import uuid

from fastapi import HTTPException

from app.agents.gateway import ModelGateway
from app.agents.quill_draft_graph import QuillDraftGraph
from app.agents.quill_schemas import (
    QuillDraftResult,
    SectionDraft,
)
from app.retrieval.schemas import CitationRead, EvidenceRead, RetrievalResponse


def _evidence(title: str, year: int, content: str) -> EvidenceRead:
    """构造包含真实文献、Chunk 和页码关系的固定测试 Evidence。"""
    return EvidenceRead(
        citation_index=1,
        document_id=uuid.uuid4(),
        chunk_id=uuid.uuid4(),
        document_title=title,
        authors=[{"name": "测试作者"}],
        publication_year=year,
        journal="教育研究测试刊",
        language_code="zh",
        page_start=2,
        page_end=2,
        section="研究设计",
        content=content,
        relevance_score=0.9,
        retrieval_score=0.04,
        vector_rank=1,
        lexical_rank=1,
    )


def _citation(item: EvidenceRead, knowledge_base_id: uuid.UUID) -> CitationRead:
    """由固定 Evidence 生成可回溯同一 Chunk 和 PDF 页码的 Citation。"""
    return CitationRead(
        citation_index=item.citation_index,
        document_id=item.document_id,
        chunk_id=item.chunk_id,
        document_title=item.document_title,
        page_start=item.page_start,
        page_end=item.page_end,
        section=item.section,
        quote=item.content,
        file_url=(
            f"/api/v1/retrieval/knowledge-bases/{knowledge_base_id}"
            f"/documents/{item.document_id}/file#page=2"
        ),
    )


class FakeDraftRetrieval:
    """为每条扩展查询返回相同文献。"""

    def __init__(self, knowledge_base_id: uuid.UUID) -> None:
        """创建固定研究证据。"""
        self.knowledge_base_id = knowledge_base_id
        self.search_calls = 0
        self.items = [_evidence("智能教育评价研究", 2024, "研究讨论了评价体系调适。")]

    async def search(self, **_kwargs) -> RetrievalResponse:
        """返回固定 KB Version 下的一条真实测试 Evidence。"""
        self.search_calls += 1
        return RetrievalResponse(
            knowledge_base_id=self.knowledge_base_id,
            knowledge_base_version=7,
            original_query="章节起草",
            detected_language="zh",
            rewritten_query="章节起草",
            expanded_queries={"zh": "章节起草", "en": "draft"},
            candidate_count=1,
            evidence=self.items,
            citations=[_citation(item, self.knowledge_base_id) for item in self.items],
            insufficient_evidence=False,
            message=None,
        )


PLAN_ARTIFACT = {
    "stage": "PLAN",
    "thesis_statement": "生成式人工智能倒逼评价体系系统性调适。",
    "sections": [
        {"section_id": "1", "title": "引言", "writing_points": ["研究背景", "问题提出"]},
        {"section_id": "2", "title": "逆向挑战机制", "writing_points": ["自动化加工", "范式转移"]},
    ],
}


def _default_section_text() -> SectionDraft:
    """构造引用合法且带占位符的章节草稿。"""
    return SectionDraft(
        section_text=(
            "生成式人工智能降低了低阶内容加工的价值，倒逼评价重心转向过程导向。[1]"
            "\n\n长期课堂证据仍有待积累。[待补充文献：课堂追踪研究]"
        )
    )


class FakeDraftGateway(ModelGateway):
    """按雕龙各结构化 Schema 返回固定草稿。"""

    def __init__(self, draft: SectionDraft | None = None) -> None:
        """允许替换章节草稿并记录指令供审计断言。"""
        self.draft = draft or _default_section_text()
        self.instructions: list[str] = []

    async def generate_answer(self, **_kwargs) -> str:
        """满足网关抽象；DRAFT 图只使用结构化接口。"""
        return ""

    async def structured(self, *, instruction=None, schema, **_kwargs):
        """根据节点请求的 Schema 返回可验证固定值。"""
        from app.agents.quill_schemas import TheoryConcept, TheoryQueryPlan

        if schema is TheoryConcept:
            return TheoryConcept(
                canonical_name="智能教育评价",
                search_terms=["智能教育评价"],
                output_language="ZH",
            )
        if schema is TheoryQueryPlan:
            return TheoryQueryPlan(queries=["评价体系调适"])
        if schema is SectionDraft:
            if instruction is not None:
                self.instructions.append(instruction)
            return self.draft.model_copy(deep=True)
        raise AssertionError(f"未处理的 Schema：{schema}")


def _initial_state(knowledge_base_id: uuid.UUID, **overrides) -> dict[str, object]:
    """构造完整图回归所需的冻结 DRAFT 输入。"""
    state = {
        "request_id": "quill-draft-test",
        "user_id": str(uuid.uuid4()),
        "conversation_id": str(uuid.uuid4()),
        "query": "起草引言章节",
        "research_topic": "生成式人工智能与教育评价",
        "stage": "DRAFT",
        "year_from": None,
        "year_to": None,
        "citation_style": "GBT7714",
        "plan_artifact": PLAN_ARTIFACT,
        "based_on_artifact_id": "b" * 32,
        "section_ref": "1",
        "section": PLAN_ARTIFACT["sections"][0],
        "output_language": "AUTO",
        "knowledge_base_id": str(knowledge_base_id),
        "knowledge_base_version": 7,
        "mode": "DEEP",
        "agent_route": [],
    }
    state.update(overrides)
    return state


def test_quill_draft_graph_publishes_grounded_section() -> None:
    """验证 DRAFT 节点顺序、引用投影与占位符统计契约。"""
    knowledge_base_id = uuid.uuid4()
    tracked_nodes: list[str] = []

    async def track_step(name, _state, operation):
        """记录图节点顺序并执行节点闭包。"""
        tracked_nodes.append(name)
        return await operation()

    graph = QuillDraftGraph(
        FakeDraftRetrieval(knowledge_base_id),
        FakeDraftGateway(),
        "固定Quill Prompt",
        track_step,
    )
    final = asyncio.run(graph.run(_initial_state(knowledge_base_id)))

    result = final["draft_result"]
    assert isinstance(result, QuillDraftResult)
    assert result.status == "SUCCESS"
    assert result.stage == "DRAFT"
    assert result.section_ref == "1"
    assert result.section_title == "引言"
    assert result.based_on_artifact_id == "b" * 32
    assert "[1]" in result.draft_text
    assert "[待补充文献" in result.draft_text
    assert result.evidence_summary["placeholder_count"] == 1
    assert final["citation_validation_status"] == "PASS"
    assert final["used_citation_indices"] == [1]
    assert final["evidence_level"] == "MEDIUM"
    assert final["intent"] == "DRAFT"
    assert result.limitations and "占位" in result.limitations[0]
    assert tracked_nodes[-1] == "CitationValidator"
    assert tracked_nodes[:5] == [
        "ConceptExtraction",
        "QueryExpansion",
        "LiteratureRetrieval",
        "EvidenceDedup",
        "MetadataNormalization",
    ]


def test_quill_draft_instruction_carries_outline_context() -> None:
    """起草指令必须携带章节要点、中心论点与全文大纲。"""

    async def run_step(_name, _state, operation):
        """执行节点闭包，不创建测试持久化步骤。"""
        return await operation()

    gateway = FakeDraftGateway()
    graph = QuillDraftGraph(
        FakeDraftRetrieval(uuid.uuid4()), gateway, "固定Quill Prompt", run_step
    )
    asyncio.run(graph.run(_initial_state(uuid.uuid4())))
    instruction = gateway.instructions[0]
    assert "逆向挑战机制" in instruction
    assert "论文中心论点" in instruction
    assert "全文大纲" in instruction
    assert "[待补充文献" in instruction
    assert "严禁编造引用" in instruction


def test_quill_draft_rejects_out_of_range_citations() -> None:
    """草稿引用越界时必须在两轮后停止发布。"""
    knowledge_base_id = uuid.uuid4()
    names: list[str] = []

    async def run_step(name, _state, operation):
        """记录节点执行顺序。"""
        names.append(name)
        return await operation()

    bad_gateway = FakeDraftGateway(
        draft=SectionDraft(section_text="论断引用越界。[9]")
    )
    graph = QuillDraftGraph(
        FakeDraftRetrieval(knowledge_base_id), bad_gateway, "固定Quill Prompt", run_step
    )
    try:
        asyncio.run(graph.run(_initial_state(knowledge_base_id)))
    except ValueError as exc:
        assert "草稿包含越界 Citation" in str(exc)
    else:
        raise AssertionError("草稿引用越界时必须失败")
    assert names.count("DraftSynthesis") == 2
    assert "CitationValidator" not in names


def test_quill_draft_options_validation() -> None:
    """DRAFT options 白名单要求 section_ref 与产物引用成对出现。"""
    from app.foreign_integration.api import _validate_options

    assert _validate_options(
        "QUILL",
        {"stage": "DRAFT", "artifact_id": "c" * 32, "section_ref": "2.1"},
    ) == {
        "stage": "DRAFT",
        "citation_style": "GBT7714",
        "artifact_id": "c" * 32,
        "section_ref": "2.1",
    }
    for payload in (
        {"stage": "DRAFT", "artifact_id": "c" * 32},
        {"stage": "DRAFT", "section_ref": "1"},
        {"stage": "DRAFT", "artifact_id": "c" * 32, "section_ref": ""},
        {"stage": "DRAFT", "section_ref": "1", "revise_instruction": "扩写"},
    ):
        try:
            _validate_options("QUILL", payload)
        except HTTPException as exc:
            assert exc.status_code == 422
        else:
            raise AssertionError(f"应拒绝非法 DRAFT 参数：{payload}")


def test_quill_revise_options_validation() -> None:
    """REVISE options 白名单要求 2-500 字的修改要求。"""
    from app.foreign_integration.api import _validate_options

    assert _validate_options(
        "QUILL",
        {"stage": "REVISE", "revise_instruction": "补充实证文献支撑"},
    ) == {
        "stage": "REVISE",
        "citation_style": "GBT7714",
        "revise_instruction": "补充实证文献支撑",
    }
    for payload in (
        {"stage": "REVISE"},
        {"stage": "REVISE", "revise_instruction": "改"},
        {"stage": "REVISE", "revise_instruction": "x" * 501},
        {"stage": "REVISE", "artifact_id": "c" * 32},
    ):
        try:
            _validate_options("QUILL", payload)
        except HTTPException as exc:
            assert exc.status_code == 422
        else:
            raise AssertionError(f"应拒绝非法 REVISE 参数：{payload}")
