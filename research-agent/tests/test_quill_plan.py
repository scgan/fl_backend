"""覆盖Quill PLAN 阶段的大纲纯创作边界、建议引用校验与产物输入。"""

from __future__ import annotations

import asyncio
import uuid

from fastapi import HTTPException

from app.agents.gateway import ModelGateway
from app.agents.quill_plan_graph import QuillPlanGraph
from app.agents.quill_schemas import (
    OutlineSection,
    PlanDraft,
    QuillPlanResult,
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


class FakePlanRetrieval:
    """为每条扩展查询返回相同文献。"""

    def __init__(self, knowledge_base_id: uuid.UUID) -> None:
        """创建固定研究证据。"""
        self.knowledge_base_id = knowledge_base_id
        self.search_calls = 0
        self.items = [
            _evidence("智能教育概念界定", 2018, "2018 年研究界定了智能教育概念。"),
            _evidence("智能课堂实践研究", 2021, "2021 年研究转向课堂应用。"),
        ]

    async def search(self, **_kwargs) -> RetrievalResponse:
        """返回固定 KB Version 下的两条真实测试 Evidence。"""
        self.search_calls += 1
        return RetrievalResponse(
            knowledge_base_id=self.knowledge_base_id,
            knowledge_base_version=7,
            original_query="写作大纲",
            detected_language="zh",
            rewritten_query="写作大纲",
            expanded_queries={"zh": "写作大纲", "en": "outline"},
            candidate_count=2,
            evidence=self.items,
            citations=[_citation(item, self.knowledge_base_id) for item in self.items],
            insufficient_evidence=False,
            message=None,
        )


def _default_draft() -> PlanDraft:
    """构造结构纯创作、建议引用合法的大纲草稿。"""
    return PlanDraft(
        thesis_statement="智能教育进入课堂的路径需要兼顾技术供给与教学组织变革。",
        sections=[
            OutlineSection(
                section_id="1",
                title="引言",
                writing_points=["研究背景", "问题提出"],
                suggested_citations=[1],
            ),
            OutlineSection(
                section_id="2",
                title="文献综述",
                writing_points=["概念界定", "研究现状梳理"],
                suggested_citations=[1, 2],
            ),
            OutlineSection(
                section_id="3",
                title="结论与展望",
                writing_points=["研究结论总结", "后续研究方向"],
            ),
        ],
    )


class FakePlanGateway(ModelGateway):
    """按Quill各结构化 Schema 返回固定大纲草稿。"""

    def __init__(self, draft: PlanDraft | None = None) -> None:
        """允许替换大纲草稿并记录指令供审计断言。"""
        self.draft = draft or _default_draft()
        self.instructions: list[str] = []

    async def generate_answer(self, **_kwargs) -> str:
        """满足网关抽象；PLAN 图只使用结构化接口。"""
        return ""

    async def structured(self, *, instruction=None, schema, **_kwargs):
        """根据节点请求的 Schema 返回可验证固定值。"""
        from app.agents.quill_schemas import TheoryConcept, TheoryQueryPlan

        if schema is TheoryConcept:
            return TheoryConcept(
                canonical_name="智能教育",
                search_terms=["智能教育", "AI education"],
                output_language="ZH",
            )
        if schema is TheoryQueryPlan:
            return TheoryQueryPlan(queries=["智能教育大纲素材", "AI education outline"])
        if schema is PlanDraft:
            if instruction is not None:
                self.instructions.append(instruction)
            return self.draft.model_copy(deep=True)
        raise AssertionError(f"未处理的 Schema：{schema}")


class InvalidPlanGateway(FakePlanGateway):
    """建议引用越界的大纲网关，用于验证发布闸门。"""

    def __init__(self) -> None:
        """构造建议引用越界的草稿。"""
        bad = PlanDraft(
            thesis_statement="中心论点。",
            sections=[
                OutlineSection(
                    section_id="1",
                    title="引言",
                    writing_points=["研究背景"],
                    suggested_citations=[9],
                ),
                OutlineSection(
                    section_id="2",
                    title="文献综述",
                    writing_points=["现状梳理[1]"],
                ),
                OutlineSection(
                    section_id="3",
                    title="结论",
                    writing_points=["总结"],
                ),
            ],
        )
        super().__init__(draft=bad)


def _initial_state(knowledge_base_id: uuid.UUID, **overrides) -> dict[str, object]:
    """构造完整图回归所需的冻结 PLAN 输入。"""
    state = {
        "request_id": "quill-plan-test",
        "user_id": str(uuid.uuid4()),
        "conversation_id": str(uuid.uuid4()),
        "query": "为智能教育研究设计写作大纲",
        "research_topic": "智能教育研究",
        "stage": "PLAN",
        "year_from": None,
        "year_to": None,
        "citation_style": "GBT7714",
        "rq_artifact": None,
        "based_on_artifact_id": None,
        "output_language": "AUTO",
        "knowledge_base_id": str(knowledge_base_id),
        "knowledge_base_version": 7,
        "mode": "DEEP",
        "agent_route": [],
    }
    state.update(overrides)
    return state


def test_quill_plan_graph_publishes_outline_with_suggested_citations() -> None:
    """验证 PLAN 节点顺序、建议引用投影与无引用章节的降级等级。"""
    knowledge_base_id = uuid.uuid4()
    tracked_nodes: list[str] = []

    async def track_step(name, _state, operation):
        """记录图节点顺序并执行节点闭包。"""
        tracked_nodes.append(name)
        return await operation()

    graph = QuillPlanGraph(
        FakePlanRetrieval(knowledge_base_id),
        FakePlanGateway(),
        "固定Quill Prompt",
        track_step,
    )
    final = asyncio.run(graph.run(_initial_state(knowledge_base_id)))

    result = final["plan_result"]
    assert isinstance(result, QuillPlanResult)
    assert result.status == "SUCCESS"
    assert result.stage == "PLAN"
    assert len(result.sections) == 3
    assert final["citation_validation_status"] == "PASS"
    assert final["used_citation_indices"] == [1, 2]
    assert final["validated_citations"]
    # 有建议引用但未覆盖全部证据时按 MEDIUM 发布。
    assert final["evidence_level"] == "MEDIUM"
    assert final["intent"] == "PLAN"
    assert "论文大纲" in final["final_answer"]
    assert "论文大纲" in result.evidence_summary or result.evidence_summary.get(
        "section_count"
    ) == 3
    assert result.references[0].endswith("2018.")
    assert tracked_nodes[-1] == "CitationValidator"
    assert tracked_nodes[:5] == [
        "ConceptExtraction",
        "QueryExpansion",
        "LiteratureRetrieval",
        "EvidenceDedup",
        "MetadataNormalization",
    ]


def test_quill_plan_carries_rq_artifact_into_instruction() -> None:
    """提供前序 RQ 产物时，综合指令必须包含其中的研究问题。"""

    async def run_step(_name, _state, operation):
        """执行节点闭包，不创建测试持久化步骤。"""
        return await operation()

    gateway = FakePlanGateway()
    graph = QuillPlanGraph(
        FakePlanRetrieval(uuid.uuid4()), gateway, "固定Quill Prompt", run_step
    )
    final = asyncio.run(
        graph.run(
            _initial_state(
                uuid.uuid4(),
                rq_artifact={
                    "stage": "RQ",
                    "research_questions": [
                        {"question": "智能教育如何影响师生角色分配？"}
                    ],
                },
                based_on_artifact_id="a" * 32,
            )
        )
    )
    assert final["plan_result"].based_on_artifact_id == "a" * 32
    assert gateway.instructions
    assert "智能教育如何影响师生角色分配？" in gateway.instructions[0]


def test_quill_plan_rejects_facts_in_structure_and_bad_citations() -> None:
    """大纲夹带 Citation 或建议引用越界时必须在两轮后停止发布。"""
    knowledge_base_id = uuid.uuid4()
    names: list[str] = []

    async def run_step(name, _state, operation):
        """记录节点执行顺序。"""
        names.append(name)
        return await operation()

    graph = QuillPlanGraph(
        FakePlanRetrieval(knowledge_base_id),
        InvalidPlanGateway(),
        "固定Quill Prompt",
        run_step,
    )
    try:
        asyncio.run(graph.run(_initial_state(knowledge_base_id)))
    except ValueError as exc:
        assert ("建议引用包含越界 Citation" in str(exc)) or (
            "大纲章节夹带 Citation 或具体事实" in str(exc)
        )
    else:
        raise AssertionError("大纲违反纯创作契约时必须失败")
    assert names.count("PlanSynthesis") == 2
    assert "CitationValidator" not in names


def test_quill_plan_options_validation() -> None:
    """PLAN options 白名单接受 32 位十六进制产物引用并拒绝非法格式。"""
    from app.foreign_integration.api import _validate_options

    assert _validate_options("QUILL", {"stage": "PLAN"}) == {
        "stage": "PLAN",
        "citation_style": "GBT7714",
    }
    assert _validate_options(
        "QUILL", {"stage": "PLAN", "artifact_id": "a" * 32}
    ) == {
        "stage": "PLAN",
        "citation_style": "GBT7714",
        "artifact_id": "a" * 32,
    }
    for payload in (
        {"stage": "PLAN", "artifact_id": "not-hex"},
        {"stage": "PLAN", "artifact_id": "a" * 31},
        {"stage": "PLAN", "artifact_id": 123},
        {"stage": "PLAN", "literature_limit": 10},
    ):
        try:
            _validate_options("QUILL", payload)
        except HTTPException as exc:
            assert exc.status_code == 422
        else:
            raise AssertionError(f"应拒绝非法 PLAN 参数：{payload}")
