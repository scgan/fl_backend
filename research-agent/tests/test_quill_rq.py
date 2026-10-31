"""覆盖Quill RQ 阶段的研究问题生成、引用校验与 options 白名单。"""

from __future__ import annotations

import asyncio
import uuid

from fastapi import HTTPException

from app.agents.gateway import ModelGateway
from app.agents.quill_rq_graph import QuillRQGraph
from app.agents.quill_schemas import (
    QuillRQResult,
    ResearchQuestion,
    RQDraft,
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


class FakeRQRetrieval:
    """为每条扩展查询返回相同文献，用于验证跨查询按 Chunk 去重。"""

    def __init__(self, knowledge_base_id: uuid.UUID) -> None:
        """创建固定研究证据。"""
        self.knowledge_base_id = knowledge_base_id
        self.search_calls = 0
        self.items = [
            _evidence("智能教育概念界定", 2018, "2018 年研究界定了智能教育概念。"),
            _evidence("智能课堂实践研究", 2021, "2021 年研究转向课堂应用。"),
            _evidence("生成式人工智能教育风险", 2024, "2024 年研究生成式人工智能风险。"),
        ]

    async def search(self, **_kwargs) -> RetrievalResponse:
        """返回固定 KB Version 下的三条真实测试 Evidence。"""
        self.search_calls += 1
        return RetrievalResponse(
            knowledge_base_id=self.knowledge_base_id,
            knowledge_base_version=7,
            original_query="研究问题",
            detected_language="zh",
            rewritten_query="研究问题",
            expanded_queries={"zh": "研究问题", "en": "research questions"},
            candidate_count=3,
            evidence=self.items,
            citations=[_citation(item, self.knowledge_base_id) for item in self.items],
            insufficient_evidence=False,
            message=None,
        )


def _default_draft() -> RQDraft:
    """构造引用合法的研究问题草稿。"""
    return RQDraft(
        research_questions=[
            ResearchQuestion(
                question="智能教育如何影响课堂师生角色分配？",
                rationale="现有研究以概念界定为主，课堂角色变化的实证讨论不足。[1][2]",
                citation_ids=[1, 2],
                theory_hints=["活动理论"],
                method_hints=["课堂民族志"],
            ),
            ResearchQuestion(
                question="生成式人工智能的教育风险如何被教师感知？",
                rationale="风险讨论集中于技术层面，教师感知视角缺乏。[3]",
                citation_ids=[3],
            ),
        ]
    )


class FakeRQGateway(ModelGateway):
    """按Quill各结构化 Schema 返回固定、带引用的测试结果。"""

    def __init__(self, draft: RQDraft | None = None) -> None:
        """允许替换问题草稿，其余结构化输出保持合法。"""
        self.draft = draft or _default_draft()

    async def generate_answer(self, **_kwargs) -> str:
        """满足网关抽象；RQ 图只使用结构化接口。"""
        return ""

    async def structured(self, *, schema, **_kwargs):
        """根据节点请求的 Schema 返回可验证固定值。"""
        from app.agents.quill_schemas import TheoryConcept, TheoryQueryPlan

        if schema is TheoryConcept:
            return TheoryConcept(
                canonical_name="智能教育",
                search_terms=["智能教育", "AI education"],
                output_language="ZH",
            )
        if schema is TheoryQueryPlan:
            return TheoryQueryPlan(queries=["智能教育研究现状", "AI education research"])
        if schema is RQDraft:
            return self.draft.model_copy(deep=True)
        raise AssertionError(f"未处理的 Schema：{schema}")


class RetryRQGateway(FakeRQGateway):
    """首轮问题引用越界、第二轮恢复合法的图级回退网关。"""

    def __init__(self, *, always_invalid: bool = False) -> None:
        """设置是否让问题生成两轮都持续失败。"""
        super().__init__()
        self.synthesis_calls = 0
        self.always_invalid = always_invalid

    async def structured(self, *, schema, **kwargs):
        """按轮次返回无效或有效输出。"""
        output = await super().structured(schema=schema, **kwargs)
        if schema is RQDraft:
            self.synthesis_calls += 1
            if self.synthesis_calls == 1 or self.always_invalid:
                bad_question = output.research_questions[0].model_copy(
                    update={"rationale": "问题依据引用越界。[9]"}
                )
                output = output.model_copy(
                    update={"research_questions": [bad_question]}
                )
        return output


def _initial_state(knowledge_base_id: uuid.UUID, **overrides) -> dict[str, object]:
    """构造完整图回归所需的冻结 RQ 输入。"""
    state = {
        "request_id": "quill-rq-test",
        "user_id": str(uuid.uuid4()),
        "conversation_id": str(uuid.uuid4()),
        "query": "为智能教育研究生成候选问题",
        "research_topic": "智能教育研究",
        "stage": "RQ",
        "year_from": None,
        "year_to": None,
        "output_language": "AUTO",
        "citation_style": "GBT7714",
        "knowledge_base_id": str(knowledge_base_id),
        "knowledge_base_version": 7,
        "mode": "DEEP",
        "agent_route": [],
    }
    state.update(overrides)
    return state


def test_quill_rq_graph_publishes_grounded_questions() -> None:
    """验证 RQ 节点顺序、意图编码与引用投影契约。"""
    knowledge_base_id = uuid.uuid4()
    tracked_nodes: list[str] = []

    async def track_step(name, _state, operation):
        """记录图节点顺序并执行节点闭包。"""
        tracked_nodes.append(name)
        return await operation()

    graph = QuillRQGraph(
        FakeRQRetrieval(knowledge_base_id),
        FakeRQGateway(),
        "固定Quill Prompt",
        track_step,
    )
    final = asyncio.run(graph.run(_initial_state(knowledge_base_id)))

    result = final["rq_result"]
    assert isinstance(result, QuillRQResult)
    assert result.status == "SUCCESS"
    assert result.stage == "RQ"
    assert len(result.research_questions) == 2
    assert result.evidence_level == "MEDIUM"
    assert final["citation_validation_status"] == "PASS"
    assert final["used_citation_indices"] == [1, 2, 3]
    assert final["intent"] == "RQ"
    assert final["agent_route"] == ["QUILL"]
    assert result.references[0].endswith("2018.")
    assert "候选研究问题" in final["final_answer"]
    assert result.academic_notice in final["final_answer"]
    assert tracked_nodes == [
        "ConceptExtraction",
        "QueryExpansion",
        "LiteratureRetrieval",
        "EvidenceDedup",
        "MetadataNormalization",
        "RQSynthesis",
        "RQValidation",
        "CitationValidator",
    ]


def test_quill_rq_retries_invalid_citations_without_retrieving_again() -> None:
    """首轮问题引用越界只回到 RQSynthesis 节点，并沿用同一批 Evidence。"""
    knowledge_base_id = uuid.uuid4()
    retrieval = FakeRQRetrieval(knowledge_base_id)
    gateway = RetryRQGateway()
    names: list[str] = []

    async def run_step(name, _state, operation):
        """记录节点执行顺序。"""
        names.append(name)
        return await operation()

    graph = QuillRQGraph(retrieval, gateway, "固定Quill Prompt", run_step)
    final = asyncio.run(graph.run(_initial_state(knowledge_base_id)))
    assert final["rq_result"].status == "SUCCESS"
    assert final["analysis_attempts"] == 2
    assert gateway.synthesis_calls == 2
    assert retrieval.search_calls == len(final["query_plan"].queries)
    assert names.count("LiteratureRetrieval") == 1
    assert names.count("RQSynthesis") == 2


def test_quill_rq_stops_after_second_invalid_analysis() -> None:
    """问题综合第二轮仍有越界引用时停止发布。"""
    knowledge_base_id = uuid.uuid4()
    retrieval = FakeRQRetrieval(knowledge_base_id)
    gateway = RetryRQGateway(always_invalid=True)
    names: list[str] = []

    async def run_step(name, _state, operation):
        """记录包括最终失败节点在内的执行顺序。"""
        names.append(name)
        return await operation()

    graph = QuillRQGraph(retrieval, gateway, "固定Quill Prompt", run_step)
    try:
        asyncio.run(graph.run(_initial_state(knowledge_base_id)))
    except ValueError as exc:
        assert "研究问题包含缺失、错位或越界 Citation" in str(exc)
    else:
        raise AssertionError("两轮问题引用都不合法时必须失败")
    assert "CitationValidator" not in names
    assert names.count("LiteratureRetrieval") == 1


def test_quill_rq_options_validation() -> None:
    """RQ options 白名单不接受 THEORY 专属或润改参数。"""
    from app.foreign_integration.api import _validate_options

    assert _validate_options("QUILL", {"stage": "RQ"}) == {
        "stage": "RQ",
        "citation_style": "GBT7714",
    }
    assert _validate_options(
        "QUILL", {"stage": "RQ", "year_from": 2015, "year_to": 2025}
    ) == {
        "stage": "RQ",
        "citation_style": "GBT7714",
        "year_from": 2015,
        "year_to": 2025,
    }
    for payload in (
        {"stage": "RQ", "literature_limit": 10},
        {"stage": "RQ", "artifact_id": "a" * 32},
        {"stage": "RQ", "action": "POLISH"},
    ):
        try:
            _validate_options("QUILL", payload)
        except HTTPException as exc:
            assert exc.status_code == 422
        else:
            raise AssertionError(f"应拒绝非法 RQ 参数：{payload}")
