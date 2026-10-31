"""覆盖智鉴规范化、证据归属、平衡检查、受限结果与引用发布闸门。"""

from __future__ import annotations

import asyncio
import json
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError

import app.agents.zhijian_graph as zhijian_graph_module
from app.agents.gateway import StructuredSchemaValidationError
from app.agents.zhijian_graph import ZhijianGraph
from app.agents.zhijian_normalization import (
    ambiguity_question,
    normalize_compare_object,
)
from app.agents.zhijian_schemas import (
    CitedFinding,
    CompareDimensionPlan,
    CompareQuery,
    CompareRetrievalContext,
    ComparisonMatrixDraftRow,
    EvidenceAssignmentBatch,
    EvidenceAssignmentDecision,
    EvidenceCellSupport,
    ParsedCompareRequest,
    TimelineItem,
    TopicItem,
    ZhijianCoreAnalysisDraft,
)
from app.retrieval.schemas import CitationRead, EvidenceRead, RetrievalResponse


def test_zhijian_structured_schema_fields_have_model_descriptions() -> None:
    """智鉴结构化输出及嵌套模型的字段都必须向模型说明业务语义。"""
    schemas = (
        ParsedCompareRequest,
        CompareDimensionPlan,
        EvidenceCellSupport,
        EvidenceAssignmentDecision,
        EvidenceAssignmentBatch,
        CitedFinding,
        ComparisonMatrixDraftRow,
        TimelineItem,
        TopicItem,
        ZhijianCoreAnalysisDraft,
    )
    missing = [
        f"{schema.__name__}.{name}"
        for schema in schemas
        for name, field in schema.model_fields.items()
        if not field.description
    ]
    assert not missing


@pytest.mark.parametrize(
    ("supports", "relevance", "error_type"),
    [
        (
            [{"object": "A", "dimension": "政策目标", "relevance": "HIGH"}],
            "IRRELEVANT",
            "assignment_irrelevant_with_supports",
        ),
        ([], "LOW", "assignment_relevant_without_supports"),
        (
            [{"object": "A", "dimension": "政策目标", "relevance": "HIGH"}] * 2,
            "HIGH",
            "assignment_duplicate_cell",
        ),
    ],
)
def test_evidence_assignment_reports_stable_rule_errors(
    supports: list[dict[str, str]], relevance: str, error_type: str
) -> None:
    """归属矛盾给出稳定类型，供网关安全地提供定向纠错。

    Args:
        supports: 模型返回的单元格支持项。
        relevance: 模型返回的总体相关度。
        error_type: 对应的安全业务错误类型。
    """
    with pytest.raises(ValidationError) as raised:
        EvidenceAssignmentDecision.model_validate(
            {
                "chunk_id": str(uuid.uuid4()),
                "supports": supports,
                "relevance": relevance,
                "assignment_reason": "测试判断",
            }
        )
    assert raised.value.errors()[0]["type"] == error_type


def test_compare_dimension_plan_rejects_punctuation_short_and_duplicate_titles() -> None:
    """验证模型维度必须满足数量、有效文字、长度和唯一性约束。"""
    with pytest.raises(ValidationError):
        CompareDimensionPlan(dimensions=["，，，。", "应用场景差异", "教师能力培养"])
    with pytest.raises(ValidationError):
        CompareDimensionPlan(dimensions=["应用场景差异", "应用场景差异", "教师能力培养"])

    plan = CompareDimensionPlan(dimensions=[",,应用场景差异,,", " 教学支持方式 ", "教师能力培养"])
    assert plan.dimensions == ["应用场景差异", "教学支持方式", "教师能力培养"]
    dimensions_schema = CompareDimensionPlan.model_json_schema()["properties"]["dimensions"]
    assert dimensions_schema["minItems"] == 3
    assert dimensions_schema["maxItems"] == 3
    assert dimensions_schema["items"]["minLength"] == 4
    assert dimensions_schema["items"]["maxLength"] == 30


def _evidence(
    *,
    chunk_id: uuid.UUID,
    document_id: uuid.UUID,
    title: str,
    year: int,
    content: str,
) -> EvidenceRead:
    """构造保留真实文档、Chunk 和页码关系的测试 Evidence。

    Args:
        chunk_id: 证据切块主键。
        document_id: 来源文档主键。
        title: 来源文档标题。
        year: 文档出版年份。
        content: 未改写的证据正文。

    Returns:
        可供智鉴检索分支返回的证据对象。
    """
    return EvidenceRead(
        citation_index=1,
        document_id=document_id,
        chunk_id=chunk_id,
        document_title=title,
        authors=[{"name": "测试作者"}],
        publication_year=year,
        journal="测试期刊",
        language_code="zh",
        page_start=3,
        page_end=3,
        section="政策分析",
        content=content,
        relevance_score=0.9,
        retrieval_score=0.04,
        vector_rank=1,
        lexical_rank=1,
    )


def _citation(
    evidence: EvidenceRead,
    knowledge_base_id: uuid.UUID,
) -> CitationRead:
    """由测试 Evidence 构造可回溯到同一真实 Chunk 的 Citation。

    Args:
        evidence: Citation 对应的测试证据。
        knowledge_base_id: 构造鉴权文件地址所需的知识库主键。

    Returns:
        与 Evidence 一一对应的引用对象。
    """
    return CitationRead(
        citation_index=evidence.citation_index,
        document_id=evidence.document_id,
        chunk_id=evidence.chunk_id,
        document_title=evidence.document_title,
        page_start=evidence.page_start,
        page_end=evidence.page_end,
        section=evidence.section,
        quote=evidence.content,
        file_url=(
            f"/api/v1/retrieval/knowledge-bases/{knowledge_base_id}"
            f"/documents/{evidence.document_id}/file#page=3"
        ),
    )


class FakeZhijianRetrieval:
    """按 A/B 查询返回含共享候选的固定双路检索结果。"""

    def __init__(self, knowledge_base_id: uuid.UUID) -> None:
        """创建具备 A、B、共享和无关候选的测试检索器。

        Args:
            knowledge_base_id: 图运行时固定的知识库主键。
        """
        self.knowledge_base_id = knowledge_base_id
        # 保存完整对象×维度查询文本，验证图确实执行六次独立召回。
        self.queries: list[str] = []
        self.a = _evidence(
            chunk_id=uuid.uuid4(),
            document_id=uuid.uuid4(),
            title="中国人工智能教育政策研究",
            year=2022,
            content="中国政策强调人工智能课程体系建设。",
        )
        self.b = _evidence(
            chunk_id=uuid.uuid4(),
            document_id=uuid.uuid4(),
            title="日本人工智能教育政策研究",
            year=2023,
            content="日本政策强调数理数据教育的衔接。",
        )
        self.shared = _evidence(
            chunk_id=uuid.uuid4(),
            document_id=uuid.uuid4(),
            title="中日人工智能教育比较",
            year=2024,
            content="中日均把教师能力建设作为政策实施条件。",
        )
        self.irrelevant = _evidence(
            chunk_id=uuid.uuid4(),
            document_id=uuid.uuid4(),
            title="校园建筑设计",
            year=2021,
            content="本文讨论校园建筑采光。",
        )

    async def search(
        self,
        *,
        knowledge_base_id: uuid.UUID,
        query: str,
        mode: str,
        filters: Any,
        evidence_limit: int | None = None,
    ) -> RetrievalResponse:
        """根据对象查询返回一侧候选，并让共享 Chunk 出现在两路召回中。

        Args:
            knowledge_base_id: 本次检索知识库主键。
            query: 包含规范对象名称的分支查询。
            mode: 固定为 FAST 的检索模式。
            filters: 图从自然语言约束生成的检索过滤器。
            evidence_limit: 智鉴独立请求的单查询证据上限。

        Returns:
            带真实 Candidate、Evidence 和 Citation 的检索响应。
        """
        del filters
        assert knowledge_base_id == self.knowledge_base_id
        assert mode == "FAST"
        assert evidence_limit == 10
        self.queries.append(query)
        items = [self.a, self.shared, self.irrelevant] if "中国" in query else [self.b, self.shared]
        return RetrievalResponse(
            knowledge_base_id=knowledge_base_id,
            knowledge_base_version=7,
            original_query=query,
            detected_language="zh",
            rewritten_query=query,
            expanded_queries={"zh": query, "en": query},
            candidate_count=len(items),
            evidence=items,
            citations=[_citation(item, knowledge_base_id) for item in items],
            insufficient_evidence=False,
            message=None,
        )


class FakeZhijianGateway:
    """按请求 Schema 返回确定结果，隔离图编排测试与外部模型波动。"""

    def __init__(self, retrieval: FakeZhijianRetrieval) -> None:
        """保存 EvidenceAssignment 所需的候选主键。

        Args:
            retrieval: 与当前图配套的固定检索器。
        """
        self.retrieval = retrieval
        # 记录归属请求次数，用于验证候选确实被拆成多个受控小批次。
        self.assignment_calls = 0

    async def generate_answer(self, **kwargs) -> str:
        """返回带全部真实测试引用的普通文本比较结果。

        Args:
            kwargs: 图传入的问题、Evidence、语言和固定 Prompt。

        Returns:
            可由 CitationValidator 校验的模型正文。
        """
        assert kwargs["system_prompt"] == "固定智鉴 Prompt"
        assert len(kwargs["evidence"]) == 3
        assert kwargs["thinking"] is False
        return "中国与日本路径各有侧重。[1][3] 双方均重视教师能力。[2]"

    async def structured(self, *, schema, **kwargs):
        """为每个结构化阶段返回通过同一 Pydantic Schema 的数据。

        Args:
            schema: 当前节点要求的输出模型类型。
            kwargs: 图传入的固定 Prompt 和任务指令。

        Returns:
            当前阶段对应的结构化测试结果。

        Raises:
            AssertionError: 图请求了测试未覆盖的结构化类型。
        """
        assert kwargs["system_prompt"] == "固定智鉴 Prompt"
        if schema is ParsedCompareRequest:
            return ParsedCompareRequest(
                topic="人工智能教育政策",
                object_a="中国",
                object_b="日本",
                year_from=2020,
                year_to=2025,
                output_language="ZH",
            )
        if schema is CompareDimensionPlan:
            return CompareDimensionPlan(dimensions=["政策目标", "实施路径", "教师培养"])
        if schema is EvidenceAssignmentBatch:
            self.assignment_calls += 1
            instruction = kwargs["instruction"]
            decisions = []
            for evidence, supports, relevance in (
                (
                    self.retrieval.a,
                    [
                        EvidenceCellSupport(object="A", dimension="政策目标", relevance="HIGH"),
                        EvidenceCellSupport(object="A", dimension="实施路径", relevance="HIGH"),
                    ],
                    "HIGH",
                ),
                (
                    self.retrieval.shared,
                    [
                        EvidenceCellSupport(object="A", dimension="教师培养", relevance="HIGH"),
                        EvidenceCellSupport(object="B", dimension="教师培养", relevance="HIGH"),
                    ],
                    "HIGH",
                ),
                (self.retrieval.irrelevant, [], "IRRELEVANT"),
                (
                    self.retrieval.b,
                    [EvidenceCellSupport(object="B", dimension="政策目标", relevance="HIGH")],
                    "HIGH",
                ),
            ):
                if str(evidence.chunk_id) not in instruction:
                    continue
                decisions.append(
                    EvidenceAssignmentDecision(
                        chunk_id=evidence.chunk_id,
                        supports=supports,
                        relevance=relevance,
                        assignment_reason="测试语义归属",
                    )
                )
            return EvidenceAssignmentBatch(decisions=decisions)
        if schema is ZhijianCoreAnalysisDraft:
            assert kwargs["thinking"] is False
            return ZhijianCoreAnalysisDraft(
                summary="中国与日本路径各有侧重。[1][3] 双方均重视教师能力。[2]",
                object_a_summary="中国强调课程体系建设。[1]",
                object_b_summary="日本强调数理数据教育衔接。[3]",
                similarities=[CitedFinding(text="双方均重视教师能力。", citation_ids=[2])],
                differences=[CitedFinding(text="两国政策实施路径不同。", citation_ids=[1, 3])],
                comparison_matrix=[
                    ComparisonMatrixDraftRow(
                        dimension="政策目标",
                        object_a="强调课程体系建设。",
                        object_b="强调数理数据教育衔接。",
                        similarity="均推进人工智能教育。",
                        difference="政策目标的表述重点不同。",
                        object_a_citation_ids=[1],
                        object_b_citation_ids=[3],
                        shared_citation_ids=[2],
                    ),
                    ComparisonMatrixDraftRow(
                        dimension="实施路径",
                        object_a="以课程体系为实施抓手。",
                        object_b="证据不足",
                        similarity="证据不足",
                        difference="证据不足",
                        object_a_citation_ids=[1],
                        object_b_citation_ids=[],
                        shared_citation_ids=[],
                    ),
                ],
                timeline=[
                    TimelineItem(
                        year=2022,
                        object="A",
                        event="中国推进人工智能课程体系建设。",
                        citation_ids=[1],
                    ),
                    TimelineItem(
                        year=2023,
                        object="B",
                        event="日本衔接数理与数据教育。",
                        citation_ids=[3],
                    ),
                ],
                topic_distribution=[
                    TopicItem(
                        topic="课程体系建设",
                        object="A",
                        summary="中国侧证据关注人工智能课程体系。",
                        citation_ids=[1],
                    ),
                    TopicItem(
                        topic="数理数据教育衔接",
                        object="B",
                        summary="日本侧证据关注数理与数据教育衔接。",
                        citation_ids=[3],
                    ),
                    TopicItem(
                        topic="教师能力建设",
                        object="SHARED",
                        summary="双方证据均将教师能力视为实施条件。",
                        citation_ids=[2],
                    ),
                    TopicItem(
                        topic="错误分侧主题",
                        object="A",
                        summary="该主题故意使用日本侧引用以验证后端过滤。",
                        citation_ids=[3],
                    ),
                ],
                limitations=["实施路径维度缺少日本侧直接证据。"],
            )
        raise AssertionError(f"未覆盖的结构化类型：{schema}")


class SharedOnlyObjectBGateway(FakeZhijianGateway):
    """模拟对象 B 只有共享材料、没有任何专属 Evidence 的归属结果。"""

    async def structured(self, *, schema, **kwargs):
        """让 B 候选也归为 SHARED，其余阶段沿用标准固定结果。

        Args:
            schema: 当前节点要求的输出模型类型。
            kwargs: 图传入的固定 Prompt 和任务指令。

        Returns:
            没有 B 专属证据的结构化测试结果。
        """
        if schema is EvidenceAssignmentBatch:
            self.assignment_calls += 1
            instruction = kwargs["instruction"]
            decisions = []
            for evidence, supports, relevance, reason in (
                (
                    self.retrieval.a,
                    [EvidenceCellSupport(object="A", dimension="政策目标", relevance="HIGH")],
                    "HIGH",
                    "A 侧专属证据",
                ),
                (
                    self.retrieval.shared,
                    [
                        EvidenceCellSupport(object="A", dimension="教师培养", relevance="HIGH"),
                        EvidenceCellSupport(object="B", dimension="教师培养", relevance="HIGH"),
                    ],
                    "HIGH",
                    "双方共享证据",
                ),
                (self.retrieval.irrelevant, [], "IRRELEVANT", "与主题无关"),
                (
                    self.retrieval.b,
                    [
                        EvidenceCellSupport(object="A", dimension="政策目标", relevance="MEDIUM"),
                        EvidenceCellSupport(object="B", dimension="政策目标", relevance="MEDIUM"),
                    ],
                    "MEDIUM",
                    "只提供双方共同背景",
                ),
            ):
                if str(evidence.chunk_id) not in instruction:
                    continue
                decisions.append(
                    EvidenceAssignmentDecision(
                        chunk_id=evidence.chunk_id,
                        supports=supports,
                        relevance=relevance,
                        assignment_reason=reason,
                    )
                )
            return EvidenceAssignmentBatch(decisions=decisions)
        return await super().structured(schema=schema, **kwargs)


class TooManyDimensionsGateway(FakeZhijianGateway):
    """模拟用户在自然语言中明确指定超过业务上限的比较维度。"""

    async def structured(self, *, schema, **kwargs):
        """只覆盖请求解析结果，其余结构化阶段沿用标准固定响应。

        Args:
            schema: 当前节点要求的输出模型类型。
            kwargs: 图传入的固定 Prompt 和任务指令。

        Returns:
            带四个明确维度的解析结果，或父类固定结果。
        """
        if schema is ParsedCompareRequest:
            return ParsedCompareRequest(
                topic="人工智能教育政策",
                object_a="中国",
                object_b="日本",
                requested_dimensions=["政策目标", "实施路径", "教师培养", "评价机制"],
                output_language="ZH",
            )
        return await super().structured(schema=schema, **kwargs)


async def _run_graph(graph: ZhijianGraph, knowledge_base_id: uuid.UUID):
    """执行不依赖数据库的完整智鉴图。

    Args:
        graph: 已注入固定检索和模型网关的智鉴图。
        knowledge_base_id: 固定知识库主键。

    Returns:
        LangGraph 最终状态。
    """
    return await graph.run(
        {
            "request_id": "m6-test",
            "user_id": str(uuid.uuid4()),
            "conversation_id": str(uuid.uuid4()),
            "query": "比较 2020 至 2025 年中国和日本的人工智能教育政策",
            "knowledge_base_id": str(knowledge_base_id),
            "knowledge_base_version": 7,
            "agent_route": [],
            "errors": [],
        }
    )


def test_zhijian_graph_assigns_balances_and_validates_citations(monkeypatch) -> None:
    """验证分批归属、双路去重、证据平衡和最终引用发布闸门。

    Args:
        monkeypatch: Pytest 属性替换夹具。
    """
    monkeypatch.setattr(zhijian_graph_module, "EVIDENCE_ASSIGNMENT_BATCH_SIZE", 2)
    knowledge_base_id = uuid.uuid4()
    retrieval = FakeZhijianRetrieval(knowledge_base_id)
    gateway = FakeZhijianGateway(retrieval)
    tracked_nodes: list[str] = []

    async def track_step(name, state, operation):
        """记录节点顺序并执行节点业务闭包。

        Args:
            name: 稳定节点名称。
            state: 当前图状态。
            operation: 节点业务闭包。

        Returns:
            节点产生的状态增量。
        """
        del state
        tracked_nodes.append(name)
        return await operation()

    graph = ZhijianGraph(retrieval, gateway, "固定智鉴 Prompt", track_step)
    final = asyncio.run(_run_graph(graph, knowledge_base_id))

    assert final["zhijian_result"].status == "SUCCESS"
    assert final["compare_object_a"].identifier == "CN"
    assert final["compare_object_b"].identifier == "JP"
    assert final["original_language"] == "zh"
    assert final["output_language"] == "ZH"
    assert final["retrieved_candidates"] == 15
    assert len(final["merged_candidates"]) == 4
    assert len(retrieval.queries) == 6
    assert all(
        sum(dimension in query for dimension in ["政策目标", "实施路径", "教师培养"]) == 1
        for query in retrieval.queries
    )
    assert {item["assignment"] for item in final["evidence_assignments"]} == {
        "A",
        "B",
        "SHARED",
    }
    assert len(final["evidence"]) == 3
    assert final["evidence_balance"].status == "SUFFICIENT"
    assert [item.status for item in final["evidence_balance"].dimension_coverage] == [
        "SUFFICIENT",
        "PARTIAL",
        "SUFFICIENT",
    ]
    assert len(final["evidence_assignments"][1]["retrieval_contexts"]) == 6
    assert final["citation_validation_status"] == "PASS"
    assert final["used_citation_indices"] == [1, 2, 3]
    assert [row.dimension for row in final["zhijian_result"].comparison_matrix] == [
        "政策目标",
        "实施路径",
        "教师培养",
    ]
    assert final["zhijian_result"].comparison_matrix[0].shared_citation_ids == []
    assert final["zhijian_result"].comparison_matrix[0].similarity == "证据不足"
    assert final["zhijian_result"].comparison_matrix[1].object_b == "证据不足"
    matrix_payload = final["zhijian_result"].comparison_matrix[0].model_dump(mode="json")
    assert "coverage_score" not in matrix_payload
    assert "coverage_level" not in matrix_payload
    assert [item.year for item in final["zhijian_result"].timeline] == [2022, 2023]
    assert [item.object for item in final["zhijian_result"].topic_distribution] == [
        "A",
        "B",
        "SHARED",
    ]
    assert [item.citation_id for item in final["zhijian_result"].references] == [
        1,
        2,
        3,
    ]
    assert "key_documents" not in final["zhijian_result"].model_dump(mode="json")
    assert final["intermediate_results"]["topic_count"] == 3
    assert "key_document_count" not in final["intermediate_results"]
    assert gateway.assignment_calls == 2
    assert tracked_nodes.index("ParseCompareRequest") < tracked_nodes.index(
        "ValidateCompareRequest"
    )
    assert "NormalizeCompareObjects" not in tracked_nodes
    assert tracked_nodes.index("EvidenceAssignment") < tracked_nodes.index("CrossAnalysis")
    assert tracked_nodes.index("CrossAnalysis") < tracked_nodes.index("CompareAnalysisValidation")
    assert "SimilarityAnalysis" not in tracked_nodes
    assert "DifferenceAnalysis" not in tracked_nodes
    assert "ResearchReviewer" not in tracked_nodes
    assert tracked_nodes.index("MatrixGenerator") < tracked_nodes.index("CitationValidator")


@pytest.mark.parametrize(
    ("dimension", "citation_ids", "expected_ids"),
    [
        pytest.param("政策目标", [1, 3], [1, 3], id="separate-sources"),
        pytest.param("政策目标", [3, 1, 3, 2, 99], [3, 1], id="filter-and-deduplicate"),
        pytest.param("教师培养", [2], [2], id="shared-source"),
        pytest.param("政策目标", [1], [], id="only-a"),
        pytest.param("政策目标", [3], [], id="only-b"),
        pytest.param("政策目标", [2], [], id="shared-source-wrong-dimension"),
        pytest.param("政策目标", [1, 2], [], id="b-support-in-wrong-dimension"),
        pytest.param("政策目标", [1, 99], [], id="unknown-source-cannot-cover-b"),
        pytest.param("政策目标", [], [], id="no-claim-sources"),
    ],
)
def test_zhijian_matrix_similarity_requires_combined_bilateral_cell_support(
    dimension: str, citation_ids: list[int], expected_ids: list[int]
) -> None:
    """共同点可综合分侧材料，但其引用集合必须在当前维度真实覆盖双方。

    Args:
        dimension: 共同点所在的比较维度。
        citation_ids: 模型为共同点实际提供的引用集合。
        expected_ids: 经过双侧单元格校验后应发布的共同点引用。
    """
    knowledge_base_id = uuid.uuid4()
    retrieval = FakeZhijianRetrieval(knowledge_base_id)
    gateway = FakeZhijianGateway(retrieval)
    original_structured = gateway.structured
    similarity = (
        "双方均重视教师能力建设。" if dimension == "教师培养"
        else "双方均关注人工智能教育的课程内容。"
    )

    async def structured_with_similarity(*, schema, **kwargs):
        """只替换共同点的引用组合，其余阶段执行完整的固定检索与归属流程。"""
        result = await original_structured(schema=schema, **kwargs)
        if schema is ZhijianCoreAnalysisDraft:
            row = ComparisonMatrixDraftRow(
                dimension=dimension,
                object_a="重视教师能力建设。" if dimension == "教师培养" else "强调课程体系建设。",
                object_b=(
                    "重视教师能力建设。" if dimension == "教师培养" else "强调数理数据教育衔接。"
                ),
                similarity=similarity,
                difference="证据不足",
                object_a_citation_ids=[2] if dimension == "教师培养" else [1],
                object_b_citation_ids=[2] if dimension == "教师培养" else [3],
                shared_citation_ids=citation_ids,
            )
            return result.model_copy(update={"comparison_matrix": [row]})
        return result

    gateway.structured = structured_with_similarity
    graph = ZhijianGraph(retrieval, gateway, "固定智鉴 Prompt", _budget_step)
    final = asyncio.run(_run_graph(graph, knowledge_base_id))
    result = final["zhijian_result"]
    row = next(item for item in result.comparison_matrix if item.dimension == dimension)

    assert result.status == "SUCCESS"
    assert final["citation_validation_status"] == "PASS"
    assert row.shared_citation_ids == expected_ids
    assert row.similarity == (similarity if expected_ids else "证据不足")
    assert set(expected_ids) <= {item.citation_id for item in result.references}
    # A/B 描述有来源也不能自动成为共同点的引用，缺引时仍须明确留空。
    assert row.object_a != "证据不足"
    assert row.object_b != "证据不足"


@pytest.mark.parametrize("recover", [True, False])
def test_zhijian_core_analysis_citation_retry_is_bounded(recover: bool) -> None:
    """核心分析越界时只重生成一次，第二轮仍越界则停止发布。

    Args:
        recover: 第二轮是否改为合法引用。
    """
    knowledge_base_id = uuid.uuid4()
    retrieval = FakeZhijianRetrieval(knowledge_base_id)

    class RetryGateway(FakeZhijianGateway):
        """按轮次模拟核心分析的越界引用和纠正输出。"""

        def __init__(self) -> None:
            """初始化模拟网关并记录核心分析输入。"""
            super().__init__(retrieval)
            self.analysis_instructions: list[str] = []

        async def structured(self, *, schema, **kwargs):
            """首轮返回越界共同点，第二轮按用例决定是否修正。

            Args:
                schema: 当前结构化输出模型。
                kwargs: 模型调用参数与纠错指令。

            Returns:
                标准或越界的结构化测试草稿。
            """
            result = await super().structured(schema=schema, **kwargs)
            if schema is not ZhijianCoreAnalysisDraft:
                return result
            self.analysis_instructions.append(kwargs["instruction"])
            if recover and len(self.analysis_instructions) == 2:
                return result
            return result.model_copy(
                update={
                    "similarities": [CitedFinding(text="双方均重视教师能力。", citation_ids=[99])]
                }
            )

    gateway = RetryGateway()
    tracked_nodes: list[str] = []
    attempts: list[int] = []
    decisions: list[dict[str, Any]] = []

    async def track_step(name, state, operation):
        """记录图级重入及校验路由，同时执行节点闭包。

        Args:
            name: 节点名称。
            state: 本轮图状态。
            operation: 节点业务闭包。

        Returns:
            节点状态增量。
        """
        tracked_nodes.append(name)
        if name == "CrossAnalysis":
            attempts.append(state.get("compare_analysis_attempts", 0))
        output = await operation()
        if name == "CompareAnalysisValidation":
            decisions.append(output)
        return output

    graph = ZhijianGraph(retrieval, gateway, "固定智鉴 Prompt", track_step)
    if recover:
        final = asyncio.run(_run_graph(graph, knowledge_base_id))
        assert final["zhijian_result"].status == "SUCCESS"
        assert final["compare_analysis_validation_status"] == "PASS"
    else:
        with pytest.raises(ValueError, match="共同点缺少 Citation 或引用编号越界"):
            asyncio.run(_run_graph(graph, knowledge_base_id))
        assert "CitationValidator" not in tracked_nodes
    assert attempts == [0, 1]
    assert [item["compare_analysis_validation_status"] for item in decisions] == ["RETRY"] + (
        ["PASS"] if recover else []
    )
    assert len(gateway.analysis_instructions) == 2
    assert (
        "上轮引用校验失败：共同点缺少 Citation 或引用编号越界" in gateway.analysis_instructions[1]
    )
    assert tracked_nodes.count("EvidenceAssignment") == 1
    assert tracked_nodes.count("CompareAnalysisValidation") == 2


@pytest.mark.parametrize("recover", [True, False])
@pytest.mark.parametrize("invalid_kind", ["coverage", "dimension"])
def test_zhijian_assignment_retry_is_bounded(recover: bool, invalid_kind: str) -> None:
    """归属候选或维度越界时只重做一次，失败不进入证据平衡。

    Args:
        recover: 第二轮是否完整返回候选。
        invalid_kind: 首轮违反候选覆盖还是维度边界。
    """
    knowledge_base_id = uuid.uuid4()
    retrieval = FakeZhijianRetrieval(knowledge_base_id)

    class RetryGateway(FakeZhijianGateway):
        """在每轮首批归属响应中模拟候选或维度边界错误。"""

        def __init__(self) -> None:
            """初始化固定检索器和归属失败记录。"""
            super().__init__(retrieval)
            self.failed_rounds: set[bool] = set()
            self.corrections: list[str] = []

        async def structured(self, *, schema, **kwargs):
            """每轮仅让一个批次违反指定归属边界。

            Args:
                schema: 当前结构化输出模型。
                kwargs: 模型指令及冻结 Prompt。

            Returns:
                正常响应或违反候选/维度边界的归属响应。
            """
            result = await super().structured(schema=schema, **kwargs)
            if schema is not EvidenceAssignmentBatch:
                return result
            correction = "上轮归属校验失败" in kwargs["instruction"]
            self.corrections.append(kwargs["instruction"])
            if (not correction or not recover) and correction not in self.failed_rounds:
                self.failed_rounds.add(correction)
                if invalid_kind == "coverage":
                    return EvidenceAssignmentBatch(decisions=result.decisions[:-1])
                invalid = result.decisions[0].model_copy(
                    update={
                        "supports": [
                            EvidenceCellSupport(
                                object="A", dimension="任务外维度", relevance="HIGH"
                            )
                        ]
                    }
                )
                return EvidenceAssignmentBatch(decisions=[invalid, *result.decisions[1:]])
            return result

    gateway = RetryGateway()
    tracked_nodes: list[str] = []
    attempts: list[int] = []

    async def track_step(name, state, operation):
        """记录归属重入和后续节点是否执行。

        Args:
            name: 节点名称。
            state: 当前图状态。
            operation: 节点业务闭包。

        Returns:
            节点状态增量。
        """
        tracked_nodes.append(name)
        if name == "EvidenceAssignment":
            attempts.append(state.get("evidence_assignment_attempts", 0))
        return await operation()

    graph = ZhijianGraph(retrieval, gateway, "固定智鉴 Prompt", track_step)
    if recover:
        final = asyncio.run(_run_graph(graph, knowledge_base_id))
        assert final["zhijian_result"].status == "SUCCESS"
        assert final["evidence_assignment_status"] == "PASS"
    else:
        failure = (
            "未逐项覆盖真实 Candidate Chunk"
            if invalid_kind == "coverage"
            else "返回了本次任务之外的比较维度"
        )
        with pytest.raises(ValueError, match=failure):
            asyncio.run(_run_graph(graph, knowledge_base_id))
        assert "EvidenceBalanceCheck" not in tracked_nodes
    assert attempts == [0, 1]
    assert tracked_nodes.count("EvidenceAssignment") == 2
    assert tracked_nodes.count("RetrieveObjectA") == 1
    assert tracked_nodes.count("RetrieveObjectB") == 1
    assert any("上轮归属校验失败" in item for item in gateway.corrections)


@pytest.mark.parametrize("recover", [True, False])
def test_zhijian_assignment_schema_retry_stays_in_failed_batch(recover: bool) -> None:
    """Schema 连续失败仅重发当前批一次，仍失败则不发布任何 Evidence。

    Args:
        recover: 第二次批次请求是否返回合法归属。
    """
    knowledge_base_id = uuid.uuid4()
    retrieval = FakeZhijianRetrieval(knowledge_base_id)

    class SchemaRetryGateway(FakeZhijianGateway):
        """模拟网关自身两轮 Schema 校验都失败的归属批次。"""

        def __init__(self) -> None:
            """记录当前批次请求及可审计的纠错提示。"""
            super().__init__(retrieval)
            self.batch_instructions: list[str] = []

        async def structured(self, *, schema, **kwargs):
            """首次或持续抛出受控 Schema 错误，其余节点使用真实形状结果。

            Args:
                schema: 当前结构化输出模型。
                kwargs: 当前节点的指令和固定 Prompt。

            Returns:
                恢复后符合归属 Schema 的批次或其他节点结果。

            Raises:
                StructuredSchemaValidationError: 当前归属批次保持非法。
            """
            if schema is EvidenceAssignmentBatch:
                self.batch_instructions.append(kwargs["instruction"])
                if len(self.batch_instructions) == 1 or not recover:
                    raise StructuredSchemaValidationError(
                        "EvidenceAssignmentBatch",
                        [
                            {
                                "attempt": 2,
                                "error_count": 1,
                                "issues": [
                                    {
                                        "path": "$.decisions[2]",
                                        "type": "assignment_relevant_without_supports",
                                    }
                                ],
                            }
                        ],
                    )
            return await super().structured(schema=schema, **kwargs)

    gateway = SchemaRetryGateway()
    tracked_nodes: list[str] = []

    async def track_step(name, state, operation):
        """记录图节点以确认失败批次不会推动后续证据发布。

        Args:
            name: 当前图节点名。
            state: 当前图状态。
            operation: 节点业务闭包。

        Returns:
            当前节点计算结果。
        """
        del state
        tracked_nodes.append(name)
        return await operation()

    graph = ZhijianGraph(retrieval, gateway, "固定智鉴 Prompt", track_step)
    if recover:
        final = asyncio.run(_run_graph(graph, knowledge_base_id))
        assert final["zhijian_result"].status == "SUCCESS"
        assert final["evidence_assignment_status"] == "PASS"
    else:
        with pytest.raises(StructuredSchemaValidationError):
            asyncio.run(_run_graph(graph, knowledge_base_id))
        assert "EvidenceBalanceCheck" not in tracked_nodes
    assert tracked_nodes.count("EvidenceAssignment") == 1
    assert len(gateway.batch_instructions) == 2
    assert "本批上次输出未通过结构与归属规则校验" in gateway.batch_instructions[1]
    assert len(retrieval.queries) == 6


def test_zhijian_rejects_more_than_three_requested_dimensions() -> None:
    """确认用户给出超过上限的维度时自然语言澄清且不执行检索。"""
    knowledge_base_id = uuid.uuid4()
    retrieval = FakeZhijianRetrieval(knowledge_base_id)
    gateway = TooManyDimensionsGateway(retrieval)

    async def track_step(name, state, operation):
        """执行节点闭包以获得完整受限结果。

        Args:
            name: 稳定节点名称。
            state: 当前图状态。
            operation: 节点业务闭包。

        Returns:
            节点产生的状态增量。
        """
        del name, state
        return await operation()

    graph = ZhijianGraph(retrieval, gateway, "固定智鉴 Prompt", track_step)
    final = asyncio.run(_run_graph(graph, knowledge_base_id))

    assert final["zhijian_result"].status == "NEEDS_CLARIFICATION"
    assert final["compare_object_a"].identifier == "CN"
    assert final["compare_object_b"].identifier == "JP"
    assert "最多支持 3 个" in final["zhijian_result"].clarification_question
    assert retrieval.queries == []


def test_shared_evidence_does_not_replace_object_b_specific_evidence() -> None:
    """确认 B 侧专属证据为零时直接返回证据不足且不调用 CrossAnalysis。"""
    knowledge_base_id = uuid.uuid4()
    retrieval = FakeZhijianRetrieval(knowledge_base_id)
    gateway = SharedOnlyObjectBGateway(retrieval)
    tracked_nodes: list[str] = []

    async def track_step(name, state, operation):
        """记录节点顺序并执行节点业务闭包。

        Args:
            name: 稳定节点名称。
            state: 当前图状态。
            operation: 节点业务闭包。

        Returns:
            节点产生的状态增量。
        """
        del state
        tracked_nodes.append(name)
        return await operation()

    graph = ZhijianGraph(retrieval, gateway, "固定智鉴 Prompt", track_step)
    final = asyncio.run(_run_graph(graph, knowledge_base_id))

    assert final["evidence_balance"].object_b_evidence_count == 0
    assert final["evidence_balance"].shared_evidence_count == 2
    assert final["zhijian_result"].status == "INSUFFICIENT_EVIDENCE"
    assert "日本" in final["zhijian_result"].summary
    assert "CrossAnalysis" not in tracked_nodes


async def _budget_step(_name, _state, operation):
    """直接执行证据预算测试的真实节点闭包，不连接模型或数据库。"""
    return await operation()


def _budget_candidate(
    index: int, side: str, dimension: str, *, relevance: str = "HIGH", score: float = 0.9,
) -> tuple[dict[str, Any], EvidenceAssignmentDecision]:
    """构造带原文、召回轨迹和独立语义归属的候选，供扩容测试复用。

    Args:
        index: 唯一测试编号，用于构造 Chunk 与文献主键。
        side: 真实语义归属 A、B 或 SHARED；召回轨迹故意统一为 A。
        dimension: 直接支持的本次比较维度。
        relevance: 模型判定相关等级；IRRELEVANT 不声明任何支持项。
        score: 检索重排分，用于验证较弱侧不会被高分一侧完全挤出。

    Returns:
        固定候选及经过实际 Schema 校验的逐 Chunk 归属判断。
    """
    evidence = _evidence(
        chunk_id=uuid.UUID(int=index + 1), document_id=uuid.UUID(int=index + 1000),
        title=f"政策研究文献{index}", year=2024, content=f"第{index}块政策研究原文证据。",
    ).model_copy(update={"relevance_score": score})
    supports = [] if relevance == "IRRELEVANT" else [
        EvidenceCellSupport(object=key, dimension=dimension, relevance=relevance)
        for key in (("A", "B") if side == "SHARED" else (side,))
    ]
    return {
        "evidence": evidence,
        "citation": _citation(evidence, uuid.UUID(int=9)),
        "retrieval_contexts": [CompareRetrievalContext(
            query_id="A:1", object="A", dimension=dimension, relevance_score=score,
        )],
    }, EvidenceAssignmentDecision(
        chunk_id=evidence.chunk_id, supports=supports, relevance=relevance,
        assignment_reason="根据测试原文判定真实对象与维度支持",
    )


def test_zhijian_requests_ten_per_query_and_merges_forty_unique_candidates() -> None:
    """六次实际检索显式请求十块，跨查询去重后至多四十块并保留弱维度召回轨迹。"""
    knowledge_base_id = uuid.uuid4()
    dimensions = ["政策目标", "实施路径", "教师培养"]
    queries = [CompareQuery(
        query_id=f"{side}:{i + 1}", object=side, dimension=dimension, text=f"{side}:{i + 1}",
    ) for side in ("A", "B") for i, dimension in enumerate(dimensions)]
    groups = {}
    for i, query in enumerate(queries):
        common, _ = _budget_candidate(0, "SHARED", query.dimension, score=0.7 + i * 0.05)
        unique = [_budget_candidate(
            i * 9 + j + 1, query.object, query.dimension,
            score=0.99 - j * 0.001 if query.object == "A" else 0.2 - i * 0.01 - j * 0.001,
        )[0] for j in range(9)]
        groups[query.text] = [common, *unique]

    class BudgetRetrieval:
        """为六个查询各返回十个候选，公共 Chunk 在每个查询中重复出现。"""

        def __init__(self):
            """记录实际检索调用参数，以验证只扩容智鉴的返回预算。"""
            self.calls = []

        async def search(self, **kwargs):
            """返回固定快照下的十块证据，不执行外部召回。"""
            self.calls.append(kwargs)
            items = [item["evidence"] for item in groups[kwargs["query"]]]
            return RetrievalResponse(
                knowledge_base_id=knowledge_base_id, knowledge_base_version=7,
                original_query=kwargs["query"], detected_language="zh",
                rewritten_query=kwargs["query"], expanded_queries={}, candidate_count=60,
                evidence=items, citations=[_citation(item, knowledge_base_id) for item in items],
                insufficient_evidence=False, message=None,
            )

    retrieval = BudgetRetrieval()
    graph = ZhijianGraph(retrieval, object(), "固定智鉴 Prompt", _budget_step)
    state = {
        "knowledge_base_id": str(knowledge_base_id), "knowledge_base_version": 7,
        "compare_request": ParsedCompareRequest(topic="教育政策", output_language="ZH"),
        "compare_queries": queries,
    }
    state["object_a_retrieval"] = asyncio.run(graph._retrieve_branch(state, "A"))
    state["object_b_retrieval"] = asyncio.run(graph._retrieve_branch(state, "B"))
    merged = asyncio.run(graph._merge_candidates(state))
    assert len(retrieval.calls) == 6
    assert all(call["mode"] == "FAST" and call["evidence_limit"] == 10 for call in retrieval.calls)
    assert merged["retrieved_candidates"] == 360
    candidates = merged["merged_candidates"]
    assert len(candidates) == len({item["evidence"].chunk_id for item in candidates}) == 40
    common = next(item for item in candidates if item["evidence"].chunk_id == uuid.UUID(int=1))
    assert len(common["retrieval_contexts"]) == 6
    assert common["evidence"].relevance_score == pytest.approx(0.95)
    for query in queries[3:]:
        assert any(
            len(item["retrieval_contexts"]) == 1
            and item["retrieval_contexts"][0].query_id == query.query_id
            for item in candidates
        )


@pytest.mark.parametrize("valid_count", [0, 7, 38])
def test_zhijian_filters_forty_candidates_before_selecting_thirty_and_numbering(
    valid_count,
) -> None:
    """四十块均须完成归属；有效证据不足不凑数，扩容时保留低分 B 侧并对齐引用。"""
    dimensions = ["政策目标", "实施路径", "教师培养"]
    pairs = [_budget_candidate(
        i, "A" if i < 35 else "B", dimensions[i % 3],
        relevance="IRRELEVANT" if i >= valid_count else "HIGH" if i < 35 else "LOW",
        score=0.99 - i * 0.001 if i < 35 else 0.1,
    ) for i in range(40)]
    decisions = {decision.chunk_id: decision for _, decision in pairs}

    class AssignmentGateway:
        """逐批返回固定语义判断，并记录是否处理完整四十块候选。"""

        def __init__(self):
            """保存批次记录，核对扩容后的模型输入覆盖。"""
            self.batches = []

        async def structured(self, *, schema, instruction, **_kwargs):
            """只返回本批输入 UUID 对应的经过 Schema 校验的归属决定。"""
            assert schema is EvidenceAssignmentBatch
            blocks = json.loads(instruction.split("\n候选：", 1)[1])
            self.batches.append(blocks)
            return EvidenceAssignmentBatch(
                decisions=[decisions[uuid.UUID(item["chunk_id"])] for item in blocks])

    gateway = AssignmentGateway()
    graph = ZhijianGraph(object(), gateway, "固定智鉴 Prompt", _budget_step)
    state = {
        "merged_candidates": [candidate for candidate, _ in pairs],
        "compare_dimensions": dimensions,
        "compare_request": ParsedCompareRequest(topic="教育政策", output_language="ZH"),
        "compare_object_a": normalize_compare_object("中国"),
        "compare_object_b": normalize_compare_object("日本"),
        "output_language": "ZH",
    }
    result = asyncio.run(graph._assign_evidence(state))
    assert len(gateway.batches) == 10
    assert {item["chunk_id"] for batch in gateway.batches for item in batch} == {
        str(item.chunk_id) for item in decisions.values()
    }
    expected_count = min(valid_count, 30)
    assert (len(result["evidence"]) == len(result["citations"])
            == len(result["evidence_assignments"]) == expected_count)
    assert result["evidence_assignment_status"] == "PASS"
    for index, (evidence, citation, assignment) in enumerate(zip(
        result["evidence"], result["citations"], result["evidence_assignments"], strict=True,
    ), start=1):
        assert evidence.chunk_id == citation.chunk_id == assignment["chunk_id"]
        assert (evidence.citation_index == citation.citation_index
                == assignment["citation_id"] == index)
        assert decisions[evidence.chunk_id].relevance != "IRRELEVANT"
    if valid_count == 38:
        # 三条较弱 B 侧实际支持不同维度，均不可被三十五条高分 A 侧挤出。
        assert {item["chunk_id"] for item in result["evidence_assignments"]
                if item["assignment"] == "B"} == {
            pairs[i][0]["evidence"].chunk_id for i in (35, 36, 37)
        }
        balance = asyncio.run(graph._check_balance({**state, **result}))["evidence_balance"]
        assert all(item.status == "SUFFICIENT" for item in balance.dimension_coverage)
        instruction = graph._analysis_instruction({**state, **result})
        assert '"citation_id": 30' in instruction
        assert result["evidence"][-1].content in instruction


def test_zhijian_final_budget_preserves_exclusive_documents_among_shared_evidence(
    monkeypatch,
) -> None:
    """共享材料占优时仍保留两侧专属证据及文献门槛，避免截断后制造证据不足。"""
    monkeypatch.setattr(zhijian_graph_module, "get_settings", lambda: SimpleNamespace(
        zhijian_min_side_evidence=2, zhijian_min_side_documents=2,
    ))
    dimensions = ["政策目标", "实施路径", "教师培养"]
    pairs = [_budget_candidate(
        i, "SHARED" if i < 32 else "A" if i < 36 else "B", dimensions[i % 3],
        relevance="HIGH" if i < 32 else "LOW", score=0.99 if i < 32 else 0.1,
    ) for i in range(40)]
    decisions = {decision.chunk_id: decision for _, decision in pairs}
    chosen = zhijian_graph_module._select_final_candidates(
        [item for item, _ in pairs], decisions, dimensions,
    )
    assert len(chosen) == 30
    for side in ("A", "B"):
        exclusive = [item for item in chosen if {
            support.object for support in decisions[item["evidence"].chunk_id].supports
        } == {side}]
        assert len(exclusive) >= 2
        assert len({item["evidence"].document_id for item in exclusive}) >= 2


def test_normalization_and_ambiguity_are_deterministic() -> None:
    """确认国家别名得到稳定 ISO 标识且关键歧义会触发追问。"""
    china = normalize_compare_object("China")
    japan = normalize_compare_object("日本")
    assert (china.identifier, china.canonical_name) == ("CN", "中国")
    assert (japan.identifier, japan.canonical_name) == ("JP", "日本")
    assert ambiguity_question("刚果") is not None


@pytest.mark.parametrize(
    "case",
    [case for case in json.loads(
        (Path(__file__).parent / "data" / "zhijian_gold.json").read_text(encoding="utf-8"),
    ) if case["category"] == "VALID_OBJECT_SCOPE"],
    ids=lambda case: case["id"],
)
def test_complete_object_scope_survives_validation_queries_and_analysis(case) -> None:
    """任意粒度的完整对象在规范化、分侧检索和分析任务中保持身份，不降格成上位实体。"""
    async def check():
        """以固定解析结果隔离模型波动，验证后续编排不会丢失对象限定或原始问题。"""
        parsed = ParsedCompareRequest(
            topic=case["topic"], object_a=case["object_a"], object_b=case["object_b"],
            requested_dimensions=["实施条件", "作用机制", "实际效果"], output_language="ZH",
        )
        gateway = SimpleNamespace(structured=AsyncMock(return_value=parsed))

        async def step(_name, _state, operation):
            """执行真实业务节点，避免依赖数据库或外部模型。"""
            return await operation()

        graph = ZhijianGraph(None, gateway, "固定智鉴 Prompt", step)
        state = {"query": case["query"]}
        state.update(await graph._parse_request(state))
        state.update(await graph._validate_request(state))
        assert not state["compare_block_status"]
        for side in ("a", "b"):
            obj = state[f"compare_object_{side}"]
            assert obj.raw == obj.canonical_name == case[f"object_{side}"]
        assert state["compare_object_a"].identifier != state["compare_object_b"].identifier
        state.update(await graph._generate_dimensions(state))
        state.update(await graph._build_queries(state))
        assert len(state["compare_queries"]) == 6
        for query in state["compare_queries"]:
            assert query.text.startswith(case[f"object_{query.object.lower()}"])
        state.update(evidence=[], evidence_assignments=[])
        instruction = graph._analysis_instruction(state)
        assert case["query"] in instruction
        assert f"对象A：{case['object_a']}" in instruction
        assert f"对象B：{case['object_b']}" in instruction

    asyncio.run(check())


@pytest.mark.parametrize(
    "case",
    [case for case in json.loads(
        (Path(__file__).parent / "data" / "zhijian_gold.json").read_text(encoding="utf-8"),
    ) if case["category"] == "VALID_SOURCE_LANGUAGE"],
    ids=lambda case: case["id"],
)
def test_foreign_source_languages_reach_retrieval_without_rejection(case) -> None:
    """旧解析字段中的外语要求应通过请求校验，并实际传给检索而非静默放宽。"""
    async def check():
        """用模型和检索替身核实来源语言限制从请求到检索的传递。"""
        parsed = ParsedCompareRequest(
            topic="教育政策", object_a=case["object_a"], object_b=case["object_b"],
            source_language_requirement=case["source_language"], output_language="ZH",
        )

        async def step(_name, _state, operation):
            """执行真实节点，隔离数据库及在线模型。"""
            return await operation()

        retrieval = SimpleNamespace(search=AsyncMock(return_value=SimpleNamespace(
            knowledge_base_version=1, evidence=[], citations=[], candidate_count=0,
            expanded_queries={},
        )))
        graph = ZhijianGraph(retrieval, None, "固定智鉴 Prompt", step)
        state = {
            "compare_request": parsed, "knowledge_base_id": str(uuid.uuid4()),
            "knowledge_base_version": 1,
        }
        state.update(await graph._validate_request(state))
        assert not state["compare_block_status"]
        state["compare_queries"] = [CompareQuery(
            query_id="A_1", object="A", dimension="政策目标", text=case["query"],
        )]
        await graph._retrieve_branch(state, "A")
        assert retrieval.search.call_args.kwargs["filters"].language_codes == [
            case["source_language"].lower(),
        ]

    asyncio.run(check())


@pytest.mark.parametrize(
    ("fields", "expected"),
    [
        ({}, []),
        ({"language_filter": "EN"}, ["en"]),
        ({"language_filter": "JA"}, ["ja"]),
        ({"source_language_requirement": "PT-BR"}, ["pt"]),
        ({"source_language_requirement": "IT"}, ["it"]),
        ({"source_language_codes": ["中文", "EN", "ja-JP", "jp"]}, ["zh", "en", "ja"]),
        ({"source_language_codes": ["FR"], "language_filter": "ZH"}, ["fr"]),
    ],
)
def test_source_filters_preserve_multilingual_and_legacy_requests(fields, expected) -> None:
    """多语种代码优先，旧单语种字段仍生效；未指定时保留不限制语言的行为。"""
    graph = ZhijianGraph(None, None, "固定智鉴 Prompt", None)
    filters = graph._filters({"compare_request": ParsedCompareRequest(
        topic="教育政策", output_language="ZH", year_from=2020, year_to=2025, **fields,
    )})
    assert filters.language_codes == expected
    assert (filters.year_from, filters.year_to) == (2020, 2025)


def test_zhijian_gold_dataset_has_required_coverage() -> None:
    """验证 Gold Dataset 数量、唯一性、国家规范化及完整对象粒度覆盖。"""
    dataset_path = Path(__file__).parent / "data" / "zhijian_gold.json"
    cases = json.loads(dataset_path.read_text(encoding="utf-8"))
    assert len(cases) >= 50
    assert len({case["id"] for case in cases}) == len(cases)
    categories = {case["category"] for case in cases}
    assert {"VALID_COUNTRY", "VALID_OBJECT_SCOPE", "MISSING_OBJECTS", "SAME_OBJECT"} <= categories
    assert {"AMBIGUOUS_OBJECT", "VALID_SOURCE_LANGUAGE", "INVALID_TIME"} <= categories
    for case in cases:
        if "expected_a" in case:
            assert normalize_compare_object(case["object_a"]).identifier == case["expected_a"]
        if "expected_b" in case:
            assert normalize_compare_object(case["object_b"]).identifier == case["expected_b"]
