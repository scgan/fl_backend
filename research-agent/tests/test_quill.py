"""覆盖Quill THEORY 阶段的概念解析、理论综合、推荐裁剪和 Citation 发布闸门。"""

from __future__ import annotations

import asyncio
import uuid

from fastapi import HTTPException

from app.agents.gateway import ModelGateway
from app.agents.quill_graph import (
    QuillGraph,
    _normalize_theory_text,
)
from app.agents.quill_schemas import (
    CitedTheory,
    LiteratureRecommendation,
    QuillRunCreate,
    TheoryAnalysisDraft,
    TheoryConcept,
    TheoryGap,
    TheoryQueryPlan,
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
        section="理论探讨",
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


class FakeQuillRetrieval:
    """为每条扩展查询返回相同文献，用于验证跨查询按 Chunk 去重。"""

    def __init__(self, knowledge_base_id: uuid.UUID) -> None:
        """创建横跨三个年份的固定研究证据。"""
        self.knowledge_base_id = knowledge_base_id
        # 记录受控检索次数，用于验证业务重试不重复获取 Evidence。
        self.search_calls = 0
        self.items = [
            _evidence("人工智能教育的概念起点", 2018, "2018 年研究界定了智能教育概念。"),
            _evidence("人工智能支持教学实践", 2021, "2021 年研究转向课堂应用与教师实践。"),
            _evidence("生成式人工智能与教育变革", 2024, "2024 年研究讨论生成式人工智能风险。"),
        ]

    async def search(self, **_kwargs) -> RetrievalResponse:
        """返回固定 KB Version 下的三条真实测试 Evidence。"""
        self.search_calls += 1
        return RetrievalResponse(
            knowledge_base_id=self.knowledge_base_id,
            knowledge_base_version=7,
            original_query="智能教育理论依据",
            detected_language="zh",
            rewritten_query="智能教育理论依据",
            expanded_queries={"zh": "智能教育理论", "en": "AI education theory"},
            candidate_count=3,
            evidence=self.items,
            citations=[_citation(item, self.knowledge_base_id) for item in self.items],
            insufficient_evidence=False,
            message=None,
        )


def _default_draft() -> TheoryAnalysisDraft:
    """构造各分区引用合法且文献标题真实的理论综合草稿。"""
    return TheoryAnalysisDraft(
        theoretical_basis=[
            CitedTheory(
                theory="技术接受模型",
                description="该模型解释教师采纳智能技术的态度与意愿。[1][2]",
                citation_ids=[1, 2],
            ),
            CitedTheory(
                theory="建构主义学习理论",
                description="该理论支持技术介入下的知识建构过程。[3]",
                citation_ids=[3],
            ),
        ],
        literature_recommendations=[
            LiteratureRecommendation(
                document_title="人工智能教育的概念起点",
                relevance_reason="该文界定核心概念，可作为理论基础入口。[1]",
                citation_ids=[1],
            ),
            LiteratureRecommendation(
                document_title="人工智能支持教学实践",
                relevance_reason="该文讨论课堂应用，与理论演进直接相关。[2]",
                citation_ids=[2],
            ),
        ],
        research_gaps=[
            TheoryGap(
                gap="理论实证检验不足",
                evidence_logic="当前证据以概念讨论为主，缺少实证检验。[1][3]",
                citation_ids=[1, 3],
            )
        ],
    )


def _capped_draft() -> TheoryAnalysisDraft:
    """构造第二条推荐使用独占引用的草稿，用于验证推荐裁剪联动。"""
    return TheoryAnalysisDraft(
        theoretical_basis=[
            CitedTheory(
                theory="技术接受模型",
                description="该模型解释教师采纳智能技术的态度。[1]",
                citation_ids=[1],
            )
        ],
        literature_recommendations=[
            LiteratureRecommendation(
                document_title="人工智能教育的概念起点",
                relevance_reason="该文界定核心概念。[1]",
                citation_ids=[1],
            ),
            LiteratureRecommendation(
                document_title="人工智能支持教学实践",
                relevance_reason="该文讨论课堂应用。[2]",
                citation_ids=[2],
            ),
        ],
        research_gaps=[
            TheoryGap(
                gap="理论实证检验不足",
                evidence_logic="当前证据缺少实证检验。[3]",
                citation_ids=[3],
            )
        ],
    )


class FakeQuillGateway(ModelGateway):
    """按Quill各结构化 Schema 返回固定、带引用的测试结果。"""

    def __init__(self, draft: TheoryAnalysisDraft | None = None) -> None:
        """允许子类替换理论综合草稿，其余结构化输出保持合法。"""
        self.draft = draft or _default_draft()

    async def generate_answer(self, **_kwargs) -> str:
        """满足网关抽象；Quill THEORY 图不调用自由文本回答接口。"""
        return ""

    async def structured(self, *, schema, **_kwargs):
        """根据节点请求的 Schema 返回可验证固定值。"""
        if schema is TheoryConcept:
            return TheoryConcept(
                canonical_name="智能教育理论",
                search_terms=["智能教育", "AI education theory"],
                output_language="ZH",
            )
        if schema is TheoryQueryPlan:
            return TheoryQueryPlan(queries=["智能教育理论框架", "AI education theory"])
        if schema is TheoryAnalysisDraft:
            return self.draft.model_copy(deep=True)
        raise AssertionError(f"未处理的 Schema：{schema}")


class RetryQuillGateway(FakeQuillGateway):
    """首轮理论引用错误、第二轮恢复合法引用的图级回退网关。"""

    def __init__(self, *, always_invalid: bool = False) -> None:
        """设置是否让理论综合两轮都持续失败。

        Args:
            always_invalid: 为真时验证理论回路的失败上限。
        """
        super().__init__()
        # 仅计模型业务生成轮次，网关内部传输重试不在此测试范围。
        self.analysis_calls = 0
        self.always_invalid = always_invalid

    async def structured(self, *, schema, **kwargs):
        """按轮次返回无效或有效输出，并保留其他节点的固定结果。

        Args:
            schema: 当前节点要求的结构化模型。
            kwargs: 透传给父测试网关的模型输入。

        Returns:
            用于触发对应图路由的结构化输出。
        """
        output = await super().structured(schema=schema, **kwargs)
        if schema is TheoryAnalysisDraft:
            self.analysis_calls += 1
            if self.analysis_calls == 1 or self.always_invalid:
                bad_theory = output.theoretical_basis[0].model_copy(
                    update={"description": "理论引用越界。[9]"}
                )
                output = output.model_copy(
                    update={
                        "theoretical_basis": [bad_theory, *output.theoretical_basis[1:]]
                    }
                )
        return output


def _initial_state(knowledge_base_id: uuid.UUID, **overrides) -> dict[str, object]:
    """构造完整图回归所需的冻结Quill输入。

    Args:
        knowledge_base_id: 测试检索 Tool 使用的知识库 ID。
        **overrides: 覆盖默认冻结规格的字段。

    Returns:
        可供 QuillGraph.run 使用的初始状态。
    """
    state = {
        "request_id": "quill-test",
        "user_id": str(uuid.uuid4()),
        "conversation_id": str(uuid.uuid4()),
        "query": "为智能教育研究整理理论依据",
        "research_topic": "智能教育的理论基础",
        "stage": "THEORY",
        "year_from": None,
        "year_to": None,
        "output_language": "AUTO",
        "citation_style": "GBT7714",
        "literature_limit": 10,
        "knowledge_base_id": str(knowledge_base_id),
        "knowledge_base_version": 7,
        "mode": "DEEP",
        "agent_route": [],
    }
    state.update(overrides)
    return state


def test_quill_theory_graph_publishes_grounded_result() -> None:
    """验证 THEORY 节点顺序、去重、推荐与不越界的引用契约。"""
    knowledge_base_id = uuid.uuid4()
    tracked_nodes: list[str] = []

    async def track_step(name, _state, operation):
        """记录图节点顺序并执行节点闭包。"""
        tracked_nodes.append(name)
        return await operation()

    graph = QuillGraph(
        FakeQuillRetrieval(knowledge_base_id),
        FakeQuillGateway(),
        "固定Quill Prompt",
        track_step,
    )
    final = asyncio.run(graph.run(_initial_state(knowledge_base_id)))

    result = final["quill_result"]
    assert result.status == "SUCCESS"
    assert result.stage == "THEORY"
    assert len(final["evidence"]) == 3
    assert result.references[0].endswith("2018.")
    assert final["citation_validation_status"] == "PASS"
    assert final["used_citation_indices"] == [1, 2, 3]
    assert final["agent_route"] == ["QUILL"]
    assert final["intent"] == "THEORY"
    assert final["output_language"] == "ZH"
    assert result.academic_notice in final["final_answer"]
    assert "## 理论依据" in final["final_answer"]
    assert "## 推荐文献" in final["final_answer"]
    assert "## 参考文献" in final["final_answer"]
    assert "[1]" in final["final_answer"]
    assert result.evidence_level == "MEDIUM"
    assert result.evidence_summary["document_count"] == 3
    assert len(result.literature_recommendations) == 2
    assert tracked_nodes == [
        "ConceptExtraction",
        "QueryExpansion",
        "LiteratureRetrieval",
        "EvidenceDedup",
        "MetadataNormalization",
        "TheorySynthesis",
        "AnalysisValidation",
        "CitationValidator",
    ]


def test_quill_retries_invalid_citations_without_retrieving_again() -> None:
    """首轮引用越界只回到理论综合节点，并沿用同一批 Evidence。"""
    knowledge_base_id = uuid.uuid4()
    retrieval = FakeQuillRetrieval(knowledge_base_id)
    gateway = RetryQuillGateway()
    names: list[tuple[str, dict[str, object]]] = []

    async def run_step(name, _state, operation):
        """保留每轮规则判定的审计摘要。"""
        output = await operation()
        names.append((name, output))
        return output

    graph = QuillGraph(retrieval, gateway, "固定Quill Prompt", run_step)
    final = asyncio.run(graph.run(_initial_state(knowledge_base_id)))
    node_names = [name for name, _output in names]
    assert final["quill_result"].status == "SUCCESS"
    assert final["analysis_attempts"] == 2
    assert gateway.analysis_calls == 2
    assert retrieval.search_calls == len(final["query_plan"].queries)
    assert node_names.count("LiteratureRetrieval") == 1
    assert node_names.count("TheorySynthesis") == 2
    assert node_names.count("AnalysisValidation") == 2
    statuses = [
        output["analysis_validation_status"]
        for name, output in names
        if name == "AnalysisValidation"
    ]
    assert statuses == ["RETRY", "PASS"]


def test_quill_stops_after_second_invalid_analysis() -> None:
    """理论综合第二轮仍有越界引用时停止发布，不进入结果组装。"""
    knowledge_base_id = uuid.uuid4()
    retrieval = FakeQuillRetrieval(knowledge_base_id)
    gateway = RetryQuillGateway(always_invalid=True)
    names: list[str] = []

    async def run_step(name, _state, operation):
        """记录包括最终失败节点在内的执行顺序。"""
        names.append(name)
        return await operation()

    graph = QuillGraph(retrieval, gateway, "固定Quill Prompt", run_step)
    try:
        asyncio.run(graph.run(_initial_state(knowledge_base_id)))
    except ValueError as exc:
        assert "理论依据包含缺失、错位或越界 Citation" in str(exc)
    else:
        raise AssertionError("两轮理论引用都不合法时必须失败")
    assert gateway.analysis_calls == 2
    assert names.count("TheorySynthesis") == 2
    assert names.count("AnalysisValidation") == 2
    assert "CitationValidator" not in names
    assert retrieval.search_calls > 0
    assert names.count("LiteratureRetrieval") == 1


def test_quill_rejects_recommendation_outside_corpus() -> None:
    """文献推荐包含知识库外标题时必须在组装结果前被拒绝。"""

    async def run_step(_name, _state, operation):
        """直接执行分析校验闭包，使测试只覆盖文献真实性契约。"""
        return await operation()

    graph = object.__new__(QuillGraph)
    graph.step_runner = run_step
    evidence = _evidence("概念研究", 2020, "研究界定了核心概念。")
    draft = TheoryAnalysisDraft(
        literature_recommendations=[
            LiteratureRecommendation(
                document_title="知识库外的虚构文献",
                relevance_reason="推荐理由看似合法。[1]",
                citation_ids=[1],
            )
        ],
    )

    try:
        asyncio.run(
            graph._analysis_validation(
                {
                    "evidence": [evidence],
                    "analysis_attempts": 2,
                    "analysis_draft": draft,
                }
            )
        )
    except ValueError as exc:
        assert "知识库外的文献标题" in str(exc)
    else:
        raise AssertionError("推荐知识库外文献时应拒绝发布")


def test_quill_caps_literature_recommendations_and_unused_citations() -> None:
    """推荐裁剪联动引用投影：被裁掉的推荐不再占用 Citation 和参考文献。"""
    knowledge_base_id = uuid.uuid4()

    async def run_step(_name, _state, operation):
        """执行节点闭包，不创建测试持久化步骤。"""
        return await operation()

    graph = QuillGraph(
        FakeQuillRetrieval(knowledge_base_id),
        FakeQuillGateway(draft=_capped_draft()),
        "固定Quill Prompt",
        run_step,
    )
    final = asyncio.run(
        graph.run(_initial_state(knowledge_base_id, literature_limit=1))
    )
    result = final["quill_result"]
    assert len(result.literature_recommendations) == 1
    assert result.literature_recommendations[0].document_title == "人工智能教育的概念起点"
    assert final["used_citation_indices"] == [1, 3]
    assert final["citation_validation_status"] == "PASS"
    assert result.limitations
    assert "literature_limit=1" in result.limitations[0]
    # 只保留仍被引用的两篇文献的参考文献。
    assert len(result.references) == 2


def test_quill_normalizes_full_width_citations() -> None:
    """外部模型使用全角括号时仍规范为可验证的真实 Citation 标记。"""
    assert _normalize_theory_text("观点【 1 】与结论［2］。") == "观点[1]与结论[2]。"


def test_quill_request_rejects_reversed_year_range() -> None:
    """确认 API 契约拒绝反向年份范围。"""
    try:
        QuillRunCreate(
            request_id="quill-invalid-year",
            knowledge_base_id=uuid.uuid4(),
            research_topic="智能教育",
            year_from=2025,
            year_to=2020,
        )
    except ValueError as exc:
        assert "year_from" in str(exc)
    else:
        raise AssertionError("反向年份范围应被拒绝")


def test_quill_structured_schema_fields_have_model_descriptions() -> None:
    """Quill结构化输出及嵌套模型的字段都必须向模型说明业务语义。"""
    schemas = (
        TheoryConcept,
        TheoryQueryPlan,
        CitedTheory,
        LiteratureRecommendation,
        TheoryGap,
        TheoryAnalysisDraft,
    )
    missing = [
        f"{schema.__name__}.{name}"
        for schema in schemas
        for name, field in schema.model_fields.items()
        if not field.description
    ]
    assert not missing


def test_quill_options_validation() -> None:
    """Quill options 白名单只接受 THEORY 阶段的合法参数。"""
    from app.foreign_integration.api import _validate_options

    assert _validate_options("QUILL", {}) == {
        "stage": "THEORY",
        "citation_style": "GBT7714",
        "literature_limit": 10,
    }
    assert _validate_options(
        "QUILL",
        {"citation_style": "APA", "literature_limit": 20, "year_from": 2015, "year_to": 2025},
    ) == {
        "stage": "THEORY",
        "citation_style": "APA",
        "literature_limit": 20,
        "year_from": 2015,
        "year_to": 2025,
    }
    invalid_payloads = (
        {"unknown": 1},
        {"citation_style": "IEEE"},
        {"literature_limit": 4},
        {"literature_limit": 21},
        {"literature_limit": "10"},
        {"year_from": 2025, "year_to": 2015},
    )
    for payload in invalid_payloads:
        try:
            _validate_options("QUILL", payload)
        except HTTPException as exc:
            assert exc.status_code == 422
        else:
            raise AssertionError(f"应拒绝非法Quill参数：{payload}")


def test_quill_registered_in_catalog_and_service() -> None:
    """Quill必须同时进入运行目录和 Service 的合法 Agent 编码集合。"""
    from app.agents.catalog import AGENT_DEFINITIONS
    from app.foreign_integration.service import AGENT_CODES

    assert "QUILL" in AGENT_CODES
    assert "QUILL" in {item[0] for item in AGENT_DEFINITIONS}
