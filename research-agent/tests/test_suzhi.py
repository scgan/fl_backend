"""覆盖溯知文献整理、主题溯源、研究空白和 Citation 发布闸门。"""

from __future__ import annotations

import asyncio
import json
import uuid
from pathlib import Path

from app.agents.gateway import ModelGateway
from app.agents.suzhi_graph import (
    SuzhiGraph,
    _fact_with_citations,
    _normalize_review_citations,
)
from app.agents.suzhi_schemas import (
    CitedReviewFinding,
    LiteratureAnalysisDraft,
    LiteratureAnalysisWithGraph,
    ProvenanceGraph,
    ProvenanceGraphEdge,
    ProvenanceGraphNode,
    ResearchGap,
    ReviewConcept,
    ReviewDraft,
    ReviewQueryPlan,
    SuzhiResult,
    SuzhiRunCreate,
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
        section="研究发现",
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


class FakeSuzhiRetrieval:
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
            original_query="人工智能与教育变革",
            detected_language="zh",
            rewritten_query="人工智能与教育变革",
            expanded_queries={"zh": "人工智能与教育变革", "en": "AI and education"},
            candidate_count=3,
            evidence=self.items,
            citations=[_citation(item, self.knowledge_base_id) for item in self.items],
            insufficient_evidence=False,
            message=None,
        )


class FakeSuzhiGateway(ModelGateway):
    """按溯知各结构化 Schema 返回固定、带引用的测试结果。"""

    async def generate_answer(self, **_kwargs) -> str:
        """满足网关抽象；溯知图不调用自由文本回答接口。"""
        return ""

    async def structured(self, *, schema, **_kwargs):
        """根据节点请求的 Schema 返回可验证固定值。"""
        if schema is ReviewConcept:
            return ReviewConcept(
                canonical_name="人工智能与教育变革",
                search_terms=["智能教育", "AI and education"],
                output_language="ZH",
            )
        if schema is ReviewQueryPlan:
            return ReviewQueryPlan(queries=["人工智能教育源流", "AI education methods"])
        if schema is LiteratureAnalysisWithGraph:
            finding = CitedReviewFinding(
                title="课堂应用",
                summary="研究由概念讨论转向课堂实践。[1][2]",
                citation_ids=[1, 2],
            )
            draft = LiteratureAnalysisDraft(
                concept_definition="人工智能与教育变革指智能技术参与教育系统变化。[1]",
                concept_citation_ids=[1],
                topics=[finding],
                theoretical_paths=[finding],
                methods=[finding],
                controversies=[
                    CitedReviewFinding(
                        title="应用与风险",
                        summary="应用扩展同时伴随风险讨论。[2][3]",
                        citation_ids=[2, 3],
                    )
                ],
                research_gaps=[
                    ResearchGap(
                        gap="长期课堂证据不足",
                        evidence_logic=(
                            "当前证据以概念与阶段性应用讨论为主，长期追踪尚少。[1][2][3]"
                        ),
                        citation_ids=[1, 2, 3],
                    )
                ],
                future_directions=[finding],
            )
            provenance_graph = ProvenanceGraph(
                title="人工智能与教育变革研究溯源图",
                nodes=[
                    ProvenanceGraphNode(
                        id="concept-origin",
                        label="智能教育概念",
                        subtitle="概念界定阶段",
                        summary="早期智能教育概念界定",
                        year=2018,
                        stage=0,
                        lane=0,
                        citation_ids=[1],
                    ),
                    ProvenanceGraphNode(
                        id="classroom-practice",
                        label="智能课堂实践",
                        subtitle="课堂应用阶段",
                        summary="研究转向课堂实践",
                        year=2021,
                        stage=1,
                        lane=0,
                        citation_ids=[1, 2],
                    ),
                ],
                edges=[
                    ProvenanceGraphEdge(
                        source="concept-origin",
                        target="classroom-practice",
                        relation="EVOLVES_TO",
                        label="演进为",
                        summary="研究从概念界定演进到课堂实践",
                        citation_ids=[1, 2],
                    )
                ],
            )
            return LiteratureAnalysisWithGraph(
                analysis_draft=draft, provenance_graph=provenance_graph
            )
        if schema is ReviewDraft:
            return ReviewDraft(
                review="研究由早期概念界定[1]，发展到课堂实践[2]与生成式人工智能风险讨论[3]。"
            )
        raise AssertionError(f"未处理的 Schema：{schema}")


class CitationlessReviewGateway(FakeSuzhiGateway):
    """模拟两轮模型正文都不输出 Citation 的真实失败模式。"""

    async def structured(self, *, schema, **kwargs):
        """仅让 ReviewDraft 连续缺少引用，其余结构化阶段沿用合法结果。

        Args:
            schema: 当前节点要求的 Pydantic 输出模型。
            kwargs: 传给父测试网关的其余结构化调用参数。

        Returns:
            无 Citation 的综述草稿，或父网关生成的合法结构化结果。
        """
        if schema is ReviewDraft:
            return ReviewDraft(review="研究经历了概念界定、课堂应用与风险讨论。")
        return await super().structured(schema=schema, **kwargs)


class RetrySuzhiGateway(FakeSuzhiGateway):
    """首轮分析和正文引用错误、第二轮恢复合法引用的图级回退网关。"""

    def __init__(self, *, always_invalid_analysis: bool = False) -> None:
        """设置是否让分析两轮都持续失败。

        Args:
            always_invalid_analysis: 为真时验证分析回路的失败上限。
        """
        # 仅计模型业务生成轮次，网关内部传输重试不在此测试范围。
        self.analysis_calls = 0
        self.review_calls = 0
        self.always_invalid_analysis = always_invalid_analysis

    async def structured(self, *, schema, **kwargs):
        """按轮次返回无效或有效输出，并保留其他节点的固定结果。

        Args:
            schema: 当前节点要求的结构化模型。
            kwargs: 透传给父测试网关的模型输入。

        Returns:
            用于触发对应图路由的结构化输出。
        """
        output = await super().structured(schema=schema, **kwargs)
        if schema is LiteratureAnalysisWithGraph:
            self.analysis_calls += 1
            if self.analysis_calls == 1 or self.always_invalid_analysis:
                bad_draft = output.analysis_draft.model_copy(
                    update={"concept_definition": "概念引用越界。[9]"}
                )
                return output.model_copy(update={"analysis_draft": bad_draft})
        if schema is ReviewDraft:
            self.review_calls += 1
            if self.review_calls == 1:
                return ReviewDraft(review="首轮正文遗漏了引用。")
        return output


def _initial_state(knowledge_base_id: uuid.UUID) -> dict[str, object]:
    """构造完整图回归所需的冻结溯知输入。

    Args:
        knowledge_base_id: 测试检索 Tool 使用的知识库 ID。

    Returns:
        可供 SuzhiGraph.run 使用的初始状态。
    """
    return {
        "request_id": "suzhi-retry-test",
        "user_id": str(uuid.uuid4()),
        "conversation_id": str(uuid.uuid4()),
        "query": "梳理人工智能与教育变革的研究演进",
        "research_topic": "人工智能与教育变革",
        "requested_concept": None,
        "year_from": None,
        "year_to": None,
        "output_language": "AUTO",
        "review_length": "SHORT",
        "citation_style": "GBT7714",
        "knowledge_base_id": str(knowledge_base_id),
        "knowledge_base_version": 7,
        "mode": "DEEP",
        "agent_route": [],
    }


def test_suzhi_keeps_thirty_unique_chunks_with_aligned_citations_and_analysis() -> None:
    """扩容后仍按最高相关性去重，第三十条来源进入文献整理和分析而不被再次截断。"""
    knowledge_base_id = uuid.uuid4()
    items = [
        _evidence(f"研究文献{index}", 2024, f"第{index}项研究的可验证事实。").model_copy(
            update={"relevance_score": 0.99 - index * 0.01}
        )
        for index in range(35)
    ]
    # 最低分块被另一查询以最高分重复召回，必须保留该分数且只占一个引用编号。
    repeated = items[-1].model_copy(update={"relevance_score": 1.0})
    raw = [*reversed(items), repeated]

    async def run_step(_name, _state, operation):
        """直接执行真实节点闭包，不调用模型或外部索引。"""
        return await operation()

    graph = SuzhiGraph(
        FakeSuzhiRetrieval(knowledge_base_id), FakeSuzhiGateway(), "固定溯知 Prompt", run_step
    )
    fixed = asyncio.run(graph._evidence_dedup({
        "raw_evidence": raw,
        "raw_citations": [_citation(item, knowledge_base_id) for item in raw],
    }))
    evidence = fixed["evidence"]
    assert len(evidence) == 30
    assert [item.chunk_id for item in evidence] == [
        items[-1].chunk_id, *[item.chunk_id for item in items[:29]]
    ]
    assert evidence[0].relevance_score == 1.0
    assert [item.citation_index for item in evidence] == list(range(1, 31))
    assert [(item.citation_index, item.chunk_id, item.document_id)
            for item in fixed["citations"]] == [
        (item.citation_index, item.chunk_id, item.document_id) for item in evidence
    ]
    documents = asyncio.run(graph._metadata_normalization(fixed))["documents"]
    assert len(documents) == 30
    instruction = graph._analysis_instruction({
        **fixed, "research_topic": "教育研究", "output_language": "ZH",
    })
    payload = json.loads(instruction.split("\n已确认 Evidence：", 1)[1])
    assert len(payload) == 30
    assert payload[-1]["citation_index"] == 30
    assert payload[-1]["chunk_id"] == str(evidence[-1].chunk_id)


def test_suzhi_graph_builds_grounded_review_without_stages() -> None:
    """验证节点顺序、去重、主题溯源与不包含分期的结果契约。"""
    knowledge_base_id = uuid.uuid4()
    tracked_nodes: list[str] = []

    async def track_step(name, _state, operation):
        """记录图节点顺序并执行节点闭包。"""
        tracked_nodes.append(name)
        return await operation()

    graph = SuzhiGraph(
        FakeSuzhiRetrieval(knowledge_base_id),
        FakeSuzhiGateway(),
        "固定溯知 Prompt",
        track_step,
    )
    final = asyncio.run(
        graph.run(
            {
                "request_id": "suzhi-test",
                "user_id": str(uuid.uuid4()),
                "conversation_id": str(uuid.uuid4()),
                "query": "梳理人工智能与教育变革的研究演进",
                "research_topic": "人工智能与教育变革",
                "requested_concept": None,
                "year_from": 2015,
                "year_to": 2025,
                "output_language": "AUTO",
                "review_length": "SHORT",
                "citation_style": "GBT7714",
                "knowledge_base_id": str(knowledge_base_id),
                "knowledge_base_version": 7,
                "mode": "DEEP",
                "agent_route": [],
            }
        )
    )

    result = final["suzhi_result"]
    assert result.status == "SUCCESS"
    assert len(final["evidence"]) == 3
    assert "stages" not in result.model_dump(mode="json")
    assert "timeline" not in result.model_dump(mode="json")
    assert "stage_count" not in result.evidence_summary
    legacy_payload = {
        **result.model_dump(mode="json"),
        "stages": [{"stage_name": "历史分期"}],
        "timeline": [{"year": 2024, "event": "历史节点"}],
    }
    restored = SuzhiResult.model_validate(legacy_payload)
    assert "stages" not in restored.model_dump(mode="json")
    assert "timeline" not in restored.model_dump(mode="json")
    assert result.research_gaps[0].evidence_logic
    assert result.provenance_graph is not None
    assert result.provenance_graph.edges[0].target == "classroom-practice"
    assert "key_documents" not in result.model_dump(mode="json")
    assert result.references[0].endswith("2018.")
    assert final["citation_validation_status"] == "PASS"
    assert final["used_citation_indices"] == [1, 2, 3]
    assert tracked_nodes[0] == "ConceptExtraction"
    assert tracked_nodes[-1] == "CitationValidator"
    assert len(tracked_nodes) == 10
    assert "ChronologicalSort" not in tracked_nodes
    assert "StageDetection" not in tracked_nodes
    assert tracked_nodes.index("TopicClustering") < tracked_nodes.index("AnalysisValidation")
    assert tracked_nodes.index("AnalysisValidation") < tracked_nodes.index("ReviewDraft")
    assert "ResearchMethodAnalysis" not in tracked_nodes
    assert "ControversyAnalysis" not in tracked_nodes
    assert "ResearchGapAnalysis" not in tracked_nodes
    assert "ProvenanceGraphBuild" not in tracked_nodes
    assert "ReferenceGraphBuild" not in tracked_nodes
    assert "KeyDocumentDetection" not in tracked_nodes
    assert tracked_nodes.index("LiteratureRetrieval") < tracked_nodes.index("ReviewDraft")


def test_suzhi_falls_back_to_validated_facts_when_review_has_no_citations() -> None:
    """模型连续漏掉 Citation 时应安全降级，而不是让完整 Run 失败。"""
    knowledge_base_id = uuid.uuid4()
    tracked_nodes: list[str] = []

    async def run_step(name, _state, operation):
        """记录每次图节点执行，不创建测试持久化步骤。"""
        tracked_nodes.append(name)
        return await operation()

    graph = SuzhiGraph(
        FakeSuzhiRetrieval(knowledge_base_id),
        CitationlessReviewGateway(),
        "固定溯知 Prompt",
        run_step,
    )
    final = asyncio.run(
        graph.run(
            {
                "request_id": "suzhi-citation-fallback",
                "user_id": str(uuid.uuid4()),
                "conversation_id": str(uuid.uuid4()),
                "query": "梳理人工智能与教育变革的研究演进",
                "research_topic": "人工智能与教育变革",
                "requested_concept": None,
                "year_from": None,
                "year_to": None,
                "output_language": "AUTO",
                "review_length": "MEDIUM",
                "citation_style": "GBT7714",
                "knowledge_base_id": str(knowledge_base_id),
                "knowledge_base_version": 7,
                "mode": "DEEP",
                "agent_route": ["QIMING", "SUZHI"],
            }
        )
    )

    assert final["review_fallback_used"] is True
    assert final["reviewer_status"] == "PASS"
    assert final["citation_validation_status"] == "PASS"
    assert _normalize_review_citations(final["final_answer"]) == final["final_answer"]
    assert {1, 2, 3} <= set(final["used_citation_indices"])
    assert "确定性组织综述" in final["suzhi_result"].limitations[-1]
    assert tracked_nodes.count("ReviewDraft") == 2
    assert tracked_nodes.count("ReviewFallback") == 1
    assert tracked_nodes.count("ResearchReviewer") == 3
    assert final["review_attempts"] == 2


def test_suzhi_retries_analysis_and_review_without_retrieving_again() -> None:
    """两类首轮校验失败只回到所属生成节点，并沿用同一批 Evidence。"""
    knowledge_base_id = uuid.uuid4()
    retrieval = FakeSuzhiRetrieval(knowledge_base_id)
    gateway = RetrySuzhiGateway()
    tracked: list[tuple[str, dict[str, object]]] = []

    async def run_step(name, _state, operation):
        """保留每轮规则判定的审计摘要。"""
        output = await operation()
        tracked.append((name, output))
        return output

    graph = SuzhiGraph(retrieval, gateway, "固定溯知 Prompt", run_step)
    final = asyncio.run(graph.run(_initial_state(knowledge_base_id)))
    names = [name for name, _output in tracked]
    assert final["suzhi_result"].status == "SUCCESS"
    assert final["analysis_attempts"] == 2
    assert final["review_attempts"] == 2
    assert final["review_fallback_used"] is False
    assert gateway.analysis_calls == 2
    assert gateway.review_calls == 2
    assert retrieval.search_calls == len(final["query_plan"].queries)
    assert names.count("LiteratureRetrieval") == 1
    assert names.count("TopicClustering") == 2
    assert names.count("AnalysisValidation") == 2
    assert "StageDetection" not in names
    assert "ChronologicalSort" not in names
    assert names.count("ReviewDraft") == 2
    assert names.count("ResearchReviewer") == 2
    assert "ReviewFallback" not in names
    analysis_statuses = [
        output["analysis_validation_status"]
        for name, output in tracked
        if name == "AnalysisValidation"
    ]
    reviewer_statuses = [
        output["reviewer_status"] for name, output in tracked if name == "ResearchReviewer"
    ]
    assert analysis_statuses == ["RETRY", "PASS"]
    assert reviewer_statuses == ["RETRY", "PASS"]


def test_suzhi_stops_after_second_invalid_analysis() -> None:
    """分析第二轮仍有越界引用时停止发布，不进入正文生成。"""
    knowledge_base_id = uuid.uuid4()
    retrieval = FakeSuzhiRetrieval(knowledge_base_id)
    gateway = RetrySuzhiGateway(always_invalid_analysis=True)
    names: list[str] = []

    async def run_step(name, _state, operation):
        """记录包括最终失败节点在内的执行顺序。"""
        names.append(name)
        return await operation()

    graph = SuzhiGraph(retrieval, gateway, "固定溯知 Prompt", run_step)
    try:
        asyncio.run(graph.run(_initial_state(knowledge_base_id)))
    except ValueError as exc:
        assert "概念定义包含缺失、错位或越界 Citation" in str(exc)
    else:
        raise AssertionError("两轮分析引用都不合法时必须失败")
    assert gateway.analysis_calls == 2
    assert gateway.review_calls == 0
    assert names.count("TopicClustering") == 2
    assert names.count("AnalysisValidation") == 2
    assert "ReviewDraft" not in names
    assert retrieval.search_calls > 0
    assert names.count("LiteratureRetrieval") == 1


def test_suzhi_request_rejects_reversed_year_range() -> None:
    """确认 API 契约拒绝反向年份范围。"""
    try:
        SuzhiRunCreate(
            request_id="suzhi-invalid-year",
            knowledge_base_id=uuid.uuid4(),
            research_topic="智能教育",
            year_from=2025,
            year_to=2020,
        )
    except ValueError as exc:
        assert "year_from" in str(exc)
    else:
        raise AssertionError("反向年份范围应被拒绝")


def test_suzhi_normalizes_full_width_citations() -> None:
    """外部模型使用全角括号时仍规范为可验证的真实 Citation 标记。"""
    assert _normalize_review_citations("观点【 1 】与结论［2］。") == "观点[1]与结论[2]。"


def test_suzhi_structured_schema_fields_have_model_descriptions() -> None:
    """溯知结构化输出及嵌套模型的字段都必须向模型说明业务语义。"""
    schemas = (
        ReviewConcept,
        ReviewQueryPlan,
        CitedReviewFinding,
        ResearchGap,
        LiteratureAnalysisDraft,
        LiteratureAnalysisWithGraph,
        ReviewDraft,
    )
    missing = [
        f"{schema.__name__}.{name}"
        for schema in schemas
        for name, field in schema.model_fields.items()
        if not field.description
    ]
    assert not missing


def test_suzhi_normalizes_internal_citations_in_place_without_trailing_summary() -> None:
    """内部 Citation 应原位转为标准编号，并删除旧版本追加的重复汇总。"""
    text = (
        "观点一[citation_id: 11]；观点二"
        "[citation_id: 3, citation_id: 5, citation_id: 15]。 [11][3][5][15]"
    )
    assert _normalize_review_citations(text) == "观点一[11]；观点二[3][5][15]。"

    partially_repeated = "已有研究指出限制[19]，这是明显空白。 [19][1][3]"
    assert _normalize_review_citations(partially_repeated) == "已有研究指出限制[19]，这是明显空白。"


def test_suzhi_analysis_instruction_requires_inline_citations_like_zhijian() -> None:
    """结构化分析 Prompt 应要求事实句原位使用 [n]，并同步返回编号数组。"""
    graph = object.__new__(SuzhiGraph)
    instruction = graph._analysis_instruction(
        {
            "research_topic": "人工智能与教育变革",
            "output_language": "ZH",
            "evidence": [_evidence("概念研究", 2020, "研究界定了核心概念。")],
        }
    )

    assert "每个事实结论必须在对应位置使用半角 [n]" in instruction
    assert "concept_citation_ids 必须列出正文实际使用的全部引用编号" in instruction
    assert "citation_ids 必须列出 summary 实际使用的全部引用编号" in instruction
    assert "不得输出 citation_id 字段名" in instruction


def test_suzhi_analysis_instruction_receives_all_fixed_evidence_and_graph_rules() -> None:
    """合并的分析 Prompt 应只发送一次完整 Evidence 并要求生成溯源图。"""
    graph = object.__new__(SuzhiGraph)
    first = _evidence("概念研究", 2020, "研究界定了核心概念。")
    second = _evidence("应用研究", 2022, "研究进一步讨论课堂应用。")
    second = second.model_copy(update={"citation_index": 2})
    instruction = graph._analysis_instruction(
        {
            "research_topic": "人工智能与教育变革",
            "output_language": "ZH",
            "evidence": [first, second],
        }
    )

    assert "已确认 Evidence" in instruction
    assert "研究界定了核心概念" in instruction
    assert "研究进一步讨论课堂应用" in instruction
    assert '"citation_index": 1' in instruction
    assert '"citation_index": 2' in instruction
    assert "生成 provenance_graph" in instruction
    assert instruction.count("研究界定了核心概念") == 1
    assert instruction.count("研究进一步讨论课堂应用") == 1


def test_suzhi_rejects_mismatched_inline_and_structured_citations() -> None:
    """分析校验应在综述生成前拒绝正文 [n] 与编号数组不一致的结论。"""

    async def run_step(_name, _state, operation):
        """直接执行分析校验闭包，使测试只覆盖 Citation 契约。"""
        return await operation()

    graph = object.__new__(SuzhiGraph)
    graph.step_runner = run_step
    evidence = _evidence("概念研究", 2020, "研究界定了核心概念。")
    draft = LiteratureAnalysisDraft(
        concept_definition="概念定义。[1]",
        concept_citation_ids=[1],
        topics=[
            CitedReviewFinding(
                title="主题",
                summary="主题结论。[1]",
                citation_ids=[1, 2],
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
                    "provenance_graph": ProvenanceGraph(),
                }
            )
        )
    except ValueError as exc:
        assert "缺失、错位或越界 Citation" in str(exc)
    else:
        raise AssertionError("正文 Citation 与 citation_ids 不一致时应拒绝发布")


def test_suzhi_rejects_out_of_range_provenance_citation() -> None:
    """分析校验应在综述生成前拒绝溯源图引用本 Run 之外的编号。"""

    async def run_step(_name, _state, operation):
        """直接执行分析校验闭包，使测试只覆盖溯源图 Citation 边界。"""
        return await operation()

    graph = object.__new__(SuzhiGraph)
    graph.step_runner = run_step
    evidence = _evidence("概念研究", 2020, "研究界定了核心概念。")
    draft = LiteratureAnalysisDraft(
        concept_definition="概念定义。[1]",
        concept_citation_ids=[1],
    )
    provenance_graph = ProvenanceGraph(
        title="概念演进",
        nodes=[
            ProvenanceGraphNode(
                id="concept-origin",
                label="概念起点",
                subtitle="概念界定",
                summary="形成核心概念",
                year=2020,
                stage=0,
                lane=0,
                citation_ids=[2],
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
                    "provenance_graph": provenance_graph,
                }
            )
        )
    except ValueError as exc:
        assert "溯源图包含越界 Citation" in str(exc)
    else:
        raise AssertionError("溯源图 Citation 越界时应拒绝发布")


def test_suzhi_rejects_out_of_range_provenance_edge_citation() -> None:
    """溯源图关系的越界引用也必须在正文生成前被拒绝。"""

    async def run_step(_name, _state, operation):
        """直接执行分析校验闭包，不写入测试步骤。"""
        return await operation()

    graph = object.__new__(SuzhiGraph)
    graph.step_runner = run_step
    evidence = _evidence("概念研究", 2020, "研究界定了核心概念。")
    draft = LiteratureAnalysisDraft(concept_definition="概念定义。[1]", concept_citation_ids=[1])
    provenance_graph = ProvenanceGraph(
        edges=[
            ProvenanceGraphEdge(
                source="origin",
                target="development",
                relation="EVOLVES_TO",
                label="演进",
                summary="研究路径继续发展",
                citation_ids=[2],
            )
        ]
    )

    try:
        asyncio.run(
            graph._analysis_validation(
                {
                    "evidence": [evidence],
                    "analysis_attempts": 2,
                    "analysis_draft": draft,
                    "provenance_graph": provenance_graph,
                }
            )
        )
    except ValueError as exc:
        assert "溯源图包含越界 Citation" in str(exc)
    else:
        raise AssertionError("溯源图关系引用越界时应在正文生成前失败")


def test_suzhi_rejects_mismatched_concept_citations_before_review() -> None:
    """概念定义由主题聚类生成，应在分析校验阶段拒绝编号错位。"""

    async def run_step(_name, _state, operation):
        """执行分析校验闭包，不创建测试持久化步骤。"""
        return await operation()

    graph = object.__new__(SuzhiGraph)
    graph.step_runner = run_step
    evidence = _evidence("概念研究", 2020, "研究界定了核心概念。")
    draft = LiteratureAnalysisDraft(
        concept_definition="概念定义。[2]",
        concept_citation_ids=[1],
    )

    try:
        asyncio.run(
            graph._analysis_validation(
                {
                    "evidence": [evidence],
                    "analysis_attempts": 2,
                    "analysis_draft": draft,
                    "provenance_graph": ProvenanceGraph(),
                }
            )
        )
    except ValueError as exc:
        assert "概念定义包含缺失、错位或越界 Citation" in str(exc)
    else:
        raise AssertionError("概念定义引用错位时应在正文生成前失败")


def test_suzhi_grounded_fact_preserves_existing_citation_positions() -> None:
    """确定性事实已有句级引用时不得把全部编号再次汇总到段末。"""
    text = "观点一[citation_id: 2]；观点二[citation_id: 1, citation_id: 5]。"
    assert _fact_with_citations(text, [2, 1, 5]) == "观点一[2]；观点二[1][5]。"


def test_suzhi_gold_dataset_has_required_coverage() -> None:
    """验证溯知 Gold Dataset 数量与核心验收类别。"""
    dataset = json.loads(
        (Path(__file__).parent / "data" / "suzhi_gold.json").read_text(encoding="utf-8")
    )
    assert len(dataset) >= 50
    assert len({item["id"] for item in dataset}) == len(dataset)
    assert {"chronology", "methods", "provenance", "gaps", "citation", "bilingual"} <= {
        item["category"] for item in dataset
    }
