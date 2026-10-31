"""覆盖雕龙修改完善（REVISE）阶段的改写边界与引用剥离契约。"""

from __future__ import annotations

import asyncio
import uuid

from app.agents.gateway import ModelGateway
from app.agents.quill_revise_graph import QuillReviseGraph
from app.agents.quill_schemas import (
    QuillReviseResult,
    RevisedText,
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


class FakeReviseRetrieval:
    """为每条扩展查询返回相同文献。"""

    def __init__(self, knowledge_base_id: uuid.UUID) -> None:
        """创建固定研究证据。"""
        self.knowledge_base_id = knowledge_base_id
        self.items = [_evidence("智能教育评价研究", 2024, "研究讨论了评价体系调适。")]

    async def search(self, **_kwargs) -> RetrievalResponse:
        """返回固定 KB Version 下的一条真实测试 Evidence。"""
        return RetrievalResponse(
            knowledge_base_id=self.knowledge_base_id,
            knowledge_base_version=7,
            original_query="学术修改",
            detected_language="zh",
            rewritten_query="学术修改",
            expanded_queries={"zh": "学术修改", "en": "revise"},
            candidate_count=1,
            evidence=self.items,
            citations=[_citation(item, self.knowledge_base_id) for item in self.items],
            insufficient_evidence=False,
            message=None,
        )


def _default_revised() -> RevisedText:
    """构造引用合法的修改后文本：旧编号被剥离、新事实带本运行引用。"""
    return RevisedText(
        revised_text=(
            "生成式人工智能通过降低低阶内容加工价值，倒逼评价重心从结果转向过程。[1]"
            "这一转移要求教师重建课堂测评素养。"
        )
    )


class FakeReviseGateway(ModelGateway):
    """按雕龙各结构化 Schema 返回固定修改文本。"""

    def __init__(self, draft: RevisedText | None = None) -> None:
        """允许替换修改文本并记录指令供审计断言。"""
        self.draft = draft or _default_revised()
        self.instructions: list[str] = []

    async def generate_answer(self, **_kwargs) -> str:
        """满足网关抽象；REVISE 图只使用结构化接口。"""
        return ""

    async def structured(self, *, instruction=None, schema, **_kwargs):
        """根据节点请求的 Schema 返回可验证固定值。"""
        from app.agents.quill_schemas import TheoryConcept, TheoryQueryPlan

        if schema is TheoryConcept:
            return TheoryConcept(
                canonical_name="学术修改",
                search_terms=["评价体系"],
                output_language="ZH",
            )
        if schema is TheoryQueryPlan:
            return TheoryQueryPlan(queries=["评价体系调适"])
        if schema is RevisedText:
            if instruction is not None:
                self.instructions.append(instruction)
            return self.draft.model_copy(deep=True)
        raise AssertionError(f"未处理的 Schema：{schema}")


def test_quill_revise_graph_strips_old_citations_and_regrounds() -> None:
    """REVISE 发布的文本剥离旧编号，新增事实使用本次真实引用。"""
    knowledge_base_id = uuid.uuid4()
    tracked_nodes: list[str] = []

    async def track_step(name, _state, operation):
        """记录图节点顺序并执行节点闭包。"""
        tracked_nodes.append(name)
        return await operation()

    gateway = FakeReviseGateway()
    graph = QuillReviseGraph(
        FakeReviseRetrieval(knowledge_base_id), gateway, "固定Quill Prompt", track_step
    )
    final = asyncio.run(
        graph.run(
            {
                "request_id": "quill-revise-test",
                "user_id": str(uuid.uuid4()),
                "conversation_id": str(uuid.uuid4()),
                # 待改文本带旧运行编号 [2][7]，模型必须剥离并用本运行 [1] 重新支撑。
                "query": "生成式人工智能倒逼评价重心转移。[2]教师需重建测评素养。[7]",
                "research_topic": "生成式人工智能与教育评价",
                "stage": "REVISE",
                "year_from": None,
                "year_to": None,
                "citation_style": "GBT7714",
                "revise_instruction": "补充实证文献支撑并规范引用",
                "output_language": "AUTO",
                "knowledge_base_id": str(knowledge_base_id),
                "knowledge_base_version": 7,
                "mode": "FAST",
                "agent_route": [],
            }
        )
    )

    result = final["revise_result"]
    assert isinstance(result, QuillReviseResult)
    assert result.status == "SUCCESS"
    assert result.stage == "REVISE"
    assert result.revise_instruction == "补充实证文献支撑并规范引用"
    assert result.evidence_level == "MEDIUM"
    assert final["citation_validation_status"] == "PASS"
    assert final["used_citation_indices"] == [1]
    assert final["intent"] == "REVISE"
    assert final["agent_route"] == ["QUILL"]
    # 修改指令进入模型输入，待改文本与剥离旧编号规则同时在场。
    assert gateway.instructions
    assert "补充实证文献支撑" in gateway.instructions[0]
    assert "一律删除不得照抄" in gateway.instructions[0]
    assert tracked_nodes[-1] == "CitationValidator"


def test_quill_revise_pure_edit_publishes_insufficient() -> None:
    """纯表达性修改没有新增事实引用时按 INSUFFICIENT 发布。"""

    async def run_step(_name, _state, operation):
        """执行节点闭包，不创建测试持久化步骤。"""
        return await operation()

    gateway = FakeReviseGateway(
        draft=RevisedText(revised_text="生成式人工智能倒逼评价重心转移，教师需重建素养。")
    )
    graph = QuillReviseGraph(
        FakeReviseRetrieval(uuid.uuid4()), gateway, "固定Quill Prompt", run_step
    )
    final = asyncio.run(
        graph.run(
            {
                "request_id": "quill-revise-pure",
                "user_id": str(uuid.uuid4()),
                "conversation_id": str(uuid.uuid4()),
                "query": "生成式人工智能倒逼评价重心转移。",
                "research_topic": "生成式人工智能与教育评价",
                "stage": "REVISE",
                "year_from": None,
                "year_to": None,
                "citation_style": "GBT7714",
                "revise_instruction": "语言更学术化",
                "output_language": "AUTO",
                "knowledge_base_id": str(uuid.uuid4()),
                "knowledge_base_version": 7,
                "mode": "FAST",
                "agent_route": [],
            }
        )
    )
    assert final["evidence_level"] == "INSUFFICIENT"
    assert final["validated_citations"] == []
    assert final["revise_result"].status == "SUCCESS"
