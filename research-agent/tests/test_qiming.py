"""覆盖启明问答、Intent、SSE、Citation、版本固定与权限隔离验收。"""

from __future__ import annotations

import asyncio
import uuid

from app.agents.qiming_graph import QimingGraph
from app.agents.schemas import ParsedLanguagePreference, QimingIntentClassification
from app.retrieval.schemas import CitationRead, EvidenceRead, RetrievalResponse


def test_qiming_structured_schema_fields_have_model_descriptions() -> None:
    """启明结构化语言解析字段必须把业务语义写入 JSON Schema。"""
    fields = {
        **QimingIntentClassification.model_fields,
        **ParsedLanguagePreference.model_fields,
    }
    descriptions = {name: field.description for name, field in fields.items()}
    assert descriptions and all(descriptions.values())


def test_qiming_validates_only_citations_used_by_final_answer() -> None:
    """验证 Run Evidence 可保留全部来源，而消息发布只采用回答实际引用项。"""
    knowledge_base_id = uuid.uuid4()
    evidence_items: list[EvidenceRead] = []
    citation_items: list[CitationRead] = []
    for index in (1, 2):
        document_id = uuid.uuid4()
        chunk_id = uuid.uuid4()
        content = f"第 {index} 条测试证据。"
        evidence_items.append(
            EvidenceRead(
                citation_index=index,
                document_id=document_id,
                chunk_id=chunk_id,
                document_title=f"测试文献 {index}",
                authors=[],
                publication_year=2024,
                journal=None,
                language_code="zh",
                page_start=index,
                page_end=index,
                section="结果",
                content=content,
                relevance_score=0.9,
                retrieval_score=0.8,
                vector_rank=index,
                lexical_rank=index,
            )
        )
        citation_items.append(
            CitationRead(
                citation_index=index,
                document_id=document_id,
                chunk_id=chunk_id,
                document_title=f"测试文献 {index}",
                page_start=index,
                page_end=index,
                section="结果",
                quote=content,
                file_url=f"/test/{index}#page={index}",
            )
        )

    class FakeRetrievalTool:
        """向图返回两条已通过权限与质量门槛的测试证据。"""

        async def search(
            self,
            *,
            knowledge_base_id: uuid.UUID,
            query: str,
            mode: str,
        ) -> RetrievalResponse:
            """返回稳定双证据响应。

            Args:
                knowledge_base_id: 图固定的知识库主键。
                query: 用户测试问题。
                mode: FAST/DEEP 模式。

            Returns:
                包含两条 Evidence/Citation 的检索响应。
            """
            del query, mode
            return RetrievalResponse(
                knowledge_base_id=knowledge_base_id,
                knowledge_base_version=1,
                original_query="测试问题",
                detected_language="zh",
                rewritten_query="测试问题",
                expanded_queries={"zh": "测试问题", "en": "test question"},
                candidate_count=2,
                evidence=evidence_items,
                citations=citation_items,
                insufficient_evidence=False,
                message=None,
            )

    class SecondCitationGateway:
        """模拟只在最终回答中采用第二条 Evidence 的模型网关。"""

        async def structured(self, *, schema, **_kwargs):
            """返回普通问答意图或用户明确要求的双语输出偏好。"""
            if schema is QimingIntentClassification:
                return QimingIntentClassification(intent="SIMPLE_QA")
            assert schema is ParsedLanguagePreference
            return ParsedLanguagePreference(output_language="BILINGUAL")

        async def generate_answer(self, **kwargs) -> str:
            """返回只含 ``[2]`` 的回答并确认固定 Prompt 已传入。

            Args:
                kwargs: ModelGateway 统一关键字参数。

            Returns:
                仅引用第二条证据的回答。
            """
            assert kwargs["system_prompt"] == "固定 Prompt"
            assert kwargs["output_language"] == "BILINGUAL"
            return "最终结论只使用第二条证据。[2]"

    tracked_nodes: list[str] = []

    async def track_step(name, state, operation):
        """记录实际执行节点并直接返回节点增量。

        Args:
            name: 稳定节点名。
            state: 当前 Agent State。
            operation: 节点业务闭包。

        Returns:
            节点产生的状态增量。
        """
        del state
        tracked_nodes.append(name)
        return await operation()

    async def execute_graph():
        """执行一次无需数据库的完整基础问答图。

        Returns:
            图最终状态。
        """
        graph = QimingGraph(
            FakeRetrievalTool(),
            SecondCitationGateway(),
            "固定 Prompt",
            track_step,
        )
        return await graph.run(
            {
                "request_id": "request-citation-subset",
                "user_id": str(uuid.uuid4()),
                "conversation_id": str(uuid.uuid4()),
                "query": "测试问题",
                "output_language": "ZH",
                "knowledge_base_id": str(knowledge_base_id),
                "knowledge_base_version": 1,
                "mode": "FAST",
                "retry_count": 0,
                "retrieval_filters": {},
                "errors": [],
            }
        )

    final = asyncio.run(execute_graph())
    assert len(final["evidence"]) == 2
    assert final["output_language"] == "BILINGUAL"
    assert final["used_citation_indices"] == [2]
    assert [item.citation_index for item in final["validated_citations"]] == [2]
    assert (
        tracked_nodes.index("RetrieveEvidence")
        < tracked_nodes.index("ParseLanguagePreference")
        < tracked_nodes.index("GenerateAnswer")
    )
