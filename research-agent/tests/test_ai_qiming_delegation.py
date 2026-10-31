"""验证 AI 研究请求实际经过启明分类并委派溯知/智鉴，搜索仍走独立 SQL 分支。"""

import asyncio
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from test_ai_assistant import Gateway, graph_retrieval, graph_state

from app.agents.ai_assistant_graph import AssistantGraph
from app.agents.ai_assistant_schemas import AssistantIntent
from app.agents.qiming_graph import QimingGraph, build_intent_instruction
from app.agents.schemas import QimingIntentClassification
from app.agents.suzhi_graph import SuzhiGraph


@pytest.mark.parametrize(
    "intent, code, result_key",
    [
        ("CONCEPT_TRACE", "SUZHI", "suzhi_result"),
        ("LITERATURE_REVIEW", "SUZHI", "suzhi_result"),
        ("COMPARE", "ZHIJIAN", "zhijian_result"),
    ],
)
def test_ai_calls_real_qiming_classifier_and_preserves_delegated_result(intent, code, result_key):
    """真实启明图消费分类结果，专业调用保持同一会话及范围，最终 AI 快照保留路由。"""
    calls, steps = [], []

    class Classifier(Gateway):
        """分别提供入口意图和真实启明意图，若调用普通生成则立即使测试失败。"""

        async def structured(self, instruction, schema, **kwargs):
            """记录实际收到的 Schema，确认启明分类没有被预设值跳过。"""
            calls.append(schema)
            if schema is AssistantIntent:
                return schema(intent="QA", search_query="双减政策的发展")
            if schema is QimingIntentClassification:
                assert "双减政策的发展" in instruction
                return schema(intent=intent)
            return await super().structured(instruction, schema, **kwargs)

    async def tracked(name, state, operation):
        """记录入口和子图的实际节点，共用一个运行回调。"""
        steps.append(name)
        return await operation()

    gateway = Classifier("QA")
    gateway.generate_answer = AsyncMock(side_effect=AssertionError("不能回退普通问答"))
    result = {
        "final_answer": "专业流程已完成",
        result_key: {"summary": "专业结果"},
        "evidence": [],
        "citations": [],
        "validated_citations": [],
        "evidence_level": "INSUFFICIENT",
        "agent_route": ["QIMING", code],
    }
    specialist = SimpleNamespace(run=AsyncMock(return_value=result))
    specialist.graph = specialist.run
    retrieval = graph_retrieval([])
    qiming = QimingGraph(
        retrieval,
        gateway,
        "依据证据回答",
        tracked,
        suzhi_graph=specialist if code == "SUZHI" else None,
        zhijian_graph=specialist if code == "ZHIJIAN" else None,
    )
    original = graph_state(query="双减政策的发展")
    final = asyncio.run(
        AssistantGraph(retrieval, gateway, "依据证据回答", tracked, qiming_graph=qiming).run(
            original
        )
    )
    assert QimingIntentClassification in calls
    specialist.run.assert_awaited_once()
    delegated = specialist.run.call_args.args[0]
    for key in [
        "conversation_id",
        "request_id",
        "user_id",
        "knowledge_base_id",
        "knowledge_base_version",
    ]:
        assert delegated[key] == original[key]
    assert calls.count(QimingIntentClassification) == 1
    assert steps.count("IntentClassifier") == 2
    assert "RetrieveEvidence" not in steps
    assert final["final_answer"] == "专业流程已完成"
    payload = final["assistant_result"]
    assert payload["intent"] == "QA" and payload["research_intent"] == intent
    assert payload["agent_route"] == ["QIMING", code]
    assert payload["research_result"][result_key] == {"summary": "专业结果"}
    retrieval.search.assert_not_awaited()
    gateway.generate_answer.assert_not_called()


def test_policy_evolution_is_explicit_in_shared_intent_rules():
    """当前唯一提示词明确覆盖短政策发展问法，同时区分普通影响问答。"""
    instruction = build_intent_instruction("双减政策的发展")
    assert "'双减政策的发展'→CONCEPT_TRACE" in instruction
    assert "'双减政策的影响'→SIMPLE_QA" in instruction


def test_ai_runs_complete_qiming_and_suzhi_graphs_with_verified_citations():
    """执行真实 AI→启明→溯知三层图，使用隔离检索及模型数据验证完整发布结果。"""
    from test_suzhi import FakeSuzhiGateway, FakeSuzhiRetrieval

    steps = []

    class TraceGateway(FakeSuzhiGateway):
        """只补充入口及启明分类，其余模型输出复用溯知完整流程测试数据。"""

        async def structured(self, instruction, schema, **kwargs):
            """让实际分类节点返回溯源意图，其他请求交给原溯知测试网关。"""
            if schema is AssistantIntent:
                return schema(intent="QA", search_query="双减政策的发展")
            if schema is QimingIntentClassification:
                return schema(intent="CONCEPT_TRACE")
            return await super().structured(instruction=instruction, schema=schema, **kwargs)

    async def tracked(name, state, operation):
        """保留三层图的实际执行节点，不连接真实数据库。"""
        steps.append(name)
        return await operation()

    knowledge_id = uuid.uuid4()
    retrieval = FakeSuzhiRetrieval(knowledge_id)
    gateway = TraceGateway()
    suzhi = SuzhiGraph(retrieval, gateway, "依据证据回答", tracked)
    qiming = QimingGraph(retrieval, gateway, "依据证据回答", tracked, suzhi_graph=suzhi)
    state = graph_state(
        query="双减政策的发展", knowledge_base_id=str(knowledge_id), knowledge_base_version=7
    )
    final = asyncio.run(
        AssistantGraph(retrieval, gateway, "依据证据回答", tracked, qiming_graph=qiming).run(state)
    )
    assert steps.count("IntentClassifier") == 2
    assert "ConceptExtraction" in steps and "TopicClustering" in steps
    assert "ResearchReviewer" in steps and "CitationValidator" in steps
    assert retrieval.search_calls > 0
    assert final["validated_citations"] and final["final_answer"]
    payload = final["assistant_result"]
    assert payload["agent_route"] == ["QIMING", "SUZHI"]
    assert payload["research_intent"] == "CONCEPT_TRACE"
    assert payload["research_result"]["suzhi_result"]
    assert final["conversation_id"] == state["conversation_id"]
