"""隔离验证 AI 入口复用表、范围权限、路由和普通 RAG 引用，不调用在线服务。"""

import asyncio
import json
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from sqlalchemy import insert, select, update
from test_conversation_sources import client as research_client
from test_conversation_sources import store as research_store
from test_literature import citation, evidence

from app.agents.ai_assistant_graph import AssistantGraph
from app.agents.ai_assistant_schemas import AssistantIntent, AssistantSearchPlan
from app.agents.ai_assistant_state import AssistantState
from app.agents.gateway import ExtractiveModelGateway
from app.foreign_integration.schema import foreign_corpus, research_conversation, research_run
from app.foreign_integration.service import create_run


@pytest.mark.parametrize(("original", "rewritten", "language"), [
    ("帮我搜索孔子教育思想的中文文献", "孔子教育思想", "ZH"),
    ("找孔子教育思想相关的英文论文", "孔子教育思想", "EN"),
    ("请用中文介绍孔子教育思想相关论文", "孔子教育思想", None),
    ("查找孔子教育思想论文，不限语种", "孔子教育思想", None),
])
def test_search_extraction_keeps_original_constraints_available_to_model(
    original, rewritten, language,
):
    """条件抽取同时传入原始提问与改写主题，模型返回的来源语言进入搜索计划。"""
    async def structured(*, instruction, schema, **_kwargs):
        """核对模型实际接收的双问题上下文，不让检索改写替代用户原文。"""
        questions = json.loads(instruction.rsplit("问题JSON：", 1)[1])
        assert questions == {"original_question": original, "question": rewritten}
        return schema(concepts=[["孔子"], ["教育思想"]], language=language)

    async def step(_name, _state, operation):
        """执行真实抽取节点，不访问数据库或在线模型。"""
        return await operation()

    graph = AssistantGraph(None, SimpleNamespace(structured=structured), "测试提示词", step)
    result = asyncio.run(graph._extract_search_conditions({
        "query": original, "retrieval_query": rewritten,
    }))
    assert result["search_plan"]["language"] == language
    assert result["search_plan"]["concepts"] == [["孔子"], ["教育思想"]]


@pytest.fixture
def store():
    """复用现有研究内存库，不启动外部服务。"""
    yield from research_store.__wrapped__()


def start_ai(store, ids=None, conversation=None, request_id=None, source="AI", code="QIMING"):
    """创建使用已有研究表的全库或固定选文运行。"""
    return create_run(
        store.session,
        actor=store.actor,
        code=code,
        request_id=request_id or uuid.uuid4().hex,
        knowledge_base_id=None,
        query="查询双减相关的文献",
        conversation_id=conversation,
        mode="FAST",
        options={"corpus_ids": ids or []},
        entry_source=source,
    )


def test_global_scope_excludes_other_private_material(store):
    """全库不复制成员，但实际查询仍只读取公共文献及本人材料；重试保持幂等。"""
    store.session.execute(
        insert(foreign_corpus),
        [
            {"id": 12, "title_cn": "本人", "status": 4, "owner_id": 7, "is_deleted": False},
            {"id": 13, "title_cn": "他人", "status": 4, "owner_id": 8, "is_deleted": False},
        ],
    )
    store.session.commit()
    key = uuid.uuid4().hex
    accepted = start_ai(store, request_id=key)
    repeated = start_ai(store, request_id=key)
    assert repeated["run_id"] == accepted["run_id"] and not repeated["created"]
    from app.foreign_integration.schema import research_run_scope

    ids = set(store.session.scalars(select(research_run_scope.c.corpus_id)))
    assert ids == set()
    from app.foreign_integration.research_scope import ResearchScope

    assert set(ResearchScope("GLOBAL", 0).corpus_ids(
        store.session, 7, include_unindexed=True,
    )) == {11, 12}
    conversation = store.session.execute(select(research_conversation)).mappings().one()
    assert conversation["entry_source"] == "AI" and conversation["knowledge_base_id"] is None
    with pytest.raises(PermissionError):
        start_ai(store, ids=[13])
    store.session.rollback()


def test_pending_and_cross_entry_are_rejected(store):
    """同会话不能并发问答，也不能拿 AI 会话冒充写作会话。"""
    first = start_ai(store)
    with pytest.raises(ValueError, match="正在执行"):
        start_ai(store, conversation=first["conversation_id"])
    store.session.rollback()
    with pytest.raises(ValueError, match="入口"):
        start_ai(store, conversation=first["conversation_id"], source="WRITING")


def test_selected_writing_scope_cannot_change(store):
    """辅助创作可复用固定选文，但后续不能悄悄切换到全库。"""
    first = start_ai(store, ids=[11], source="WRITING", code="QUILL")
    store.session.execute(update(research_run).values(status="SUCCESS"))
    store.session.commit()
    with pytest.raises(ValueError, match="不能切换选文"):
        start_ai(store, conversation=first["conversation_id"], source="WRITING", code="QUILL")


class Gateway(ExtractiveModelGateway):
    """固定任务意图，回答复用离线网关，检验真实 Citation 流程。"""

    def __init__(self, intent, research_intent="SIMPLE_QA"):
        """保存本测试的任务意图。"""
        super().__init__()
        self.intent = intent
        self.research_intent = research_intent

    async def structured(self, instruction, schema, **kwargs):
        """返回限定任务或语言 Schema，不请求模型。"""
        if schema.__name__ == "QimingIntentClassification":
            return schema(intent=self.research_intent)
        if schema is AssistantIntent:
            return schema(intent=self.intent, search_query="双减教师工作负担")
        if schema is AssistantSearchPlan:
            return schema(concepts=[["双减"]])
        return schema(output_language="ZH")


@pytest.mark.parametrize("intent", ["SEARCH", "QA"])
def test_routes_and_citation_validation(intent):
    """查找与写作只返回候选，直接问答必须走引用校验再发布。"""
    item = evidence(1, 11)
    retrieval = SimpleNamespace(
        discover=AsyncMock(
            return_value={
                "documents": [
                    {
                        "corpus_id": 11,
                        "title": "双减",
                        "author": "",
                        "year": None,
                        "source": "",
                        "snippet": "",
                    }
                ],
                "total": 1,
                "document_index": {"ids": [11], "total": 1},
            }
        ),
        search=AsyncMock(
            return_value=SimpleNamespace(
                evidence=[item],
                citations=[citation(item)],
                knowledge_base_version=123,
                detected_language="zh",
                expanded_queries={},
                candidate_count=1,
                message=None,
            )
        ),
    )
    steps = []

    async def tracked(name, state, operation):
        """记录实际执行节点，直接执行测试闭包。"""
        steps.append(name)
        return await operation()

    graph = AssistantGraph(retrieval, Gateway(intent), "仅依据证据", tracked)
    final = asyncio.run(
        graph.run(
            AssistantState(
                request_id=uuid.uuid4().hex,
                user_id="7",
                conversation_id=uuid.uuid4().hex,
                query="双减问题",
                knowledge_base_id=str(uuid.UUID(int=0)),
                knowledge_base_version=123,
                scope_type="GLOBAL",
                mode="FAST",
                history=[],
            )
        )
    )
    assert final["intent"] == intent
    assert final["assistant_result"]["documents"][0]["corpus_id"] == 11
    if intent == "QA":
        assert "CitationValidator" in steps and final["validated_citations"]
    else:
        assert "GenerateAnswer" not in steps and not final["validated_citations"]
        assert final["assistant_result"]["document_index"] == {"ids": [11], "total": 1}
    assert "AuthenticateContext" in steps and "RoutingDecision" in steps
    assert "WritingHandoff" not in steps
    assert ("QimingQA" in steps) == (intent == "QA")
    assert steps[-1] == "Synthesizer"


@pytest.mark.parametrize(
    "intent, source, research_intent",
    [
        ("QA", "AI", "SIMPLE_QA"),
        ("SEARCH", "AI", "SIMPLE_QA"),
        ("QA", "DATABASE", "SIMPLE_QA"),
        ("QA", "AI", "CONCEPT_TRACE"),
    ],
)
def test_executor_publishes_existing_message_evidence_and_citation_tables(
    store, monkeypatch, intent, source, research_intent
):
    """完整运行穿过冻结、步骤、证据和发布事务，不仅检查图的返回值。"""
    import json

    from sqlalchemy.orm import Session

    from app.foreign_integration.literature import LiteratureMaterialTool
    from app.foreign_integration.retrieval import SourcePoint
    from app.foreign_integration.schema import (
        research_citation,
        research_evidence,
        research_message,
    )
    from app.foreign_integration.service import RunExecutor

    store.session.execute(
        update(foreign_corpus).values(
            title_cn="双减研究",
            attachment=json.dumps([{"file_path": "/test.pdf"}]),
        )
    )
    store.session.commit()
    accepted = start_ai(store, source=source)
    item = evidence(1, 11)

    class Retrieval(LiteratureMaterialTool):
        """只替代外部索引查询，保留真实选文和版本复核。"""

        async def search(self, **kwargs):
            """返回可回查来源并登记受控来源映射。"""
            self.sources[item.chunk_id] = SourcePoint(
                corpus_id=11,
                point_id="1",
                chunk_key="one",
                file_path="/test.pdf",
                content=item.content,
                page_start=1,
                page_end=1,
                score=1,
            )
            return SimpleNamespace(
                evidence=[item],
                citations=[citation(item)],
                knowledge_base_version=self.fixed_version,
                detected_language="zh",
                expanded_queries={},
                candidate_count=1,
                message=None,
            )

    def factory():
        """每个步骤使用独立短事务会话。"""
        return Session(store.engine)

    monkeypatch.setattr("app.foreign_integration.service.session_factory", lambda: factory)
    monkeypatch.setattr("app.foreign_integration.literature.session_factory", lambda: factory)
    monkeypatch.setattr("app.foreign_integration.service.LiteratureMaterialTool", Retrieval)
    monkeypatch.setattr(
        "app.foreign_integration.service.get_model_gateway",
        lambda *a, **k: Gateway(intent, research_intent),
    )

    if research_intent == "CONCEPT_TRACE":

        def delegated(executor, gateway):
            """替代专业模型输出，保留实际启明分类、检索来源和发布事务。"""

            async def trace(specialist_state):
                """登记受控测试来源，返回符合专业发布契约的结果。"""
                await executor.retrieval.search()
                return {
                    "final_answer": "政策溯源结果[1]",
                    "suzhi_result": {"summary": "政策溯源"},
                    "evidence": [item],
                    "citations": [citation(item)],
                    "validated_citations": [citation(item)],
                    "evidence_level": "LOW",
                    "agent_route": ["QIMING", "SUZHI"],
                }

            return {"SUZHI": SimpleNamespace(run=trace)}

        monkeypatch.setattr(RunExecutor, "_delegated_graphs", delegated)

    def fail(self, exc):
        """在测试中直接暴露异常，避免脱敏错误掩盖回归原因。"""
        raise exc

    monkeypatch.setattr(
        "app.foreign_integration.service.get_foreign_settings",
        lambda: SimpleNamespace(qdrant_collection="test", opensearch_index="foreign_chunks_v1"),
    )
    monkeypatch.setattr(RunExecutor, "_fail", fail)
    asyncio.run(RunExecutor(accepted["run_id"]).execute())
    store.session.expire_all()
    row = store.session.execute(select(research_run)).mappings().one()
    assert row["status"] == "SUCCESS", row["error_message"]
    if source == "AI":
        assert row["result_payload"]["result_type"] == "AI_ASSISTANT"
    else:
        assert row["result_payload"] is None
    if research_intent == "CONCEPT_TRACE":
        assert row["executor_agent_code"] == "SUZHI"
        assert row["result_payload"]["payload"]["research_intent"] == "CONCEPT_TRACE"
        assert row["result_payload"]["payload"]["research_result"]["suzhi_result"]
    # 即使启明或溯知参与执行，也只保存原来的一个 Run 和入口会话。
    assert len(list(store.session.scalars(select(research_run)))) == 1
    # 兼容旧 DATABASE 请求时统一存为 RESEARCH，专业委派不改变归一后的入口。
    expected_source = "RESEARCH" if source == "DATABASE" else source
    assert store.session.scalar(select(research_conversation.c.entry_source)) == expected_source
    assert len(list(store.session.scalars(select(research_message)))) == 2
    assert len(list(store.session.scalars(select(research_evidence)))) == (
        1 if intent == "QA" else 0
    )
    assert len(list(store.session.scalars(select(research_citation)))) == (
        1 if intent == "QA" else 0
    )


@pytest.fixture
def client(store):
    """复用绑定内存会话和身份的真实 HTTP 路由夹具。"""
    yield from research_client.__wrapped__(store)


def test_http_ai_creation_history_and_selected_writing_handoff(client, monkeypatch):
    """HTTP 入口无需知识库，历史独立，写作交接只需现有选文参数。"""

    async def skip_execution(run_id):
        """只验证 HTTP 契约和持久化，图执行另有完整事务测试。"""

    monkeypatch.setattr("app.foreign_integration.api._execute_run", skip_execution)
    payload = {
        "request_id": uuid.uuid4().hex,
        "entry_source": "AI",
        "query": "查询双减相关文献",
        "options": {},
    }
    response = client.post("/api/v1/agents/QIMING/runs", json=payload)
    assert response.status_code == 202, response.text
    accepted = response.json()["data"]
    repeated = client.post("/api/v1/agents/QIMING/runs", json=payload)
    assert repeated.json()["data"]["run_id"] == accepted["run_id"]
    history = client.get("/api/v1/conversations", params={"entry_source": "AI"}).json()["data"]
    assert history["total"] == 1
    detail = client.get(
        f"/api/v1/conversations/{accepted['conversation_id']}", params={"entry_source": "AI"}
    ).json()["data"]
    assert detail["assistant_runs"][0]["id"] == accepted["run_id"]
    assert (
        client.get("/api/v1/conversations", params={"entry_source": "RESEARCH"}).json()["data"][
            "total"
        ]
        == 0
    )
    invalid = client.post(
        "/api/v1/agents/QIMING/runs",
        json={
            **payload,
            "request_id": uuid.uuid4().hex,
            "options": {"corpus_ids": [True]},
        },
    )
    assert invalid.status_code == 422
    writing = client.post(
        "/api/v1/agents/QUILL/runs",
        json={
            "request_id": uuid.uuid4().hex,
            "entry_source": "WRITING",
            "query": "拟定双减研究问题",
            "options": {"corpus_ids": [11], "stage": "RQ"},
        },
    )
    assert writing.status_code == 202, writing.text


def graph_state(**overrides):
    """构造由服务层冻结的最小 AI 图上下文，允许各用例替换目标字段。"""
    state = AssistantState(
        request_id=uuid.uuid4().hex,
        user_id="7",
        conversation_id=uuid.uuid4().hex,
        query="双减问题",
        knowledge_base_id=str(uuid.UUID(int=0)),
        knowledge_base_version=123,
        scope_type="GLOBAL",
        mode="FAST",
        history=[],
    )
    state.update(overrides)
    return state


def graph_retrieval(items, citations=None, version=123):
    """只提供公共检索协议，不提供 Foreign 私有方法或数据库访问。"""
    return SimpleNamespace(
        discover=AsyncMock(return_value={"documents": [], "total": 0}),
        search=AsyncMock(
            return_value=SimpleNamespace(
                evidence=items,
                citations=citations if citations is not None else [citation(x) for x in items],
                knowledge_base_version=version,
                detected_language="zh",
                expanded_queries={},
                candidate_count=len(items),
                message=None,
            )
        ),
    )


async def untracked(name, state, operation):
    """单元测试直接执行节点闭包，持久化由完整运行测试单独覆盖。"""
    return await operation()


def test_ai_graph_has_independent_state_and_conditional_routes():
    """AI 入口有自己的状态和条件路由，不扩展启明 State 或调用启明图。"""
    from app.agents.qiming_state import QimingState

    graph = AssistantGraph(graph_retrieval([]), Gateway("SEARCH"), "证据优先", untracked)
    assert "assistant_result" in graph.graph.channels
    assert "history" in graph.graph.channels
    assert "assistant_result" not in QimingState.__annotations__
    assert "retrieval_query" not in QimingState.__annotations__
    edges = graph.graph.get_graph().edges
    assert any(
        edge.source == "RoutingDecision" and edge.target == "QimingQA" and edge.conditional
        for edge in edges
    )
    assert any(
        edge.source == "RoutingDecision"
        and edge.target == "ExtractSearchConditions"
        and edge.conditional
        for edge in edges
    )


@pytest.mark.parametrize("intent", ["QA", "SEARCH"])
def test_empty_evidence_never_generates_a_research_conclusion(intent):
    """零证据三条分支都返回明确说明，不使用模型记忆补全文献结论。"""
    gateway = Gateway(intent)
    gateway.generate_answer = AsyncMock(side_effect=AssertionError("不应调用生成模型"))
    graph = AssistantGraph(graph_retrieval([]), gateway, "证据优先", untracked)
    final = asyncio.run(graph.run(graph_state()))
    if intent == "QA":
        assert final["evidence_level"] == "INSUFFICIENT"
    assert not final["assistant_result"]["documents"]
    assert not final["validated_citations"]
    gateway.generate_answer.assert_not_called()


def test_changed_scope_fails_before_answer_generation():
    """范围版本不符必须终止图，不产生最终回答或调用回答生成。"""
    gateway = Gateway("QA")
    gateway.generate_answer = AsyncMock()
    graph = AssistantGraph(
        graph_retrieval([evidence(1, 11)], version=124), gateway, "证据优先", untracked
    )
    with pytest.raises(RuntimeError, match="版本.*变化"):
        asyncio.run(graph.run(graph_state()))
    gateway.generate_answer.assert_not_called()


@pytest.mark.parametrize("invalid", ["index", "quote", "document"])
def test_invalid_citation_cannot_reach_synthesizer(invalid):
    """编号、原文摘录或文献映射错误都在发布节点之前被拒绝。"""
    item = evidence(1, 11)
    source = citation(item)
    gateway = Gateway("QA")
    if invalid == "index":
        gateway.generate_answer = AsyncMock(return_value="错误结论[2]")
    elif invalid == "quote":
        source = source.model_copy(update={"quote": "原文没有的摘录"})
    else:
        source = source.model_copy(update={"document_id": uuid.UUID(int=12)})
    steps = []

    async def record(name, state, operation):
        """只记录实际执行节点，检查失败分支不会进入发布。"""
        steps.append(name)
        return await operation()

    graph = AssistantGraph(graph_retrieval([item], [source]), gateway, "证据优先", record)
    with pytest.raises(ValueError, match="引用|Citation|编号"):
        asyncio.run(graph.run(graph_state()))
    assert "Synthesizer" not in steps


def test_original_language_request_and_standalone_query_are_kept_separate():
    """检索使用补全问题，生成仍保留用户原始输出语言及其指代背景。"""
    retrieval = graph_retrieval([evidence(1, 11)])
    gateway = Gateway("QA")
    gateway.generate_answer = AsyncMock(return_value="Evidence supports the finding.[1]")
    graph = AssistantGraph(retrieval, gateway, "证据优先", untracked)
    final = asyncio.run(graph.run(graph_state(query="用英文解释第二篇的结论")))
    assert "双减教师工作负担" in retrieval.search.call_args.kwargs["query"]
    generated = gateway.generate_answer.call_args.kwargs["query"]
    assert "用英文解释第二篇的结论" in generated and "双减教师工作负担" in generated
    assert final["assistant_result"]["query"] == "用英文解释第二篇的结论"
