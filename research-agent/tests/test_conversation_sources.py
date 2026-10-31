"""按页面入口验证历史隔离、跨模块写作、幂等及旧表迁移，不连接业务数据库。"""

import uuid

import pytest
import test_foreign_integration as integration
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, insert, inspect, select, text, update

from app.foreign_integration.api import router
from app.foreign_integration.auth import current_actor
from app.foreign_integration.database import get_session
from app.foreign_integration.migrate import _ensure_conversation_sources
from app.foreign_integration.schema import (
    research_agent,
    research_conversation,
    research_prompt,
    research_run,
)
from app.foreign_integration.service import create_run


@pytest.fixture
def store():
    """复用共库测试数据，补齐写作和阅读 Agent。"""
    fixture = integration.ForeignIntegrationTests()
    fixture.setUp()
    for code in ("QUILL", "LITERATURE"):
        agent_id = uuid.uuid4().hex
        fixture.session.execute(insert(research_agent).values(
            id=agent_id, code=code, name=code, description=code, model_config_id="m" * 32,
            enabled=True, is_deleted=False,
        ))
        fixture.session.execute(insert(research_prompt).values(
            id=uuid.uuid4().hex, agent_id=agent_id, version=1,
            system_prompt="证据优先", status="ACTIVE", is_deleted=False,
        ))
    fixture.session.commit()
    yield fixture
    fixture.tearDown()


def start(store, code="QIMING", source=None, conversation_id=None, request_id=None):
    """从真实创建逻辑生成会话，不执行模型任务。"""
    return create_run(
        store.session, actor=store.actor, code=code,
        request_id=request_id or uuid.uuid4().hex,
        knowledge_base_id=None if code == "LITERATURE" else 9,
        query="研究问题", conversation_id=conversation_id, mode="FAST",
        options={"task": "ANALYZE", "corpus_ids": [11]} if code == "LITERATURE" else {},
        entry_source=source,
    )


@pytest.fixture
def client(store):
    """挂载真实历史接口，身份和连接限定为隔离内存库。"""
    app = FastAPI()

    @app.middleware("http")
    async def request_id(request, call_next):
        """补足响应信封的请求 ID。"""
        request.state.request_id = "test-conversation-source"
        return await call_next(request)

    app.include_router(router, prefix="/api/v1")
    app.dependency_overrides[current_actor] = lambda: store.actor
    app.dependency_overrides[get_session] = lambda: store.session
    with TestClient(app) as result:
        yield result


def test_writing_keeps_source_when_switching_between_qiming_and_quill(store):
    """先问启明再使用 Quill、再问启明，全程为同一写作会话。"""
    first = start(store, source="WRITING")
    conversation_id = first["conversation_id"]
    second = start(store, "QUILL", conversation_id=conversation_id)
    third = start(store, source="WRITING", conversation_id=conversation_id)
    assert first["run_id"] != second["run_id"] != third["run_id"]
    conversation = store.session.execute(select(research_conversation)).mappings().one()
    assert conversation["entry_source"] == "WRITING"
    assert conversation["database_id"] is None
    assert second["conversation_id"] == third["conversation_id"] == conversation_id


def test_api_creates_writing_conversation_even_when_executor_is_qiming(store, client, monkeypatch):
    """真实 HTTP 创建接口采用页面来源，而不把写作入口的启明请求归到研究。"""
    async def skip_execution(run_id):
        """只验证持久化接线，不启动外部检索和模型任务。"""

    monkeypatch.setattr("app.foreign_integration.api._execute_run", skip_execution)
    payload = {
        "request_id": uuid.uuid4().hex, "knowledge_base_id": 9,
        "query": "研究问题", "entry_source": "WRITING",
    }
    first = client.post("/api/v1/agents/QIMING/runs", json=payload)
    assert first.status_code == 202
    conversation_id = first.json()["data"]["conversation_id"]
    second = client.post("/api/v1/agents/QUILL/runs", json={
        **payload, "request_id": uuid.uuid4().hex, "conversation_id": conversation_id,
        "options": {"stage": "THEORY"},
    })
    assert second.status_code == 202
    conversation = store.session.execute(select(research_conversation)).mappings().one()
    assert conversation["entry_source"] == "WRITING"
    invalid = client.post("/api/v1/agents/QIMING/runs", json={
        **payload, "request_id": uuid.uuid4().hex, "entry_source": "OTHER",
    })
    assert invalid.status_code == 422


@pytest.mark.parametrize("initial,target", [("WRITING", "RESEARCH"), ("RESEARCH", "WRITING")])
def test_cannot_continue_or_replay_idempotent_request_from_another_source(store, initial, target):
    """同用户同知识库也不能跨入口继续会话或复用已有幂等运行。"""
    request_id = uuid.uuid4().hex
    first = start(store, source=initial, request_id=request_id)
    with pytest.raises(ValueError, match="不能切换入口"):
        start(store, source=target, conversation_id=first["conversation_id"])
    with pytest.raises(ValueError, match="request_id"):
        start(store, source=target, request_id=request_id)
    replay = start(store, source=initial, request_id=request_id)
    assert replay["created"] is False
    assert replay["run_id"] == first["run_id"]


@pytest.mark.parametrize("code,source", [
    ("QUILL", "RESEARCH"), ("LITERATURE", "WRITING"),
    ("ZHIJIAN", "WRITING"), ("QIMING", "LITERATURE"),
])
def test_incompatible_entry_and_agent_are_rejected(store, code, source):
    """入口值不能让写作/阅读 Agent 的直接请求混入研究会话。"""
    with pytest.raises(ValueError, match="入口与 Agent"):
        start(store, code, source)


def test_history_filters_before_pagination_and_rejects_wrong_source_detail(store, client):
    """只有启明运行的写作会话也能找回，三类历史分别查询并拒绝错入口恢复。"""
    research = start(store)
    writing = [start(store, source="WRITING") for _ in range(22)]
    literature = start(store, "LITERATURE")
    store.session.commit()
    response = client.get("/api/v1/conversations")
    assert response.status_code == 200
    assert [row["id"] for row in response.json()["data"]["items"]] == [research["conversation_id"]]
    found = []
    for page in (1, 2):
        response = client.get("/api/v1/conversations", params={
            "entry_source": "WRITING", "page": page,
        })
        items = response.json()["data"]["items"]
        assert len(items) == (20 if page == 1 else 2)
        assert all(item["entry_source"] == "WRITING" for item in items)
        found.extend(item["id"] for item in items)
    assert set(found) == {row["conversation_id"] for row in writing}
    reading = client.get("/api/v1/literature/conversations").json()["data"]
    assert reading["total"] == 1
    assert reading["items"][0]["id"] == literature["conversation_id"]
    detail_url = f"/api/v1/conversations/{writing[0]['conversation_id']}"
    assert client.get(detail_url, params={"entry_source": "RESEARCH"}).status_code == 404
    detail = client.get(detail_url, params={"entry_source": "WRITING"})
    assert detail.status_code == 200
    assert detail.json()["data"]["entry_source"] == "WRITING"
    store.session.execute(update(research_conversation).where(
        research_conversation.c.id == writing[0]["conversation_id"],
    ).values(user_id=8))
    assert client.get(detail_url, params={"entry_source": "WRITING"}).status_code == 404


def test_professional_history_does_not_include_writing_delegations(store, client):
    """即使两个入口最终均由溯知执行，研究历史仍只列研究入口运行。"""
    research = start(store)
    writing = start(store, source="WRITING")
    store.session.execute(update(research_run).values(executor_agent_code="SUZHI"))
    store.session.commit()
    for source, expected in (("RESEARCH", research), ("WRITING", writing)):
        response = client.get("/api/v1/agent-runs", params={
            "executor_agent_code": "SUZHI", "entry_source": source,
        })
        data = response.json()["data"]
        assert data["total"] == 1
        assert [item["id"] for item in data["items"]] == [expected["run_id"]]
    default = client.get("/api/v1/agent-runs", params={"executor_agent_code": "SUZHI"})
    assert default.json()["data"]["items"][0]["id"] == research["run_id"]


def test_legacy_schema_migration_backfills_sources_without_overwriting_classification():
    """在缺字段旧表上执行两次迁移，回填阅读/Quill，保留已分类数据及预留字段。"""
    engine = create_engine("sqlite://")
    with engine.begin() as connection:
        connection.exec_driver_sql(
            "CREATE TABLE research_conversation (id VARCHAR(32) PRIMARY KEY, "
            "user_id BIGINT, knowledge_base_id BIGINT, is_deleted BOOLEAN, updated_at DATETIME)"
        )
        connection.exec_driver_sql("CREATE TABLE research_agent (id VARCHAR(32), code VARCHAR(32))")
        connection.exec_driver_sql(
            "CREATE TABLE research_run (id VARCHAR(32), "
            "conversation_id VARCHAR(32), agent_id VARCHAR(32))"
        )
        connection.execute(text(
            "INSERT INTO research_conversation VALUES "
            "('research',7,9,0,CURRENT_TIMESTAMP), ('writing',7,9,0,CURRENT_TIMESTAMP), "
            "('literature',7,NULL,0,CURRENT_TIMESTAMP), ('qiming_only',7,9,0,CURRENT_TIMESTAMP), "
            "('old_writing',7,9,0,CURRENT_TIMESTAMP)"
        ))
        connection.execute(text(
            "INSERT INTO research_agent VALUES ('q','QIMING'),('w','QUILL'),('d','DIAOLONG')"
        ))
        connection.execute(text(
            "INSERT INTO research_run VALUES ('1','research','q'),('2','writing','w'),"
            "('3','writing','q'),('4','qiming_only','q'),('5','old_writing','d')"
        ))
    _ensure_conversation_sources(engine)
    with engine.begin() as connection:
        sources = dict(connection.execute(text(
            "SELECT id, entry_source FROM research_conversation"
        )).all())
        assert sources == {
            "research": "RESEARCH", "writing": "WRITING", "literature": "LITERATURE",
            "qiming_only": "RESEARCH", "old_writing": "WRITING",
        }
        connection.execute(text(
            "UPDATE research_conversation SET entry_source='WRITING', database_id=123 "
            "WHERE id='qiming_only'"
        ))
        connection.execute(text(
            "UPDATE research_conversation SET entry_source=NULL, "
            "knowledge_base_id=NULL, database_id=456 WHERE id='research'"
        ))
    _ensure_conversation_sources(engine)
    with engine.connect() as connection:
        assert connection.execute(text(
            "SELECT entry_source, database_id FROM research_conversation WHERE id='qiming_only'"
        )).one() == ("WRITING", 123)
        assert connection.execute(text(
            "SELECT entry_source, database_id FROM research_conversation WHERE id='research'"
        )).one() == ("RESEARCH", 456)
        assert len(inspect(connection).get_indexes("research_conversation")) == 1
    engine.dispose()
