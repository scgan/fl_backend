"""数据库研究边界回归：根选项、有效子树、去重、来源隔离及执行期间的范围复核。"""

import asyncio
import hashlib
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, insert, select, update
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from app.common.config import Settings
from app.foreign_integration.api import router
from app.foreign_integration.auth import ForeignActor, current_actor
from app.foreign_integration.database import get_session, scope_version
from app.foreign_integration.research_scope import ResearchScope, root_databases
from app.foreign_integration.retrieval import ForeignRetrievalTool
from app.foreign_integration.schema import (
    foreign_corpus,
    foreign_corpus_database,
    foreign_corpus_index_job,
    foreign_database,
    foreign_knowledge_base,
    foreign_knowledge_base_document,
    foreign_user,
    metadata,
    research_agent,
    research_conversation,
    research_message,
    research_model_config,
    research_prompt,
    research_run,
    research_run_scope,
)
from app.foreign_integration.service import RunExecutor, create_run, read_run
from app.retrieval.schemas import RetrievalFilters


@pytest.fixture
def store():
    """创建两棵根树、删除分支与各类文献的独立内存库，不连接业务数据。"""
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    metadata.create_all(engine)
    with Session(engine) as session:
        session.execute(
            insert(foreign_user),
            [
                {
                    "id": key,
                    "username": f"user{key}",
                    "status": 1,
                    "role": '["2"]',
                    "is_deleted": False,
                }
                for key in (7, 8)
            ],
        )
        session.execute(
            insert(foreign_database),
            [
                {"id": key, "database_name": name, "parent_id": parent, "is_deleted": deleted}
                for key, name, parent, deleted in (
                    (9, "根库甲", -1, False),
                    (10, "子库", 9, False),
                    (11, "孙库", 10, False),
                    (20, "根库乙", -1, False),
                    (30, "删除分支", 9, True),
                    (31, "断开的孙库", 30, False),
                    (40, "删除根", -1, True),
                )
            ],
        )
        session.execute(
            insert(foreign_corpus),
            [
                {
                    "id": key,
                    "title_cn": f"文献{key}",
                    "status": 1 if key == 16 else 2,
                    "is_vectorized": key != 17,
                    "is_deleted": key == 18,
                    "attachment": "[]",
                }
                for key in range(11, 19)
            ],
        )
        session.execute(
            insert(foreign_corpus_database),
            [
                {"corpus_id": corpus, "database_id": database}
                for corpus, database in (
                    (11, 9),
                    (11, 10),
                    (12, 10),
                    (13, 11),
                    (14, 20),
                    (15, 31),
                    (16, 9),
                    (17, 9),
                    (18, 9),
                )
            ],
        )
        session.execute(
            insert(foreign_corpus_index_job),
            [
                {
                    "corpus_id": key,
                    "status": "SUCCESS",
                    "vector_status": "SUCCESS",
                    "lexical_status": "SUCCESS",
                    "build_id": f"build-{key}",
                    "is_deleted": False,
                    "attachment_digest": hashlib.sha256(b"[]").hexdigest(),
                    "lexical_index": "foreign_chunks_v1",
                }
                for key in (11, 12, 13)
            ],
        )
        session.execute(insert(foreign_knowledge_base).values(id=9, owner_id=7, is_deleted=False))
        session.execute(
            insert(foreign_knowledge_base_document).values(
                knowledge_base_id=9,
                corpus_id=14,
                is_deleted=False,
            )
        )
        session.execute(
            insert(research_model_config).values(
                id="model",
                provider="OPENAI_COMPATIBLE",
                model_name="test",
                enabled=True,
            )
        )
        for code in ("QIMING", "ZHIJIAN", "SUZHI", "RONGTONG"):
            session.execute(
                insert(research_agent).values(
                    id=code,
                    code=code,
                    name=code,
                    description="test",
                    model_config_id="model",
                    enabled=True,
                )
            )
            session.execute(
                insert(research_prompt).values(
                    id=code,
                    agent_id=code,
                    version=1,
                    system_prompt="依据证据回答",
                    status="ACTIVE",
                )
            )
        session.commit()
        yield engine, session
    engine.dispose()


def create(session, **overrides):
    """构造数据库研究任务，允许测试覆盖来源或会话。"""
    return create_run(
        session,
        **{
            "actor": ForeignActor(7, "user7", 2),
            "code": "QIMING",
            "request_id": uuid.uuid4().hex,
            "knowledge_base_id": None,
            "database_id": 9,
            "entry_source": "DATABASE",
            "query": "请分析研究问题",
            "conversation_id": None,
            "mode": "FAST",
            "options": {},
            **overrides,
        },
    )


@pytest.mark.parametrize("code", ["SUZHI", "QIMING"])
@pytest.mark.parametrize("timeout", [None, 300, 420])
def test_execution_passes_configured_model_timeout_to_shared_gateway(
    store, monkeypatch, code, timeout,
):
    """直接溯知及启明共享网关都使用环境预算，未配置时保留原等待时间。

    Args:
        store: 隔离的内存数据库及业务会话。
        monkeypatch: 环境配置替换夹具。
        code: 直接专业入口或可委派专业子图的启明入口。
        timeout: 外部配置的等待秒数；None 表示不设置。
    """
    engine, session = store
    if timeout is None:
        monkeypatch.delenv("RESEARCH_MODEL_TIMEOUT_SECONDS", raising=False)
    else:
        monkeypatch.setenv("RESEARCH_MODEL_TIMEOUT_SECONDS", str(timeout))
    settings = Settings(_env_file=None)
    accepted = create(session, code=code)
    executor = RunExecutor(accepted["run_id"])
    with (
        patch("app.foreign_integration.service.get_settings", return_value=settings),
        patch("app.foreign_integration.service.get_foreign_settings"),
        patch("app.foreign_integration.service.ForeignRetrievalTool"),
        patch(
            "app.foreign_integration.service.session_factory", return_value=lambda: Session(engine)
        ),
        patch("app.foreign_integration.service.get_model_gateway") as factory,
        patch.object(executor, "_run_graph", new=AsyncMock(return_value={})) as run_graph,
        patch.object(executor, "_publish") as publish,
        patch.object(executor, "_fail") as fail,
    ):
        asyncio.run(executor.execute())
    fail.assert_not_called()
    assert factory.call_args.kwargs["timeout_seconds"] == (timeout or 60)
    assert run_graph.call_args.args[2] is factory.return_value
    publish.assert_called_once()


def client_for(session):
    """组装拥有真实路由与内存授权替身的测试客户端。"""
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[current_actor] = lambda: ForeignActor(7, "user7", 2)
    app.dependency_overrides[get_session] = lambda: session

    @app.middleware("http")
    async def trace(request, call_next):
        """提供统一接口响应所需的请求追踪标识。"""
        request.state.request_id = "database-test"
        return await call_next(request)

    return TestClient(app)


def test_root_selection_and_descendant_deduplication(store):
    """只可选择根，文献含有效子孙且去重，不含其他根、删除分支或无效文献。"""
    _, session = store
    assert [row["id"] for row in root_databases(session)] == [9, 20]
    scope = ResearchScope.from_ids(None, 9)
    assert scope.corpus_ids(session, 7) == [11, 12, 13]
    assert ResearchScope.from_ids(9).corpus_ids(session, 7) == [14]
    assert scope.graph_id != ResearchScope.from_ids(9).graph_id
    for invalid in (10, 30, 40, 999):
        with pytest.raises(PermissionError):
            create(session, database_id=invalid)
    with pytest.raises(PermissionError):
        ResearchScope.from_ids(9).authorize(session, 8)


@pytest.mark.parametrize("code", ["QIMING", "ZHIJIAN", "SUZHI", "RONGTONG"])
def test_all_agents_share_database_binding_without_copying_members(store, code):
    """四个图入口只保存真实来源，执行恢复不枚举数据库成员。"""
    engine, session = store
    accepted = create(session, code=code)
    run = read_run(session, accepted["run_id"], ForeignActor(7, "user7", 2))
    assert run["database_id"] == 9 and run["knowledge_base_id"] is None
    assert run["entry_source"] == "RESEARCH"
    assert set(
        session.scalars(
            select(research_run_scope.c.corpus_id).where(
                research_run_scope.c.run_id == run["id"],
            )
        )
    ) == set()
    with patch(
        "app.foreign_integration.service.session_factory", return_value=lambda: Session(engine)
    ):
        context, _, _, fixed = RunExecutor(run["id"])._load_context()
        assert ResearchScope.from_record(context).resource_id == 9
        assert fixed is None


def test_binding_idempotency_and_merged_history(store):
    """合并历史来源后仍拒绝同号串库，旧入口别名可继续原资源会话。"""
    _, session = store
    accepted = create(session, request_id="database-idempotent")
    assert not create(session, request_id="database-idempotent")["created"]
    for changes in (
        {"database_id": 20},
        {"database_id": None, "knowledge_base_id": 9, "entry_source": "RESEARCH"},
    ):
        with pytest.raises(ValueError):
            create(session, request_id="database-idempotent", **changes)
        with pytest.raises(ValueError):
            create(session, conversation_id=accepted["conversation_id"], **changes)
    with pytest.raises(ValueError):
        create(session, knowledge_base_id=9)
    create(session, conversation_id=accepted["conversation_id"], entry_source="RESEARCH")
    create(session, database_id=None, knowledge_base_id=9, entry_source="RESEARCH")
    client = client_for(session)
    database_history = client.get("/conversations", params={"entry_source": "DATABASE"}).json()
    knowledge_history = client.get("/conversations", params={"entry_source": "RESEARCH"}).json()
    assert len(database_history["data"]["items"]) == len(knowledge_history["data"]["items"]) == 2
    assert all(row["entry_source"] == "RESEARCH" for row in knowledge_history["data"]["items"])
    assert (
        client.get(
            f"/conversations/{accepted['conversation_id']}", params={"entry_source": "RESEARCH"}
        ).status_code
        == 200
    )
    assert (
        client.get(
            f"/agent-runs/{accepted['run_id']}", params={"entry_source": "RESEARCH"}
        ).status_code
        == 200
    )
    assert client.get("/research-databases").json()["data"]["items"] == root_databases(session)


def test_retrieval_uses_current_members_and_changes_do_not_block_publish(store):
    """后续检索使用当前成员与双路索引；分类移动不让已经执行的任务发布失败。"""
    engine, session = store
    accepted = create(session)
    executor = RunExecutor(accepted["run_id"])
    with patch(
        "app.foreign_integration.service.session_factory", return_value=lambda: Session(engine)
    ):
        run, _, _, _ = executor._load_context()
    tool = ForeignRetrievalTool(
        actor_id=7,
        knowledge_base_id=0,
        database_id=9,
        fixed_scope=None,
        fixed_version=scope_version([11, 12, 13]),
        settings=SimpleNamespace(opensearch_index="foreign_chunks_v1"),
    )
    with patch(
        "app.foreign_integration.retrieval.session_factory", return_value=lambda: Session(engine)
    ):
        assert set(tool._load_scope(RetrievalFilters())) == {11, 12, 13}
        session.execute(
            update(foreign_corpus_index_job)
            .where(
                foreign_corpus_index_job.c.corpus_id == 12,
            )
            .values(lexical_status="PENDING")
        )
        session.commit()
        assert set(tool._load_scope(RetrievalFilters())) == {11, 13}
        session.execute(
            update(foreign_database).where(foreign_database.c.id == 10).values(parent_id=20)
        )
        session.commit()
        assert set(tool._load_scope(RetrievalFilters())) == {11}
    executor.retrieval = tool
    session.execute(update(research_run).where(
        research_run.c.id == accepted["run_id"],
    ).values(status="RUNNING"))
    session.commit()
    with (
        patch(
            "app.foreign_integration.service.session_factory", return_value=lambda: Session(engine)
        ),
    ):
        executor._publish(run, {"final_answer": "证据不足。", "evidence_level": "INSUFFICIENT"})
    assert session.scalar(
        select(research_message.c.id).where(
            research_message.c.role == "ASSISTANT",
        )
    )


def test_deleted_database_history_is_readable_but_cannot_execute(store):
    """删除根库后历史可读且无摘要，继续运行及恢复执行上下文被拒绝。"""
    engine, session = store
    accepted = create(session)
    session.execute(
        update(foreign_database)
        .where(
            foreign_database.c.id == 9,
        )
        .values(is_deleted=True)
    )
    session.commit()
    client = client_for(session)
    history = client.get(
        "/agent-runs",
        params={
            "executor_agent_code": "QIMING",
            "entry_source": "DATABASE",
        },
    ).json()["data"]["items"][0]
    assert not history["database_accessible"]
    assert history["summary"] == ""
    assert (
        client.get(
            f"/conversations/{accepted['conversation_id']}", params={"entry_source": "DATABASE"}
        ).status_code
        == 200
    )
    with pytest.raises(PermissionError):
        create(session, conversation_id=accepted["conversation_id"])
    with (
        patch(
            "app.foreign_integration.service.session_factory", return_value=lambda: Session(engine)
        ),
        pytest.raises(PermissionError),
    ):
        RunExecutor(accepted["run_id"])._load_context()


def test_api_rejects_ambiguous_or_nonroot_sources_before_dispatch(store):
    """HTTP 入口复用统一来源规则，非法请求不会调度图或写入运行。"""
    _, session = store
    client = client_for(session)
    payload = {
        "request_id": "api-database-root",
        "database_id": 9,
        "entry_source": "DATABASE",
        "query": "分析研究方向",
    }
    with patch("app.foreign_integration.api._execute_run", new_callable=AsyncMock):
        assert client.post("/agents/qiming/runs", json=payload).status_code == 202
        assert (
            client.post(
                "/agents/qiming/runs",
                json={
                    **payload,
                    "knowledge_base_id": 9,
                },
            ).status_code
            == 422
        )
        assert (
            client.post(
                "/agents/qiming/runs",
                json={
                    **payload,
                    "request_id": "api-database-child",
                    "database_id": 10,
                },
            ).status_code
            == 403
        )
    assert session.scalar(select(research_conversation.c.database_id)) == 9
    assert len(session.scalars(select(research_run.c.id)).all()) == 1


def test_fixed_database_history_filters_before_pagination_and_guards_details(store):
    """固定智能体按数据库先筛选再分页，总数准确，跨库及跨用户详情不可恢复。"""
    _, session = store
    first = create(session, database_id=9)
    create(session, database_id=20)
    second = create(session, database_id=9)
    create(session, database_id=20)
    create(session, database_id=None, knowledge_base_id=9, entry_source="RESEARCH")
    create(session, actor=ForeignActor(8, "user8", 2))
    client = client_for(session)
    expected_conversations = {first["conversation_id"], second["conversation_id"]}
    expected_runs = {first["run_id"], second["run_id"]}
    for endpoint, extra, expected in (
        ("/conversations", {}, expected_conversations),
        ("/agent-runs", {"executor_agent_code": "QIMING"}, expected_runs),
    ):
        params = {"entry_source": "DATABASE", "database_id": 9, "page_size": 1, **extra}
        pages = [client.get(endpoint, params={**params, "page": page}) for page in (1, 2, 3)]
        assert all(response.status_code == 200 for response in pages)
        data = [response.json()["data"] for response in pages]
        assert all(item["total"] == 2 for item in data)
        assert {item["items"][0]["id"] for item in data[:2]} == expected
        assert data[2]["items"] == []
        unfiltered = client.get(endpoint, params={"entry_source": "DATABASE", **extra})
        assert unfiltered.json()["data"]["total"] == 5
        for invalid in (0, -1):
            response = client.get(endpoint, params={**params, "database_id": invalid})
            assert response.status_code == 422
        assert client.get(
            endpoint, params={**params, "entry_source": "RESEARCH"},
        ).status_code == 200
    for path in (
        f"/conversations/{first['conversation_id']}",
        f"/agent-runs/{first['run_id']}",
    ):
        params = {"entry_source": "DATABASE", "database_id": 9}
        assert client.get(path, params=params).status_code == 200
        assert client.get(path, params={**params, "database_id": 20}).status_code == 404
        assert client.get(path, params={"entry_source": "DATABASE"}).status_code == 200
