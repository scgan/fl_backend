"""验证 AI 与数据库研究的全库/根库范围、固定会话绑定及入口历史隔离。"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from sqlalchemy import select, update
from test_database_research import client_for, create
from test_database_research import store as database_store

from app.foreign_integration.literature import (
    LiteratureMaterialTool,
    selection_rows,
    selection_version,
)
from app.foreign_integration.research_scope import ResearchScope
from app.foreign_integration.schema import foreign_database, research_run_scope


@pytest.fixture
def store():
    """复用两棵根数据库及其后代的隔离测试数据。"""
    yield from database_store.__wrapped__()


def test_ai_root_scope_includes_unindexed_metadata_and_stays_in_ai_history(store):
    """AI 根库只记录资源引用；当前范围含未索引题录，历史仍与其他入口隔离。"""
    _, session = store
    accepted = create(session, entry_source="AI")
    ids = set(
        session.scalars(
            select(research_run_scope.c.corpus_id).where(
                research_run_scope.c.run_id == accepted["run_id"],
            )
        )
    )
    assert ids == set()
    assert set(ResearchScope.from_ids(None, 9).corpus_ids(
        session, 7, include_unindexed=True,
    )) == {11, 12, 13, 17}
    client = client_for(session)
    assert client.get("/conversations", params={"entry_source": "AI"}).json()["data"]["total"] == 1
    for source in ["DATABASE", "RESEARCH", "WRITING"]:
        assert (
            client.get("/conversations", params={"entry_source": source}).json()["data"]["total"]
            == 0
        )
        assert (
            client.get(
                "/agent-runs", params={"entry_source": source, "executor_agent_code": "QIMING"}
            ).json()["data"]["total"]
            == 0
        )
        assert (
            client.get(
                f"/agent-runs/{accepted['run_id']}", params={"entry_source": source}
            ).status_code
            == 404
        )
    with pytest.raises(PermissionError):
        create(session, entry_source="AI", database_id=10)
    with pytest.raises(ValueError):
        create(
            session,
            entry_source="AI",
            database_id=None,
            conversation_id=accepted["conversation_id"],
        )


def test_database_global_creates_and_restores_without_fake_database_id(store):
    """统一研究允许全库，兼容 DATABASE 查询且不得在同会话换为根库。"""
    _, session = store
    accepted = create(session, database_id=None)
    ids = set(
        session.scalars(
            select(research_run_scope.c.corpus_id).where(
                research_run_scope.c.run_id == accepted["run_id"],
            )
        )
    )
    assert ids == set()
    assert set(ResearchScope("GLOBAL", 0).corpus_ids(
        session, 7, include_unindexed=True,
    )) == {11, 12, 13, 14, 15, 17}
    client = client_for(session)
    detail = client.get(
        f"/conversations/{accepted['conversation_id']}", params={"entry_source": "DATABASE"}
    )
    assert detail.status_code == 200 and detail.json()["data"]["database_id"] is None
    assert detail.json()["data"]["entry_source"] == "RESEARCH"
    assert (
        client.get(
            "/agent-runs", params={"entry_source": "DATABASE", "executor_agent_code": "QIMING"}
        ).json()["data"]["total"]
        == 1
    )
    with pytest.raises(ValueError):
        create(session, conversation_id=accepted["conversation_id"])


def test_http_accepts_ai_root_and_database_global_without_executing_models(store):
    """API 接受两种范围，但仍拒绝 AI 跨入口恢复和虚构数据库零值。"""
    _, session = store
    client = client_for(session)
    with patch("app.foreign_integration.api._execute_run", new_callable=AsyncMock):
        for source, database in [("AI", 9), ("DATABASE", None)]:
            payload = {
                "request_id": f"test-scope-{source}",
                "entry_source": source,
                "query": "查询相关文献",
            }
            if database is not None:
                payload["database_id"] = database
            response = client.post("/agents/QIMING/runs", json=payload)
            assert response.status_code == 202, response.text
        assert (
            client.post(
                "/agents/QIMING/runs",
                json={
                    "request_id": "invalid-global-zero",
                    "entry_source": "DATABASE",
                    "database_id": 0,
                    "query": "研究问题",
                },
            ).status_code
            == 422
        )


def test_ai_root_tool_rechecks_root_permission(store, monkeypatch):
    """题录与正文工具都不能在根库被删除后继续读取冻结文献。"""
    engine, session = store
    from sqlalchemy.orm import Session

    ids = [11, 12, 13, 17]
    version = selection_version(selection_rows(session, 7, ids))
    monkeypatch.setattr(
        "app.foreign_integration.literature.session_factory", lambda: lambda: Session(engine)
    )
    tool = LiteratureMaterialTool(
        actor_id=7,
        knowledge_base_id=0,
        database_id=9,
        fixed_scope=ids,
        fixed_version=version,
        settings=SimpleNamespace(),
    )
    assert set(tool._selection()) == set(ids)
    session.execute(
        update(foreign_database).where(foreign_database.c.id == 9).values(is_deleted=True)
    )
    session.commit()
    with pytest.raises(PermissionError):
        tool._selection()
