"""验证研究来源合并、资源边界、历史分页与幂等迁移，不访问外部模型或业务库。"""

from unittest.mock import AsyncMock, patch

import pytest
from sqlalchemy import select, update
from test_database_research import client_for, create
from test_database_research import store as database_store

from app.foreign_integration.migrate import _ensure_conversation_sources
from app.foreign_integration.research_scope import ResearchScope
from app.foreign_integration.schema import (
    research_conversation,
    research_message,
    research_run,
    research_run_scope,
)


@pytest.fixture
def store():
    """复用包含同号数据库和知识库的隔离内存库。"""
    yield from database_store.__wrapped__()


@pytest.mark.parametrize("code", ["QIMING", "ZHIJIAN", "SUZHI", "RONGTONG"])
@pytest.mark.parametrize("binding", [{"database_id": 9}, {"knowledge_base_id": 9}, {}])
def test_canonical_entry_accepts_all_resource_types(store, code, binding):
    """四个模块均只保存全库或真实资源引用，实际范围来自当前资源关联。"""
    _, session = store
    with patch("app.foreign_integration.api._execute_run", new_callable=AsyncMock):
        response = client_for(session).post(f"/agents/{code}/runs", json={
            "request_id": f"unified-{code}-{next(iter(binding), 'global')}",
            "entry_source": "RESEARCH", "query": "研究教育政策", **binding,
            "options": {"target_language": "EN"} if code == "RONGTONG" else {},
        })
    assert response.status_code == 202, response.text
    accepted = response.json()["data"]
    row = session.execute(select(research_conversation).where(
        research_conversation.c.id == accepted["conversation_id"],
    )).mappings().one()
    assert row["entry_source"] == "RESEARCH"
    assert row["database_id"] == binding.get("database_id")
    assert row["knowledge_base_id"] == binding.get("knowledge_base_id")
    ids = set(session.scalars(select(research_run_scope.c.corpus_id).where(
        research_run_scope.c.run_id == accepted["run_id"],
    )))
    assert ids == set()
    current = set(ResearchScope.from_record(row).corpus_ids(
        session, 7, include_unindexed=not binding,
    ))
    assert current == ({11, 12, 13} if "database_id" in binding else {14}
                       if "knowledge_base_id" in binding else {11, 12, 13, 14, 15, 17})


def test_migration_preserves_conversations_and_related_records(store):
    """只归一旧来源；迁移两次后范围、时间、消息及 Run 的所有字段都保持原值。"""
    engine, session = store
    database = create(session)
    global_scope = create(session, database_id=None)
    create(session, database_id=None, knowledge_base_id=9, entry_source="RESEARCH")
    session.execute(update(research_conversation).where(
        research_conversation.c.id.in_([
            database["conversation_id"], global_scope["conversation_id"],
        ]),
    ).values(entry_source="DATABASE"))
    session.commit()
    tables = [research_conversation, research_run, research_message, research_run_scope]
    before = {table.name: [dict(row) for row in session.execute(select(table)).mappings()]
              for table in tables}
    _ensure_conversation_sources(engine)
    _ensure_conversation_sources(engine)
    session.expire_all()
    for table in tables:
        expected = before[table.name]
        if table is research_conversation:
            expected = [{**row, "entry_source": "RESEARCH"} for row in expected]
        assert [dict(row) for row in session.execute(select(table)).mappings()] == expected
    client = client_for(session)
    history = client.get("/conversations", params={"entry_source": "RESEARCH", "page_size": 1})
    assert history.json()["data"]["total"] == 3
    pages = [client.get("/conversations", params={"page": page, "page_size": 1}).json()["data"]
             for page in (1, 2, 3)]
    assert len({data["items"][0]["id"] for data in pages}) == 3
    assert client.get(f"/conversations/{database['conversation_id']}", params={
        "entry_source": "RESEARCH",
    }).status_code == 200
    follow_up = create(
        session, conversation_id=database["conversation_id"], entry_source="RESEARCH",
    )
    assert follow_up["conversation_id"] == database["conversation_id"]
