"""验证任务范围按资源查询、全库不枚举成员，以及已取得证据在数据变动后仍可发布。

使用真实 SQLite 事务和受控外部召回替身，覆盖旧表缺失、旧运行恢复、权限过滤及引用快照。
"""

import hashlib
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from sqlalchemy import create_engine, event, insert, inspect, select, update
from sqlalchemy.orm import Session
from test_database_research import client_for, create
from test_database_research import store as database_store
from test_hybrid_retrieval import example
from test_hybrid_retrieval import tool as example_tool
from test_literature import citation, evidence

from app.agents.ai_assistant_schemas import AssistantSearchPlan
from app.foreign_integration.auth import ForeignActor
from app.foreign_integration.lexical import lexical_search
from app.foreign_integration.literature import LiteratureMaterialTool
from app.foreign_integration.retrieval import ForeignRetrievalTool, SourcePoint
from app.foreign_integration.schema import (
    RESEARCH_TABLES,
    foreign_corpus,
    foreign_corpus_index_job,
    foreign_database,
    foreign_knowledge_base_document,
    metadata,
    research_citation,
    research_evidence,
    research_message,
    research_run,
    research_run_scope,
)
from app.foreign_integration.service import RunExecutor, soft_delete_conversation
from app.retrieval.schemas import RetrievalFilters


@pytest.fixture
def store():
    """复用包含根分类、知识库及索引状态的隔离内存数据库。"""
    yield from database_store.__wrapped__()


@pytest.mark.parametrize("source,binding,options", [
    ("RESEARCH", {}, {}),
    ("AI", {}, {}),
    ("RESEARCH", {"database_id": 9}, {}),
    ("AI", {"database_id": 9}, {}),
    ("RESEARCH", {"knowledge_base_id": 9}, {}),
    ("AI", {}, {"corpus_ids": [13, 11]}),
])
def test_create_and_restore_without_legacy_table_or_member_queries(store, source, binding, options):
    """旧表不存在时仍可提交、恢复及删除任务；资源任务不在提交时枚举文献。"""
    engine, session = store
    research_run_scope.drop(engine)
    statements = []

    def record(_connection, _cursor, statement, _parameters, _context, _executemany):
        """记录真实 SQL，防止仅换掉关联写入但仍扫描整库。"""
        statements.append(statement.lower())

    event.listen(engine, "before_cursor_execute", record)
    try:
        accepted = create(
            session, entry_source=source, options=options, **{"database_id": None, **binding},
        )
        if not options.get("corpus_ids"):
            assert not any("from corpus " in sql or "from corpus\n" in sql for sql in statements)
        statements.clear()
        with patch(
            "app.foreign_integration.service.session_factory", return_value=lambda: Session(engine),
        ):
            run, _, _, selected = RunExecutor(accepted["run_id"])._load_context()
        assert selected == (options.get("corpus_ids") or None)
        assert not any("from corpus " in sql or "from corpus\n" in sql for sql in statements)
        assert run["scope_version"] == 1
        session.execute(update(research_run).values(status="SUCCESS"))
        session.commit()
        soft_delete_conversation(session, accepted["conversation_id"], ForeignActor(7, "user7", 2))
        assert session.scalar(select(research_run.c.is_deleted))
        assert not any("research_run_corpus" in sql for sql in statements)
    finally:
        event.remove(engine, "before_cursor_execute", record)


def test_large_global_creation_does_not_read_or_copy_68000_members(store):
    """六万八千篇文献时，任务提交只做账号及配置校验，不查询题录或建立范围关联。"""
    engine, session = store
    session.execute(insert(foreign_corpus), [
        {"id": key, "status": 2, "is_deleted": False, "title_cn": "双减政策"}
        for key in range(1000, 69000)
    ])
    session.commit()
    research_run_scope.drop(engine)
    with (
        patch("app.foreign_integration.service.selection_rows",
              side_effect=AssertionError("全库题录")),
        patch("app.foreign_integration.research_scope.ResearchScope.corpus_ids",
              side_effect=AssertionError("全库成员")),
    ):
        accepted = create(session, database_id=None)
        repeated = create(session, database_id=None, request_id="large-global-idempotency")
        assert not create(
            session, database_id=None, request_id="large-global-idempotency",
        )["created"]
    assert accepted["created"] and repeated["created"]


def test_old_global_run_paginates_against_current_data_without_legacy_table(store, monkeypatch):
    """旧运行即使保留历史范围指纹，新增文献及题名变动也不阻止翻页或引入库限制。"""
    engine, session = store
    accepted = create(session, database_id=None, entry_source="AI")
    session.execute(update(research_run).values(
        status="SUCCESS", scope_version=987654321,
        result_payload={"result_type": "AI_ASSISTANT", "payload": {
            "search_plan": {"concepts": [["双减"]]},
        }},
    ))
    session.execute(insert(foreign_corpus).values(
        id=100, title_cn="新增双减文献", status=2, is_deleted=False,
    ))
    session.execute(update(foreign_corpus).where(foreign_corpus.c.id == 11).values(
        title_cn="双减新题名",
    ))
    session.commit()
    research_run_scope.drop(engine)
    monkeypatch.setattr("app.foreign_integration.api.get_foreign_settings", lambda: SimpleNamespace(
        opensearch_index="foreign_chunks_v1",
    ))
    response = client_for(session).get(f"/agent-runs/{accepted['run_id']}/documents")
    assert response.status_code == 200, response.text
    assert {item["corpus_id"] for item in response.json()["data"]["documents"]} == {11, 100}


def test_ai_discovery_uses_current_root_members_without_metadata_preload(store, monkeypatch):
    """AI 根库题录搜索直接按当前关联分页，新增成员不依赖原任务清单或完整题录预读。"""
    engine, session = store
    tool = LiteratureMaterialTool(
        actor_id=7, knowledge_base_id=0, database_id=9, fixed_scope=None, fixed_version=1,
        settings=SimpleNamespace(opensearch_index="foreign_chunks_v1"),
    )
    monkeypatch.setattr(
        "app.foreign_integration.literature.session_factory", lambda: lambda: Session(engine),
    )
    monkeypatch.setattr(tool, "_selection", lambda: pytest.fail("题录发现不应预读完整材料"))
    session.execute(update(foreign_corpus).values(title_cn="双减政策"))
    session.commit()
    assert tool._discover_sync(AssistantSearchPlan(concepts=[["双减"]]))["total"] == 4
    session.execute(update(foreign_database).where(foreign_database.c.id == 10).values(
        parent_id=20,
    ))
    session.commit()
    assert tool._discover_sync(AssistantSearchPlan(concepts=[["双减"]]))["total"] == 2


def test_global_recall_checks_candidates_and_excludes_private_or_stale_sources(store, monkeypatch):
    """全库不生成索引过滤清单；命中后只查候选，仍排除他人材料和旧构建。"""
    engine, session = store
    rows, point, _ = example()
    session.execute(update(foreign_corpus).where(foreign_corpus.c.id.in_([11, 12, 13, 14])).values(
        attachment=rows[11]["attachment"], is_vectorized=True,
    ))
    session.execute(update(foreign_corpus).where(foreign_corpus.c.id == 12).values(
        status=4, owner_id=8,
    ))
    session.execute(update(foreign_corpus).where(foreign_corpus.c.id == 13).values(
        status=4, owner_id=7,
    ))
    session.execute(update(foreign_corpus_index_job).values(
        build_id="build-1", attachment_digest=rows[11]["index_attachment_digest"],
    ))
    session.execute(insert(foreign_corpus_index_job).values(
        corpus_id=14, status="SUCCESS", vector_status="SUCCESS", lexical_status="SUCCESS",
        build_id="build-1", attachment_digest=rows[11]["index_attachment_digest"],
        lexical_index="foreign_chunks_v1", is_deleted=False,
    ))
    session.commit()
    points = [SimpleNamespace(
        id=key * 1000000, score=0.9,
        payload={**point.payload, "corpus_id": key, "build_id": "old" if key == 14 else "build-1"},
    ) for key in (11, 12, 13, 14)]
    tool = ForeignRetrievalTool(
        actor_id=7, knowledge_base_id=0, fixed_scope=None, fixed_version=1,
        settings=example_tool().settings,
    )

    def recall(_query, scope_rows, _mode):
        """模拟索引返回混合候选，要求全库没有提前构造文献或构建白名单。"""
        assert scope_rows is None
        return points, [], points

    monkeypatch.setattr(tool, "_recall", recall)
    monkeypatch.setattr(
        "app.foreign_integration.retrieval.session_factory", lambda: lambda: Session(engine),
    )
    result = tool._search_sync("learning evidence", "FAST", RetrievalFilters(), None)
    assert {item.document_id.int for item in result.evidence} == {11, 13}
    assert {source.corpus_id for source in tool.sources.values()} == {11, 13}


def test_global_index_queries_do_not_include_full_membership_filters():
    """Qdrant 仅下推模型限制，OpenSearch 不发送全量 corpus_id 或 build_id terms。"""
    tool = example_tool()
    with (
        patch("app.foreign_integration.retrieval.embed_texts", return_value=[[0.1] * 1024]),
        patch("app.foreign_integration.retrieval.QdrantClient") as constructor,
        patch("app.foreign_integration.retrieval.lexical_search", return_value=[]) as words,
    ):
        constructor.return_value.query_points.return_value.points = []
        tool._recall("双减", None, "FAST")
    conditions = constructor.return_value.query_points.call_args.kwargs["query_filter"].must
    assert {condition.key for condition in conditions} == {"embedding_model", "embedding_dimension"}
    assert words.call_args.args[2:4] == (None, None)
    with patch("app.foreign_integration.lexical.OpenSearch") as constructor:
        constructor.return_value.search.return_value = {"hits": {"hits": []}}
        lexical_search(tool.settings, "双减", None, None, 30)
        query = constructor.return_value.search.call_args.kwargs["body"]["query"]
        assert query["bool"]["filter"] == []


def test_acquired_citation_publishes_after_membership_and_attachment_changes(store, monkeypatch):
    """取得证据后即使成员移出、文献删除或附件换版，仍发布原正文、页码及引文快照。"""
    engine, session = store
    accepted = create(session)
    session.execute(update(research_run).values(status="RUNNING"))
    session.commit()
    monkeypatch.setattr(
        "app.foreign_integration.service.session_factory", lambda: lambda: Session(engine),
    )
    run, _, _, _ = RunExecutor(accepted["run_id"])._load_context()
    item = evidence(1, 11)
    source = SourcePoint(
        corpus_id=11, point_id=str(item.chunk_id.int), chunk_key="captured-build/chunk-1",
        file_path="old.pdf", content=item.content, page_start=1, page_end=1, score=0.9,
    )
    executor = RunExecutor(accepted["run_id"])
    executor.retrieval = SimpleNamespace(sources={item.chunk_id: source})
    monkeypatch.setattr(
        "app.foreign_integration.service.session_factory", lambda: lambda: Session(engine),
    )
    monkeypatch.setattr(
        "app.foreign_integration.service.get_foreign_settings",
        lambda: SimpleNamespace(qdrant_collection="test"),
    )
    executor._save_evidence(session, {"evidence": [item]})
    session.execute(update(foreign_corpus).where(foreign_corpus.c.id == 11).values(
        is_deleted=True, attachment='[{"file_path":"new.pdf"}]', is_vectorized=False,
    ))
    session.execute(update(foreign_knowledge_base_document).values(is_deleted=True))
    session.execute(update(foreign_database).where(foreign_database.c.id == 9).values(
        is_deleted=True,
    ))
    session.commit()
    research_run_scope.drop(engine)
    executor._publish(run, {
        "final_answer": "依据已取得的证据回答[1]。", "evidence_level": "LOW",
        "validated_citations": [citation(item)],
    })
    session.expire_all()
    assert session.scalar(select(research_run.c.status)) == "SUCCESS"
    saved = session.execute(select(research_evidence)).mappings().one()
    assert saved["content_snapshot"] == item.content and saved["file_path"] == "old.pdf"
    assert saved["content_sha256"] == hashlib.sha256(item.content.encode()).hexdigest()
    assert session.scalar(select(research_citation.c.quote_text)) == citation(item).quote
    assert len(list(session.scalars(select(research_message.c.id)))) == 2


def test_retired_scope_table_is_not_created_for_new_deployments():
    """新研究表清单不创建废弃关联表，也不再将它作为启动就绪条件。"""
    engine = create_engine("sqlite://")
    try:
        metadata.create_all(engine, tables=RESEARCH_TABLES)
        assert "research_run_corpus" not in inspect(engine).get_table_names()
    finally:
        engine.dispose()
