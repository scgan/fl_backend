"""在隔离内存库验证 AI 题录 SQL 的组合逻辑、权限、分页和特殊字符处理。"""

from types import SimpleNamespace

import pytest
from pydantic import ValidationError
from sqlalchemy import event, insert, update
from test_ai_assistant import client as ai_client
from test_ai_assistant import start_ai
from test_ai_assistant import store as ai_store

from app.agents.ai_assistant_schemas import AssistantSearchPlan
from app.foreign_integration.assistant_search import search_document_index, search_documents
from app.foreign_integration.literature import LiteratureMaterialTool
from app.foreign_integration.research_scope import ResearchScope
from app.foreign_integration.schema import foreign_corpus, research_run


@pytest.fixture
def store():
    """复用 AI 测试的隔离数据库。"""
    yield from ai_store.__wrapped__()


@pytest.fixture
def client(store):
    """复用绑定隔离数据库和用户身份的 HTTP 客户端。"""
    yield from ai_client.__wrapped__(store)


def search(store, concepts, **kwargs):
    """对内存题录执行真实 SQL，允许测试替换结构化筛选。"""
    return search_documents(
        store.session, 7, list(range(11, 100)), AssistantSearchPlan(concepts=concepts, **kwargs)
    )


def test_concepts_synonyms_abstract_and_literal_wildcards(store):
    """不同概念 AND、组内 OR，摘要可命中且通配符及 SQL 字符串按字面处理。"""
    store.session.execute(
        update(foreign_corpus).values(
            title_cn="双减政策",
            abstract_cn="教师工作量发生变化",
            language="中文",
            publish_time=2022,
        )
    )
    result = search(store, [["双减"], ["工作负担", "工作量"]], language="ZH", year_from=2021)
    assert result["total"] == 1 and result["documents"][0]["readable"] is False
    assert search(store, [["双减"], ["学生"]])["total"] == 0
    assert search(store, [["双减"]], keyword_only=True)["total"] == 0
    assert search(store, [["双减"]], year_to=2021)["total"] == 0
    for word in ["%", "_", "' OR 1=1 --"]:
        assert search(store, [[word]])["total"] == 0
    store.session.execute(update(foreign_corpus).values(author="张三"))
    assert search(store, [], author="张三")["total"] == 1
    store.session.execute(update(foreign_corpus).values(keyword_cn='["100%有效"]'))
    assert search(store, [["100%"]], keyword_only=True)["total"] == 1


def test_more_than_eight_papers_stable_database_pagination_and_scope(store):
    """独立论文按十篇翻页，排除他人材料、未审核题录与冻结范围外论文。"""
    store.session.execute(
        insert(foreign_corpus),
        [
            {"id": value, "title_cn": "双减", "status": 2, "owner_id": None, "is_deleted": False}
            for value in range(12, 37)
        ],
    )
    store.session.execute(
        insert(foreign_corpus),
        [
            {"id": 37, "title_cn": "双减", "status": 4, "owner_id": 8, "is_deleted": False},
            {"id": 38, "title_cn": "双减", "status": 1, "owner_id": None, "is_deleted": False},
            {"id": 39, "title_cn": "双减", "status": 4, "owner_id": 7, "is_deleted": False},
        ],
    )
    plan = AssistantSearchPlan(concepts=[["双减"]])
    first = search(store, [["双减"]])
    second = search_documents(store.session, 7, list(range(11, 100)), plan, page=2)
    assert first["total"] == 26
    third = search_documents(store.session, 7, list(range(11, 100)), plan, page=3)
    assert len(first["documents"]) == len(second["documents"]) == 10
    assert len(third["documents"]) == 6
    ids = [
        item["corpus_id"] for item in first["documents"] + second["documents"] + third["documents"]
    ]
    assert len(set(ids)) == 26 and ids == sorted(ids, reverse=True)
    assert search_documents(store.session, 7, [12], plan)["total"] == 1
    assert search_documents(store.session, 7, [], plan)["total"] == 0


def test_backend_caps_results_at_500_and_rejects_page_51(store):
    """超过五百篇只开放五十页，数量上限不能靠客户端改页码绕过。"""
    ids = list(range(12, 523))
    store.session.execute(
        insert(foreign_corpus),
        [{"id": value, "title_cn": "双减", "status": 2, "is_deleted": False} for value in ids],
    )
    plan = AssistantSearchPlan(concepts=[["双减"]])
    first = search_documents(store.session, 7, ids, plan)
    last = search_documents(store.session, 7, ids, plan, page=50)
    assert first["total"] == last["total"] == 500
    assert first["page_size"] == 10 and len(last["documents"]) == 10
    assert last["documents"][-1]["corpus_id"] == sorted(ids, reverse=True)[499]
    with pytest.raises(ValueError, match="分页"):
        search_documents(store.session, 7, ids, plan, page=51)
    with pytest.raises(ValueError, match="分页"):
        search_documents(store.session, 7, ids, plan, page_size=500)


@pytest.mark.parametrize(
    "data",
    [
        {"concepts": []},
        {"concepts": [[]]},
        {"concepts": [[" "]]},
        {"concepts": [["双减"]], "year_from": 2024, "year_to": 2021},
    ],
)
def test_invalid_plan_is_rejected(data):
    """无效模型条件必须失败，不能悄悄退化为无条件全库查询。"""
    with pytest.raises(ValidationError):
        AssistantSearchPlan(**data)


def test_http_pagination_uses_saved_plan_with_current_metadata(store, client, monkeypatch):
    """翻页复用条件并允许题录变化；未完成任务和其他用户仍不能读取。"""
    accepted = start_ai(store)
    run_id = accepted["run_id"]
    assert client.get(f"/api/v1/agent-runs/{run_id}/documents").status_code == 409
    store.session.execute(update(foreign_corpus).values(title_cn="双减"))
    store.session.execute(
        update(research_run).values(
            status="SUCCESS",
            result_payload={
                "result_type": "AI_ASSISTANT",
                "payload": {"search_plan": {"concepts": [["双减"]]}},
            },
        )
    )
    store.session.commit()
    monkeypatch.setattr(
        "app.foreign_integration.api.get_foreign_settings",
        lambda: SimpleNamespace(opensearch_index="foreign_chunks_v1"),
    )
    changed = client.get(f"/api/v1/agent-runs/{run_id}/documents")
    assert changed.status_code == 200
    assert changed.json()["data"]["documents"][0]["title"] == "双减"
    response = client.get(f"/api/v1/agent-runs/{run_id}/documents?page=2")
    assert response.status_code == 200 and response.json()["data"]["page"] == 2
    assert client.get(f"/api/v1/agent-runs/{run_id}/documents?page=0").status_code == 422
    assert client.get(f"/api/v1/agent-runs/{run_id}/documents?page=51").status_code == 422
    store.session.execute(update(research_run).values(user_id=8))
    store.session.commit()
    assert client.get(f"/api/v1/agent-runs/{run_id}/documents").status_code == 404


def test_first_discovery_builds_snapshot_and_pages_skip_keyword_scans(store, monkeypatch):
    """首次检索生成完整 ID 快照；翻页只读取十条详情，不重复主题匹配和计数。"""
    from sqlalchemy.orm import Session

    store.session.execute(update(foreign_corpus).values(title_cn="双减研究"))
    store.session.execute(insert(foreign_corpus), [
        {"id": key, "title_cn": "双减研究", "status": 2, "is_deleted": False}
        for key in range(12, 37)
    ])
    store.session.commit()
    monkeypatch.setattr(
        "app.foreign_integration.literature.session_factory", lambda: lambda: Session(store.engine),
    )
    tool = SimpleNamespace(actor_id=7, document_order=None,
                           settings=SimpleNamespace(opensearch_index="foreign_chunks_v1"),
                           scope=ResearchScope("GLOBAL", 0))
    plan = AssistantSearchPlan(concepts=[["双减"]])
    first = LiteratureMaterialTool._discover_sync(tool, plan)
    index = first["document_index"]
    assert first["total"] == len(index["ids"]) == 26
    assert len(first["documents"]) == 10
    statements = []

    def capture(_connection, _cursor, statement, _parameters, _context, _executemany):
        """记录翻页 SQL，用于验证不会重扫主题词或重复计数。"""
        statements.append(statement.lower())

    event.listen(store.engine, "before_cursor_execute", capture)
    try:
        second = search_documents(store.session, 7, None, plan, page=2,
                                  document_ids=index["ids"], total=index["total"])
    finally:
        event.remove(store.engine, "before_cursor_execute", capture)
    assert [doc["corpus_id"] for doc in second["documents"]] == index["ids"][10:20]
    assert not any("like" in statement or "count(" in statement for statement in statements)


@pytest.mark.parametrize("ids", [[], [11]])
def test_http_uses_initial_snapshot_including_empty_without_rebuilding(
    store, client, monkeypatch, ids,
):
    """已持久化的初始快照（包括零命中）直接用于翻页，不再执行整库检索。"""
    run_id = start_ai(store)["run_id"]
    store.session.execute(update(research_run).values(
        status="SUCCESS", result_payload={"result_type": "AI_ASSISTANT", "payload": {
            "search_plan": {"concepts": [["双减"]]},
            "document_index": {"ids": ids, "total": len(ids)},
        }},
    ))
    store.session.commit()
    monkeypatch.setattr("app.foreign_integration.api.get_foreign_settings",
                        lambda: SimpleNamespace(opensearch_index="foreign_chunks_v1"))

    def unexpected(*_args, **_kwargs):
        """快照路径重新执行主题检索视为测试失败。"""
        raise AssertionError("不应重新检索")

    monkeypatch.setattr(
        "app.foreign_integration.assistant_search.search_document_index", unexpected,
    )
    first = client.get(f"/api/v1/agent-runs/{run_id}/documents").json()["data"]
    assert first["total"] == len(ids)
    assert len(first["documents"]) == len(ids)
    second = client.get(f"/api/v1/agent-runs/{run_id}/documents?page=2")
    assert second.status_code == 200 and second.json()["data"]["documents"] == []


def test_snapshot_cannot_read_deleted_or_newly_private_documents(store):
    """缓存只保留顺序，不允许翻页继续读取已删除或转为其他用户私有的文献。"""
    store.session.execute(update(foreign_corpus).values(title_cn="双减研究"))
    store.session.execute(insert(foreign_corpus).values(
        id=12, title_cn="双减研究", status=2, is_deleted=False,
    ))
    plan = AssistantSearchPlan(concepts=[["双减"]])
    index = search_document_index(store.session, 7, None, plan)
    assert set(index["ids"]) == {11, 12}
    store.session.execute(update(foreign_corpus).where(foreign_corpus.c.id == 11)
                          .values(is_deleted=True))
    store.session.execute(update(foreign_corpus).where(foreign_corpus.c.id == 12)
                          .values(owner_id=8, status=4))
    result = search_documents(store.session, 7, None, plan,
                              document_ids=index["ids"], total=index["total"])
    assert result["documents"] == []
