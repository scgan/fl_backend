"""验证自然语言阅读的来源指代、冻结范围、真实阅读图及同AI会话发布，不调用在线服务。"""

import asyncio
import json
import uuid
from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest
from sqlalchemy import insert, select, update
from sqlalchemy.orm import Session
from test_ai_assistant import client as ai_client
from test_ai_assistant import start_ai
from test_ai_assistant import store as ai_store
from test_literature import FakeGateway, citation, evidence

from app.agents.ai_assistant_schemas import AssistantIntent, AssistantReadingContext
from app.foreign_integration.ai_reading import (
    freeze_reading_scope,
    previous_reading_analysis,
    reading_followup_scope,
    reading_sources,
    reading_target_documents,
)
from app.foreign_integration.literature import LiteratureMaterialTool
from app.foreign_integration.retrieval import SourcePoint
from app.foreign_integration.schema import (
    foreign_corpus,
    research_citation,
    research_conversation,
    research_message,
    research_run,
    research_step,
)
from app.foreign_integration.service import RunExecutor, create_run


@pytest.fixture
def store():
    """建立三篇公开材料和现有AI运行所需的隔离数据库。"""
    yield from ai_store.__wrapped__()


@pytest.fixture
def client(store):
    """复用真实HTTP路由，所有数据库及身份均为测试替身。"""
    yield from ai_client.__wrapped__(store)


def document(key):
    """生成与真实题录契约一致的展示数据。"""
    return {
        "corpus_id": key, "title": f"双减文献{key}", "author": "",
        "year": 2025, "source": "", "snippet": "",
    }


def seed_source(store, intent="SEARCH", conversation=None):
    """保存前一轮题录或实际引用，故意让候选集合大于引用集合。"""
    accepted = start_ai(store, conversation=conversation)
    payload = {
        "intent": intent, "documents": [document(11), document(12)],
        "citations": [{"corpus_id": 12, "document_title": "双减文献12"}],
    }
    if intent == "SEARCH":
        payload["search_plan"] = {"concepts": [["双减"]]}
    store.session.execute(update(research_run).where(
        research_run.c.id == accepted["run_id"]
    ).values(status="SUCCESS", final_answer="已有研究结果", result_payload={
        "result_type": "AI_ASSISTANT", "payload": payload,
    }, created_at=datetime(2026, 1, 1)))
    store.session.commit()
    return accepted


def prepare_materials(store):
    """补齐源文献，保留额外第三篇以验证阅读不会退回父全库范围。"""
    store.session.execute(insert(foreign_corpus), [
        {"id": key, "title_cn": f"双减文献{key}", "status": 2, "is_deleted": False,
         "is_vectorized": True, "attachment": json.dumps([{"file_path": "/test.pdf"}])}
        for key in (12, 13)
    ])
    store.session.execute(update(foreign_corpus).values(
        attachment=json.dumps([{"file_path": "/test.pdf"}])
    ))
    store.session.commit()


def next_run(store, source, query="基于上述文献分析", context=None, reading=None):
    """在原AI会话创建自然语言追问，不传入独立文献阅读入口参数。"""
    accepted = create_run(
        store.session, actor=store.actor, code="QIMING", request_id=uuid.uuid4().hex,
        knowledge_base_id=None, query=query, conversation_id=source["conversation_id"],
        mode="FAST", options={
            "corpus_ids": [], **({"reading_context": context} if context else {}),
            **({"reading": reading} if reading else {}),
        },
        entry_source="AI",
    )
    row = store.session.execute(select(research_run).where(
        research_run.c.id == accepted["run_id"]
    )).mappings().one()
    conversation = store.session.execute(select(research_conversation).where(
        research_conversation.c.id == accepted["conversation_id"],
    )).mappings().one()
    return accepted, {
        **dict(row), "knowledge_base_id": conversation["knowledge_base_id"],
        "database_id": conversation["database_id"], "entry_source": conversation["entry_source"],
    }


def test_sources_distinguish_search_candidates_and_actual_citations(store):
    """搜索按候选顺序，研究回答默认实际引用而非全部Evidence候选。"""
    prepare_materials(store)
    source = seed_source(store, "QA")
    _, run = next_run(store, source)
    groups = reading_sources(store.session, run, lexical_index="test")
    assert groups[0]["kind"] == "CITATIONS"
    assert [doc["corpus_id"] for doc in groups[0]["documents"]] == [12]
    state = {"reading_sources": groups, "reading_task": "ASK"}
    scope = freeze_reading_scope(store.session, run, state, {11, 12, 13})
    assert scope["corpus_ids"] == [12]
    selected = freeze_reading_scope(store.session, run, {
        **state, "reading_source_kind": "CANDIDATES", "reading_positions": [2, 1],
    }, {11, 12, 13})
    assert selected["corpus_ids"] == [12, 11]


def test_scope_clarifies_missing_ambiguous_or_invalid_references(store):
    """无来源、超量或序号越界时澄清；越权材料不能被模型选入。"""
    prepare_materials(store)
    source = seed_source(store)
    _, run = next_run(store, source)
    groups = reading_sources(store.session, run, lexical_index="test")
    for overrides in [
        {"reading_sources": []}, {"reading_positions": [0]}, {"reading_positions": [3]},
        {"reading_positions": [1, 1]}, {"reading_source_run_id": "f" * 32},
        {"needs_clarification": True},
        {"reading_sources": [{**groups[0], "documents": [document(i) for i in range(1, 22)]}]},
    ]:
        scope = freeze_reading_scope(store.session, run, {
            "reading_sources": groups, **overrides,
        }, {11, 12, 13})
        assert scope["status"] == "NEEDS_INPUT"
    with pytest.raises(PermissionError):
        freeze_reading_scope(store.session, run, {"reading_sources": groups}, {11})


@pytest.mark.parametrize("overrides,expected", [
    ({}, [12]),
    ({"reading_source_kind": "CANDIDATES"}, [12]),
    ({"reading_reference": "SELECTED"}, [12]),
    ({"reading_reference": "LIST", "reading_source_kind": "CANDIDATES"}, [13, 12]),
    ({"reading_reference": "LIST", "reading_source_kind": "SELECTED"}, [13, 12]),
    ({"reading_reference": "SPECIFIC", "reading_positions": [1]}, [13]),
    ({"reading_positions": [2, 1]}, [12, 13]),
    ({"reading_all_results": True}, [13, 12]),
])
def test_selection_default_and_explicit_reference_override(store, overrides, expected):
    """泛指在服务端固定为勾选集合；整页、篇名对应序号及全部结果覆盖默认选择。"""
    prepare_materials(store)
    source = seed_source(store)
    _, run = next_run(store, source, context={
        "source_run_id": source["run_id"], "page": 1, "selected_ids": [12],
    })
    groups = reading_sources(store.session, run, lexical_index="test")
    scope = freeze_reading_scope(store.session, run, {
        "reading_sources": groups, "reading_task": "ANALYZE", **overrides,
    }, {11, 12, 13})
    assert scope["corpus_ids"] == expected


def test_missing_selection_or_unresolved_title_never_expands_to_entire_list(store):
    """明确所选但未选、点名却未定位时询问；普通泛指无勾选仍使用当前列表。"""
    prepare_materials(store)
    source = seed_source(store)
    _, run = next_run(store, source)
    groups = reading_sources(store.session, run, lexical_index="test")
    for reference in ["SELECTED", "SPECIFIC"]:
        scope = freeze_reading_scope(store.session, run, {
            "reading_sources": groups, "reading_reference": reference,
        }, {11, 12, 13})
        assert scope["status"] == "NEEDS_INPUT" and not scope["source_documents"]
    assert freeze_reading_scope(store.session, run, {
        "reading_sources": groups,
    }, {11, 12, 13})["corpus_ids"] == [11, 12]


def test_page_hint_and_cross_page_selection_are_revalidated(store):
    """当前页序号和跨页勾选均由原查询重建，伪造来源与非命中文献被拒绝。"""
    prepare_materials(store)
    store.session.execute(insert(foreign_corpus), [
        {"id": key, "title_cn": f"双减文献{key}", "status": 2, "is_deleted": False}
        for key in range(20, 31)
    ])
    store.session.commit()
    source = seed_source(store)
    _, run = next_run(store, source, context={
        "source_run_id": source["run_id"], "page": 2, "selected_ids": [30, 12],
    })
    groups = reading_sources(store.session, run, lexical_index="test")
    assert groups[0]["kind"] == "CANDIDATES" and len(groups[0]["documents"]) == 3
    assert groups[0]["page"] == 2
    assert [doc["corpus_id"] for doc in groups[1]["documents"]] == [30, 12]
    specific = freeze_reading_scope(store.session, run, {
        "reading_sources": groups, "reading_source_kind": "CANDIDATES",
        "reading_reference": "SPECIFIC", "reading_positions": [1],
    }, {11, 12, 13, *range(20, 31)})
    assert specific["corpus_ids"] == [groups[0]["documents"][0]["corpus_id"]]
    scope = freeze_reading_scope(store.session, run, {
        "reading_sources": groups, "reading_source_kind": "CANDIDATES",
    }, {11, 12, 13, *range(20, 31)})
    assert scope["corpus_ids"] == [30, 12] and scope["source_kind"] == "SELECTED"
    run["request_payload"] = {"reading_context": {
        "source_run_id": "f" * 32, "page": 1, "selected_ids": [],
    }}
    with pytest.raises(ValueError, match="不属于当前会话"):
        reading_sources(store.session, run, lexical_index="test")


@pytest.mark.parametrize("task,intent,positions,selected_ids,expected", [
    ("ANALYZE", "SEARCH", [], [], [11, 12]),
    ("ANALYZE", "SEARCH", [], [12], [12]),
    ("ASK", "SEARCH", [], [], [11, 12]),
    ("ASK", "SEARCH", [2], [], [12]),
    ("ASK", "QA", [], [], [12]),
])
def test_real_reading_graph_publishes_in_same_ai_conversation(
    store, monkeypatch, task, intent, positions, selected_ids, expected
):
    """真实执行阅读图、步骤和发布事务，验证子集、引用及可恢复的后续阅读来源。"""
    prepare_materials(store)
    source = seed_source(store, intent)
    accepted, run = next_run(store, source, context={
        "source_run_id": source["run_id"], "page": 1, "selected_ids": selected_ids,
    })
    loaded = []
    gateway_options = []
    classifier_calls = []

    class ReadingGateway(FakeGateway):
        """任务识别固定，其余分析和引用生成使用现有离线阅读网关。"""

        async def structured(self, instruction, schema, **kwargs):
            """检查分类器确实获得来源，再运行真实分析输出契约。"""
            if schema is AssistantIntent:
                classifier_calls.append(instruction)
                assert len(classifier_calls) == 1, "显式追问不能再次调用任务分类模型"
                assert source["run_id"] in instruction
                return schema(
                    intent="READING", search_query=run["query"], reading_task=task,
                    reading_positions=positions,
                )
            return await super().structured(instruction, schema, **kwargs)

    class Materials(LiteratureMaterialTool):
        """只替换索引IO，保留真实冻结版本检查及发布来源映射。"""

        async def load_materials(self):
            """提供本轮绑定子集的原文，第三篇全库材料不会进入阅读。"""
            self._selection()
            loaded.append(self.document_order)
            items = [evidence(key, key) for key in self.document_order]
            for item in items:
                self.sources[item.chunk_id] = SourcePoint(
                    corpus_id=item.document_id.int, point_id=str(item.chunk_id),
                    chunk_key=str(item.chunk_id), file_path="/test.pdf", content=item.content,
                    page_start=1, page_end=1, score=1,
                )
            return ([{
                "corpus_id": key, "title": f"文献{key}", "author": "", "status": "READY",
                "reason": "", "chunk_count": 1,
            } for key in self.document_order], items, [citation(item) for item in items])

        async def recommend(self, readings):
            """推荐不访问外部服务，不影响本次阅读分析。"""
            return {"documents": [], "status": "SUCCESS"}

    def factory():
        """每个线程使用独立短数据库会话。"""
        return lambda: Session(store.engine)

    def fail(executor, error):
        """直接暴露执行失败，避免脱敏掩盖断言原因。"""
        raise error

    def gateway_factory(*args, **kwargs):
        """记录阅读专用预算，确保不沿用普通问答较短的输出上限。"""
        gateway_options.append(kwargs)
        return ReadingGateway()

    monkeypatch.setattr("app.foreign_integration.service.session_factory", factory)
    monkeypatch.setattr("app.foreign_integration.literature.session_factory", factory)
    monkeypatch.setattr("app.foreign_integration.service.LiteratureMaterialTool", Materials)
    monkeypatch.setattr("app.foreign_integration.service.get_model_gateway", gateway_factory)
    monkeypatch.setattr("app.foreign_integration.service.get_foreign_settings", lambda: (
        SimpleNamespace(qdrant_collection="test", opensearch_index="test",
                        literature_analysis_concurrency=2)
    ))
    monkeypatch.setattr(RunExecutor, "_fail", fail)
    asyncio.run(RunExecutor(accepted["run_id"]).execute())
    store.session.expire_all()
    saved = store.session.execute(select(research_run).where(
        research_run.c.id == accepted["run_id"]
    )).mappings().one()
    payload = saved["result_payload"]["payload"]
    assert saved["status"] == "SUCCESS", saved["error_message"]
    assert saved["output_language"] == "ZH"
    assert saved["result_payload"]["result_type"] == "AI_ASSISTANT"
    assert payload["intent"] == "READING" and payload["reading_result"]["task"] == task
    assert payload["reading_result"]["corpus_ids"] == expected
    if task == "ASK":
        # 子图明确回答第二篇时，公开候选列表只展示实际目标，不以父集合两篇误报。
        assert [doc["corpus_id"] for doc in payload["documents"]] == [12]
    assert loaded == [expected]
    assert [options["max_tokens"] for options in gateway_options] == [6000, 16000]
    assert store.session.scalar(select(research_conversation.c.entry_source)) == "AI"
    assert len(list(store.session.scalars(select(research_conversation)))) == 1
    assert len(list(store.session.scalars(select(research_message).where(
        research_message.c.run_id == accepted["run_id"]
    )))) == 2
    steps = store.session.execute(select(research_step).where(
        research_step.c.run_id == accepted["run_id"]
    )).mappings().all()
    scope_step = next(step for step in steps if step["node_name"] == "ResolveReadingScope")
    assert scope_step["output_summary"]["reading_scope"]["corpus_ids"] == expected
    assert bool(list(store.session.scalars(select(research_citation)))) == (task == "ASK")
    if task == "ANALYZE":
        assert previous_reading_analysis(store.session, run, payload["reading_result"])
    # 下一轮只依赖已保存结果恢复阅读范围，不要求浏览器缓存仍在。
    store.session.execute(update(research_run).where(research_run.c.id == saved["id"]).values(
        created_at=datetime(2026, 1, 1) + timedelta(seconds=1)
    ))
    store.session.commit()
    _, following = next_run(store, source, "继续分析不足")
    groups = reading_sources(store.session, following, lexical_index="test")
    assert groups[0]["kind"] == "READING"
    assert [doc["corpus_id"] for doc in groups[0]["documents"]] == (
        [12] if task == "ASK" else expected
    )
    if task == "ANALYZE":
        # 用推荐按钮的显式契约复用已创建追问，完整验证ASK子图及原分析上下文。
        store.session.execute(update(research_run).where(
            research_run.c.id == following["id"]
        ).values(query="如何构建基于大数据的教育评价新体系？", request_payload={
            **following["request_payload"],
            "reading": {"source_run_id": saved["id"], "task": "ASK"},
        }))
        store.session.commit()
        asyncio.run(RunExecutor(following["id"]).execute())
        store.session.expire_all()
        continued = store.session.scalar(select(research_run.c.result_payload).where(
            research_run.c.id == following["id"]
        ))["payload"]["reading_result"]
        assert continued["task"] == "ASK" and continued["source_run_id"] == saved["id"]
        assert continued["corpus_ids"] == expected and len(classifier_calls) == 1
        assert loaded == [expected, expected]
        child_scope = {**following, "request_payload": {
            "reading": {"source_run_id": saved["id"], "task": "ASK"},
        }}
        assert previous_reading_analysis(store.session, child_scope, continued) == {
            "overview": payload["reading_result"]["overview"],
        }


def seed_reading_card(store):
    """保存带材料版本的成功分析卡片，供HTTP和来源校验测试复用。"""
    source = seed_source(store)
    scope = freeze_reading_scope(store.session, {
        "user_id": store.actor.id, "conversation_id": source["conversation_id"],
        "request_payload": {}, "entry_source": "AI", "knowledge_base_id": None,
        "database_id": None,
    }, {"reading_sources": [{"source_run_id": source["run_id"], "kind": "CANDIDATES",
                              "documents": [document(11), document(12)]}]}, {11, 12, 13})
    store.session.execute(update(research_run).where(research_run.c.id == source["run_id"]).values(
        result_payload={"result_type": "AI_ASSISTANT", "payload": {
            "intent": "READING", "reading_result": {**scope, "status": "SUCCESS",
                                                      "overview": {"overview": []}},
        }},
    ))
    store.session.commit()
    return source


def test_explicit_followup_rejects_cross_context_and_reads_current_material(store):
    """追问仅使用本人同会话阅读卡片；材料变更按新版本读取，不能回退全库。"""
    prepare_materials(store)
    source = seed_reading_card(store)
    scope = reading_followup_scope(
        store.session, 7, source["conversation_id"], source["run_id"]
    )
    assert scope["task"] == "ASK" and scope["corpus_ids"] == [11, 12]
    for actor, conversation in [(8, source["conversation_id"]), (7, "f" * 32)]:
        with pytest.raises(ValueError):
            reading_followup_scope(store.session, actor, conversation, source["run_id"])
    search = seed_source(store, conversation=source["conversation_id"])
    with pytest.raises(ValueError):
        reading_followup_scope(store.session, 7, source["conversation_id"], search["run_id"])
    store.session.execute(update(foreign_corpus).where(foreign_corpus.c.id == 12).values(
        attachment='[{"file_path":"/replaced.pdf"}]',
    ))
    store.session.commit()
    updated = reading_followup_scope(store.session, 7, source["conversation_id"], source["run_id"])
    assert updated["corpus_ids"] == [11, 12]
    assert updated["scope_version"] != scope["scope_version"]


def test_ask_target_projection_keeps_next_ordinal_and_explicit_followup_consistent(store):
    """旧快照父集合两篇而目标一篇时，展示、来源恢复、第一篇和按钮追问都只用该篇。"""
    prepare_materials(store)
    source = seed_reading_card(store)
    payload = store.session.scalar(select(research_run.c.result_payload).where(
        research_run.c.id == source["run_id"]
    ))
    reading = payload["payload"]["reading_result"]
    reading.update(task="ASK", target_corpus_ids=[12])
    store.session.execute(update(research_run).where(research_run.c.id == source["run_id"]).values(
        result_payload=payload,
    ))
    store.session.commit()
    assert [doc["corpus_id"] for doc in reading_target_documents(reading)] == [12]
    _, run = next_run(store, source, "第一篇如何分析难点？", context={
        "source_run_id": source["run_id"], "page": 1, "selected_ids": [],
    })
    groups = reading_sources(store.session, run, lexical_index="test")
    assert [doc["corpus_id"] for doc in groups[0]["documents"]] == [12]
    scope = freeze_reading_scope(store.session, run, {
        "reading_sources": groups, "reading_positions": [1], "reading_task": "ASK",
    }, {11, 12, 13})
    assert scope["corpus_ids"] == [12]
    continued = reading_followup_scope(
        store.session, 7, source["conversation_id"], source["run_id"]
    )
    assert continued["corpus_ids"] == [12] and continued["scope_version"] == scope["scope_version"]
    with pytest.raises(ValueError):
        reading_target_documents({**reading, "target_corpus_ids": [999]})


def test_http_explicit_followup_uses_source_card_and_rejects_mixed_operations(
    store, client, monkeypatch
):
    """真实API接受ASK操作并保存来源；伪造来源及混入写作/页面提示均被拒绝。"""
    prepare_materials(store)
    source = seed_reading_card(store)

    async def skip_execution(run_id):
        """只测试HTTP创建和校验，阅读执行由完整图测试覆盖。"""

    monkeypatch.setattr("app.foreign_integration.api._execute_run", skip_execution)
    base = {"entry_source": "AI", "query": "教师专业发展中哪些培训模块有效？",
            "conversation_id": source["conversation_id"]}
    reading = {"source_run_id": source["run_id"], "task": "ASK"}
    invalid_options = [
        {"reading": {**reading, "task": "ANALYZE"}},
        {"reading": {**reading, "source_run_id": "f" * 32}},
        {"reading": reading, "writing": {"knowledge_base_id": 9, "stage": "PLAN"}},
        {"reading": reading, "reading_context": {"source_run_id": source["run_id"]}},
        {"reading": reading, "corpus_ids": [11]},
    ]
    for options in invalid_options:
        response = client.post("/api/v1/agents/QIMING/runs", json={
            **base, "request_id": uuid.uuid4().hex, "options": options,
        })
        assert response.status_code in {403, 409, 422}
    response = client.post("/api/v1/agents/QIMING/runs", json={
        **base, "request_id": uuid.uuid4().hex, "options": {"reading": reading},
    })
    assert response.status_code == 202, response.text
    run_id = response.json()["data"]["run_id"]
    assert store.session.scalar(select(research_run.c.request_payload).where(
        research_run.c.id == run_id
    ))["reading"] == reading


def test_http_context_validation_preserves_writing_boundary(client, monkeypatch):
    """来源提示通过API严格校验；阅读提示不能混进知识库写作参数。"""
    async def skip_execution(run_id):
        """仅验证HTTP参数，图执行由真实事务测试覆盖。"""

    monkeypatch.setattr("app.foreign_integration.api._execute_run", skip_execution)
    base = {"request_id": uuid.uuid4().hex, "entry_source": "AI", "query": "分析上述文献"}
    context = {"source_run_id": "a" * 32, "page": 1, "selected_ids": []}
    for invalid in [{**context, "page": 0}, {**context, "selected_ids": [11, 11]},
                    {**context, "selected_ids": [True]},
                    {**context, "source_run_id": "bad"}]:
        response = client.post("/api/v1/agents/QIMING/runs", json={
            **base, "options": {"reading_context": invalid},
        })
        assert response.status_code == 422
    response = client.post("/api/v1/agents/QIMING/runs", json={
        **base, "options": {"reading_context": context,
                            "writing": {"knowledge_base_id": 9, "stage": "PLAN"}},
    })
    assert response.status_code == 422
    assert AssistantReadingContext.model_validate(context).page == 1


def test_explicit_old_card_survives_source_window_and_preserves_citation_numbers(store):
    """旧卡片超过默认来源窗口仍能恢复；同篇多个引用编号不改变篇目顺序。"""
    prepare_materials(store)
    first = seed_source(store, "QA")
    original = store.session.execute(select(research_run.c.result_payload).where(
        research_run.c.id == first["run_id"]
    )).scalar_one()
    original["payload"]["citations"] = [
        {"corpus_id": 12, "citation_index": 2, "document_title": "文献12"},
        {"corpus_id": 12, "citation_index": 4, "document_title": "文献12"},
    ]
    store.session.execute(update(research_run).where(research_run.c.id == first["run_id"]).values(
        result_payload=original,
    ))
    store.session.commit()
    for number in range(11):
        later = seed_source(store, conversation=first["conversation_id"])
        store.session.execute(update(research_run).where(
            research_run.c.id == later["run_id"]
        ).values(created_at=datetime(2026, 1, 2) + timedelta(seconds=number)))
        store.session.commit()
    _, run = next_run(store, first, context={
        "source_run_id": first["run_id"], "page": 1, "selected_ids": [],
    })
    groups = reading_sources(store.session, run, lexical_index="test")
    assert groups[0]["source_run_id"] == first["run_id"]
    assert groups[0]["documents"][0]["citation_indices"] == [2, 4]
    assert len({item["source_run_id"] for item in groups}) == 11


def test_all_results_requires_selection_and_publish_uses_acquired_snapshot(
    store, monkeypatch
):
    """全部结果不能偷换成当前页；取得材料后题录变化不会让分析发布失败。"""
    prepare_materials(store)
    source = seed_source(store)
    accepted, run = next_run(store, source)
    groups = reading_sources(store.session, run, lexical_index="test")
    groups[0]["total"] = 50
    clarified = freeze_reading_scope(store.session, run, {
        "reading_sources": groups, "reading_all_results": True,
    }, {11, 12, 13})
    assert clarified["status"] == "NEEDS_INPUT" and "50" in clarified["message"]
    scope = freeze_reading_scope(store.session, run, {"reading_sources": groups}, {11, 12, 13})
    executor = RunExecutor(accepted["run_id"])
    executor.retrieval = SimpleNamespace(sources={})
    monkeypatch.setattr("app.foreign_integration.service.session_factory", lambda: (
        lambda: Session(store.engine)
    ))
    store.session.execute(update(research_run).where(
        research_run.c.id == accepted["run_id"],
    ).values(status="RUNNING"))
    store.session.execute(update(foreign_corpus).values(title_cn="生成期间更新的题录"))
    store.session.commit()
    executor._publish(run, {
        "intent": "READING", "final_answer": "已完成综合分析", "validated_citations": [],
        "reading_result": {**scope, "status": "SUCCESS", "task": "ANALYZE"},
    })
    assert len(list(store.session.scalars(select(research_message).where(
        research_message.c.run_id == accepted["run_id"], research_message.c.role == "ASSISTANT",
    )))) == 1
