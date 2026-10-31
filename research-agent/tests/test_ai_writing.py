"""验证 AI 知识库写作的任务识别、权限边界、阶段产物及同会话发布，不调用在线服务。"""

import asyncio
import json
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from sqlalchemy import insert, select, update
from sqlalchemy.orm import Session
from test_ai_assistant import Gateway
from test_ai_assistant import client as ai_client
from test_ai_assistant import store as ai_store
from test_literature import citation, evidence

from app.agents.ai_assistant_graph import AssistantGraph
from app.agents.ai_assistant_state import AssistantState
from app.foreign_integration.ai_writing import existing_writing_action, writing_artifact
from app.foreign_integration.literature import LiteratureMaterialTool
from app.foreign_integration.retrieval import SourcePoint
from app.foreign_integration.schema import (
    foreign_corpus,
    foreign_knowledge_base,
    foreign_knowledge_base_document,
    research_conversation,
    research_run,
    research_run_scope,
    research_writing_artifact,
)
from app.foreign_integration.service import RunExecutor, create_run


@pytest.fixture
def store():
    """建立本人知识库及公共文献的隔离数据。"""
    yield from ai_store.__wrapped__()


@pytest.fixture
def client(store):
    """挂载真实 API，身份固定为测试用户。"""
    yield from ai_client.__wrapped__(store)


def start(store, writing, conversation=None, request_id=None):
    """创建保留 AI 入口、按写作知识库冻结范围的真实 Run。"""
    return create_run(
        store.session,
        actor=store.actor,
        code="QIMING",
        request_id=request_id or uuid.uuid4().hex,
        knowledge_base_id=None,
        query="双减政策的教师支持研究",
        conversation_id=conversation,
        mode="FAST",
        options={"corpus_ids": [], "writing": writing},
        entry_source="AI",
    )


def test_planning_requires_library_and_does_not_retrieve():
    """自然语言写作先形成任务卡，不能拿全库证据冒充知识库写作。"""
    retrieval = SimpleNamespace(search=AsyncMock())

    async def tracked(name, state, operation):
        """执行真实图节点但省略持久化。"""
        return await operation()

    final = asyncio.run(
        AssistantGraph(retrieval, Gateway("WRITING"), "证据优先", tracked).run(
            AssistantState(
                request_id=uuid.uuid4().hex,
                user_id="7",
                conversation_id=uuid.uuid4().hex,
                query="生成双减论文提纲",
                knowledge_base_id=str(uuid.UUID(int=0)),
                knowledge_base_version=1,
                scope_type="GLOBAL",
                history=[],
                mode="FAST",
            )
        )
    )
    assert final["assistant_result"]["writing_result"]["status"] == "NEEDS_INPUT"
    assert final["intent"] == "WRITING"
    retrieval.search.assert_not_awaited()


def test_records_writing_library_and_allows_membership_changes(store, monkeypatch):
    """只记录写作知识库，不复制库外或库内成员；幂等、越权和资源身份仍受控。"""
    store.session.execute(
        insert(foreign_corpus).values(
            id=12, title_cn="库外文献", status=2, is_vectorized=True, is_deleted=False
        )
    )
    store.session.commit()
    request = uuid.uuid4().hex
    options = {"knowledge_base_id": 9, "stage": "PLAN"}
    accepted = start(store, options, request_id=request)
    assert start(store, options, request_id=request)["created"] is False
    assert set(store.session.scalars(select(research_run_scope.c.corpus_id))) == set()
    assert store.session.scalar(select(research_conversation.c.knowledge_base_id)) is None
    assert store.session.scalar(select(research_conversation.c.entry_source)) == "AI"
    with pytest.raises(PermissionError):
        start(store, {"knowledge_base_id": 999, "stage": "PLAN"})
    store.session.rollback()
    store.session.execute(update(foreign_knowledge_base_document).values(is_deleted=True))
    store.session.commit()
    monkeypatch.setattr(
        "app.foreign_integration.service.session_factory", lambda: lambda: Session(store.engine)
    )
    run, _, _, ids = RunExecutor(accepted["run_id"])._load_context()
    from app.foreign_integration.research_scope import ResearchScope

    scope = ResearchScope.for_run(run)
    assert scope.resource_id == 9 and ids is None
    assert scope.corpus_ids(store.session, 7) == []


@pytest.mark.parametrize(
    "writing",
    [
        {"knowledge_base_id": True, "stage": "PLAN"},
        {"knowledge_base_id": 9, "stage": "UNKNOWN"},
        {"knowledge_base_id": 9, "stage": "PLAN", "owner_id": 8},
    ],
)
def test_http_rejects_untrusted_options(client, writing):
    """阶段及严格 ID 白名单在 API 层生效。"""
    response = client.post(
        "/api/v1/agents/QIMING/runs",
        json={
            "request_id": uuid.uuid4().hex,
            "entry_source": "AI",
            "query": "生成提纲",
            "options": {"writing": writing},
        },
    )
    assert response.status_code == 422


def test_http_accepts_writing_and_rejects_mixed_scope(client, monkeypatch):
    """真实 API 接通知识库写作，不能同时提交数据库或任意选文。"""

    async def skip_execution(run_id):
        """只验证请求落库，执行链由其他集成测试覆盖。"""

    monkeypatch.setattr("app.foreign_integration.api._execute_run", skip_execution)
    body = {
        "request_id": uuid.uuid4().hex,
        "entry_source": "AI",
        "query": "生成论文提纲",
        "options": {"writing": {"knowledge_base_id": 9, "stage": "PLAN"}},
    }
    first = client.post("/api/v1/agents/QIMING/runs", json=body)
    assert first.status_code == 202, first.text
    assert client.post("/api/v1/agents/QIMING/runs", json=body).json()["data"]["created"] is False
    mixed = {
        **body,
        "request_id": uuid.uuid4().hex,
        "options": {**body["options"], "corpus_ids": [11]},
    }
    assert client.post("/api/v1/agents/QIMING/runs", json=mixed).status_code == 422


def test_actual_writing_cannot_use_old_handoff_citation_exemption():
    """实际写作结果缺少引用时必须拒绝发布，旧交接标签不能跳过检查。"""
    executor = RunExecutor("test")
    executor.retrieval = SimpleNamespace()
    with pytest.raises(ValueError, match="Citation"):
        executor._publish(
            {"executor_agent_code": "QIMING", "entry_source": "AI"},
            {
                "final_answer": "未经证据支持的研究结论",
                "intent": "WRITING",
                "writing_result": {"status": "SUCCESS"},
            },
        )


def test_card_action_is_reused_while_pending_and_after_success_but_failed_can_retry(store):
    """更换 request_id 也不能重复执行确认卡；失败或取消允许重新提交。"""
    source = create_run(
        store.session,
        actor=store.actor,
        code="QIMING",
        request_id=uuid.uuid4().hex,
        knowledge_base_id=None,
        query="双减政策的教师支持研究",
        conversation_id=None,
        mode="FAST",
        options={},
        entry_source="AI",
    )
    store.session.execute(
        update(research_run)
        .where(research_run.c.id == source["run_id"])
        .values(
            status="SUCCESS",
            result_payload={
                "result_type": "AI_ASSISTANT",
                "payload": {
                    "writing_result": {
                        "status": "NEEDS_INPUT",
                        "stage": "PLAN",
                        "research_topic": "双减政策的教师支持研究",
                    }
                },
            },
        )
    )
    store.session.commit()
    options = {
        "knowledge_base_id": 9,
        "stage": "PLAN",
        "source_run_id": source["run_id"],
        "source_action": "confirm",
    }
    child = start(store, options, source["conversation_id"])
    for status in ["PENDING", "RUNNING", "SUCCESS"]:
        store.session.execute(
            update(research_run).where(research_run.c.id == child["run_id"]).values(status=status)
        )
        store.session.commit()
        repeated = start(store, options, source["conversation_id"])
        assert repeated["created"] is False and repeated["run_id"] == child["run_id"]
    for status in ["FAILED", "CANCELLED"]:
        store.session.execute(
            update(research_run).where(research_run.c.id == child["run_id"]).values(status=status)
        )
        store.session.commit()
        child = start(store, options, source["conversation_id"])
        assert child["created"] is True
    assert len(list(store.session.scalars(select(research_run)))) == 4


def test_card_action_rejects_forged_source_and_cross_conversation(store):
    """来源 Run 必须是当前会话中的对应卡片，不能只凭一个操作标签去重。"""
    existing = start(store, {"knowledge_base_id": 9, "stage": "PLAN"})
    store.session.execute(update(research_run).values(status="SUCCESS"))
    store.session.commit()
    for source in [uuid.uuid4().hex, existing["run_id"]]:
        with pytest.raises(ValueError, match="来源卡片"):
            start(
                store,
                {
                    "knowledge_base_id": 9,
                    "stage": "PLAN",
                    "source_run_id": source,
                    "source_action": "confirm",
                },
                existing["conversation_id"],
            )
        store.session.rollback()

    other = start(store, {"knowledge_base_id": 9, "stage": "PLAN"})
    store.session.execute(
        update(research_run)
        .where(research_run.c.id == existing["run_id"])
        .values(result_payload={"payload": {"writing_result": {"status": "NEEDS_INPUT"}}})
    )
    store.session.commit()
    with pytest.raises(ValueError, match="来源卡片"):
        start(
            store,
            {
                "knowledge_base_id": 9,
                "stage": "PLAN",
                "source_run_id": existing["run_id"],
                "source_action": "confirm",
            },
            other["conversation_id"],
        )


def test_explicit_task_change_requires_correction_but_unclear_card_allows_choice(store):
    """明确任务不能被提交参数静默覆盖；主动更正和澄清选择均可执行。"""
    source = start(store, {"knowledge_base_id": 9, "stage": "PLAN"})
    card = {"status": "NEEDS_INPUT", "stage": "PLAN", "task_selection_mode": "FIXED"}
    store.session.execute(
        update(research_run)
        .where(research_run.c.id == source["run_id"])
        .values(status="SUCCESS", result_payload={"payload": {"writing_result": card}})
    )
    store.session.commit()
    options = {
        "knowledge_base_id": 9,
        "stage": "RQ",
        "source_run_id": source["run_id"],
        "source_action": "confirm",
    }
    with pytest.raises(ValueError, match="更正任务"):
        start(store, options, source["conversation_id"])
    store.session.rollback()
    child = start(store, {**options, "task_corrected": True}, source["conversation_id"])
    assert child["created"] is True
    store.session.execute(
        update(research_run).where(research_run.c.id == child["run_id"]).values(status="CANCELLED")
    )
    store.session.execute(
        update(research_run)
        .where(research_run.c.id == source["run_id"])
        .values(
            result_payload={
                "payload": {
                    "writing_result": {**card, "stage": None, "task_selection_mode": "CHOOSE"}
                }
            }
        )
    )
    store.session.commit()
    assert start(store, options, source["conversation_id"])["created"] is True


@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize(
    ("stage", "target", "action", "extra"),
    [
        ("RQ", "PLAN", "plan:0", {}),
        ("PLAN", "DRAFT", "draft:1", {"section_ref": "1"}),
        ("DRAFT", "REVISE", "revise", {"revise_instruction": "补充文献支持"}),
    ],
)
def test_stage_actions_reuse_only_the_same_question_section_or_version(
    store, legacy, stage, target, action, extra
):
    """新旧历史均按具体操作去重，不锁住另一选题或另一章节。"""
    source = start(store, {"knowledge_base_id": 9, "stage": "PLAN"})
    topic = "双减政策的教师支持研究"
    artifact_id = uuid.uuid4().hex
    result = {
        "status": "SUCCESS",
        "stage": stage,
        "artifact_id": artifact_id,
        "research_questions": [{"question": topic}, {"question": "另一研究问题"}],
        "sections": [{"section_id": "1"}, {"section_id": "2"}],
    }
    store.session.execute(
        update(research_run)
        .where(research_run.c.id == source["run_id"])
        .values(status="SUCCESS", result_payload={"payload": {"writing_result": result}})
    )
    store.session.commit()
    child = start(store, {"knowledge_base_id": 9, "stage": "PLAN"}, source["conversation_id"])
    options = {
        "knowledge_base_id": 9,
        "stage": target,
        "artifact_id": artifact_id,
        "source_run_id": source["run_id"],
        "source_action": action,
        **extra,
    }
    saved = {
        key: value
        for key, value in options.items()
        if not legacy or key not in {"source_run_id", "source_action"}
    }
    store.session.execute(
        update(research_run)
        .where(research_run.c.id == child["run_id"])
        .values(status="SUCCESS", request_payload={"writing": saved})
    )
    store.session.commit()
    assert (
        existing_writing_action(store.session, source["conversation_id"], 7, options, topic)["id"]
        == child["run_id"]
    )
    if stage in {"RQ", "PLAN"}:
        changed = {**options, "source_action": "plan:1" if stage == "RQ" else "draft:2"}
        if stage == "PLAN":
            changed["section_ref"] = "2"
        assert (
            existing_writing_action(
                store.session,
                source["conversation_id"],
                7,
                changed,
                "另一研究问题" if stage == "RQ" else topic,
            )
            is None
        )


def test_executor_saves_nested_artifact_and_rejects_cross_context(store, client, monkeypatch):
    """真实执行发布引用与嵌套阶段产物，并拒绝跨库、跨会话和错误章节。"""
    item = evidence(1, 11)
    store.session.execute(
        update(foreign_corpus).values(attachment=json.dumps([{"file_path": "/test.pdf"}]))
    )
    store.session.execute(
        insert(foreign_knowledge_base).values(id=10, owner_id=7, is_deleted=False)
    )
    store.session.commit()
    accepted = start(
        store, {"knowledge_base_id": 9, "stage": "PLAN", "writing_requirements": "分三级标题"}
    )

    class Retrieval(LiteratureMaterialTool):
        """替换外部查询，保留固定范围及真实引用来源检查。"""

        async def search(self, **kwargs):
            """登记发布事务可回查的 Evidence。"""
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
            return SimpleNamespace(evidence=[item], citations=[citation(item)])

    async def plan(graph, state):
        """模拟现有 Quill 子图的已校验输出，验证委派及数据库事务。"""
        assert state["research_topic"] == "双减政策的教师支持研究"
        assert state["writing_requirements"] == "分三级标题"
        await graph.retrieval.search()
        return {
            "final_answer": "理论基础[1]",
            "evidence": [item],
            "validated_citations": [citation(item)],
            "citations": [citation(item)],
            "evidence_level": "LOW",
            "used_citation_indices": [1],
            "plan_result": {
                "status": "SUCCESS",
                "stage": "PLAN",
                "evidence_level": "LOW",
                "research_topic": "双减研究",
                "sections": [
                    {"section_id": "1", "title": "理论基础", "writing_points": ["理论依据"]}
                ],
            },
        }

    def fail(executor, error):
        """暴露实际执行异常，避免脱敏消息掩盖测试失败。"""
        raise error

    def factory():
        """为执行器和材料工具提供独立的短事务会话。"""
        return lambda: Session(store.engine)

    monkeypatch.setattr("app.foreign_integration.service.session_factory", factory)
    monkeypatch.setattr("app.foreign_integration.literature.session_factory", factory)
    monkeypatch.setattr("app.foreign_integration.service.LiteratureMaterialTool", Retrieval)
    monkeypatch.setattr(
        "app.foreign_integration.service.get_model_gateway", lambda *a, **k: Gateway("WRITING")
    )
    monkeypatch.setattr(
        "app.foreign_integration.service.get_foreign_settings",
        lambda: SimpleNamespace(qdrant_collection="test", opensearch_index="foreign_chunks_v1"),
    )
    monkeypatch.setattr("app.agents.quill_plan_graph.QuillPlanGraph.run", plan)
    monkeypatch.setattr(RunExecutor, "_fail", fail)
    asyncio.run(RunExecutor(accepted["run_id"]).execute())
    store.session.expire_all()
    row = store.session.execute(select(research_run)).mappings().one()
    assert row["status"] == "SUCCESS" and row["executor_agent_code"] == "QUILL"
    payload = row["result_payload"]["payload"]["writing_result"]
    artifact = store.session.execute(select(research_writing_artifact)).mappings().one()
    assert artifact["id"] == payload["artifact_id"]
    assert artifact["citation_ids"] == [1]
    assert artifact["conversation_id"] == accepted["conversation_id"]
    options = {
        "knowledge_base_id": 9,
        "stage": "DRAFT",
        "artifact_id": artifact["id"],
        "section_ref": "1",
    }
    assert writing_artifact(store.session, accepted["conversation_id"], 7, options)["sections"]
    for conversation, changes in [
        (uuid.uuid4().hex, {}),
        (accepted["conversation_id"], {"knowledge_base_id": 10}),
        (accepted["conversation_id"], {"section_ref": "99"}),
    ]:
        with pytest.raises(ValueError):
            writing_artifact(store.session, conversation, 7, {**options, **changes})

    async def draft(graph, state):
        """核对委派的章节来自真实提纲，并发布下一阶段的原稿。"""
        assert state["section_ref"] == "1"
        assert state["section"]["title"] == "理论基础"
        await graph.retrieval.search()
        return {
            "final_answer": "章节原稿[1]",
            "evidence": [item],
            "citations": [citation(item)],
            "validated_citations": [citation(item)],
            "used_citation_indices": [1],
            "evidence_level": "LOW",
            "draft_result": {
                "status": "SUCCESS",
                "stage": "DRAFT",
                "evidence_level": "LOW",
                "section_ref": "1",
                "research_topic": "双减研究",
                "draft_text": "章节原稿[1]",
            },
        }

    async def revise(graph, state):
        """学术修改读取保存的原稿，不能把外层主题当作待改原文。"""
        assert state["query"] == "章节原稿[1]"
        assert state["revise_instruction"] == "补充文献支持"
        result = await draft(graph, {**state, "section_ref": "1", "section": {"title": "理论基础"}})
        result["revise_result"] = {
            **result.pop("draft_result"),
            "stage": "REVISE",
            "revised_text": "修改后的章节[1]",
        }
        return result

    monkeypatch.setattr("app.agents.quill_draft_graph.QuillDraftGraph.run", draft)
    monkeypatch.setattr("app.agents.quill_revise_graph.QuillReviseGraph.run", revise)
    next_run = start(store, options, accepted["conversation_id"])
    asyncio.run(RunExecutor(next_run["run_id"]).execute())
    store.session.expire_all()
    drafted = (
        store.session.execute(select(research_run).where(research_run.c.id == next_run["run_id"]))
        .mappings()
        .one()
    )
    draft_id = drafted["result_payload"]["payload"]["writing_result"]["artifact_id"]
    modified = start(
        store,
        {
            "knowledge_base_id": 9,
            "stage": "REVISE",
            "artifact_id": draft_id,
            "revise_instruction": "补充文献支持",
        },
        accepted["conversation_id"],
    )
    asyncio.run(RunExecutor(modified["run_id"]).execute())
    store.session.expire_all()
    assert len(list(store.session.scalars(select(research_writing_artifact)))) == 3
    exported = client.post("/api/v1/quill/export-docx", json={"artifact_id": artifact["id"]})
    assert exported.status_code == 200 and exported.content.startswith(b"PK")
