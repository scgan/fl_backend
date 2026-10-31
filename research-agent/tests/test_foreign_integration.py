"""验证 Foreign 共库运行的外键边界、幂等、伪删除与真实来源页码。"""

from __future__ import annotations

import asyncio
import json
import unittest
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import jwt
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, insert, select, update
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from app.agents.gateway import ExtractiveModelGateway
from app.foreign_integration.api import router
from app.foreign_integration.auth import ForeignActor, current_actor
from app.foreign_integration.database import get_session
from app.foreign_integration.retrieval import _source_from_hit
from app.foreign_integration.schema import (
    foreign_corpus,
    foreign_knowledge_base,
    foreign_knowledge_base_document,
    foreign_user,
    metadata,
    research_agent,
    research_conversation,
    research_evidence,
    research_message,
    research_model_config,
    research_prompt,
    research_run,
    research_run_scope,
)
from app.foreign_integration.service import (
    RunExecutor,
    create_run,
    read_run,
    soft_delete_conversation,
    soft_delete_run,
)


class ForeignIntegrationTests(unittest.TestCase):
    """在内存库验证共库映射，不接触远程 Foreign 数据。"""

    def setUp(self) -> None:
        """建立 Foreign 主表与研究表的最小可信数据。"""
        self.engine = create_engine(
            "sqlite+pysqlite:///:memory:",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        metadata.create_all(self.engine)
        self.session = Session(self.engine)
        self.session.execute(
            insert(foreign_user).values(
                id=7,
                username="collector",
                role=json.dumps(["2"]),
                status=1,
                is_deleted=False,
            )
        )
        self.session.execute(
            insert(foreign_knowledge_base).values(
                id=9,
                owner_id=7,
                is_deleted=False,
            )
        )
        self.session.execute(
            insert(foreign_corpus).values(
                id=11,
                title_cn="测试文献",
                status=2,
                is_deleted=False,
                is_vectorized=True,
                attachment="[]",
            )
        )
        self.session.execute(
            insert(foreign_knowledge_base_document).values(
                knowledge_base_id=9,
                corpus_id=11,
                is_deleted=False,
            )
        )
        self.session.execute(
            insert(research_model_config).values(
                id="m" * 32,
                provider="OPENAI_COMPATIBLE",
                model_name="test",
                enabled=True,
                is_deleted=False,
            )
        )
        self.session.execute(
            insert(research_agent).values(
                id="a" * 32,
                code="QIMING",
                name="启明",
                description="问答",
                model_config_id="m" * 32,
                enabled=True,
                is_deleted=False,
            )
        )
        self.session.execute(
            insert(research_prompt).values(
                id="p" * 32,
                agent_id="a" * 32,
                version=2,
                system_prompt="只根据证据回答",
                status="ACTIVE",
                is_deleted=False,
            )
        )
        self.session.commit()
        self.actor = ForeignActor(id=7, username="collector", role=2)

    def tearDown(self) -> None:
        """释放内存数据库连接。"""
        self.session.close()
        self.engine.dispose()

    def test_run_list_filters_owner_module_deleted_and_revoked_scope(self) -> None:
        """新列表路由准确分页并隔离他人、其他模块、软删除和撤权摘要。"""
        accepted = []
        for index in range(5):
            result = create_run(
                self.session,
                actor=self.actor,
                code="QIMING",
                request_id=f"history-{index}",
                knowledge_base_id=9,
                query="研究问题",
                conversation_id=None,
                mode="FAST",
                options={},
            )
            accepted.append(result["run_id"])
        self.session.execute(
            update(research_run).values(
                executor_agent_code="SUZHI",
                status="SUCCESS",
                final_answer="受保护的回答",
            )
        )
        self.session.execute(
            update(research_run).where(research_run.c.id == accepted[2]).values(is_deleted=True)
        )
        self.session.execute(
            update(research_run).where(research_run.c.id == accepted[3]).values(user_id=8)
        )
        self.session.execute(
            update(research_run)
            .where(research_run.c.id == accepted[4])
            .values(executor_agent_code="ZHIJIAN")
        )
        self.session.commit()
        app = FastAPI()

        @app.middleware("http")
        async def request_id(request, call_next):
            """为隔离路由测试提供响应信封所需的请求标识。"""
            request.state.request_id = "test-history"
            return await call_next(request)

        app.include_router(router, prefix="/api/v1")
        app.dependency_overrides[current_actor] = lambda: self.actor
        app.dependency_overrides[get_session] = lambda: self.session
        with TestClient(app) as client:
            first = client.get(
                "/api/v1/agent-runs", params={"executor_agent_code": "SUZHI", "page_size": 1}
            )
            self.assertEqual(first.status_code, 200)
            data = first.json()["data"]
            self.assertEqual(data["total"], 2)
            self.assertEqual(len(data["items"]), 1)
            second = client.get(
                "/api/v1/agent-runs",
                params={"executor_agent_code": "SUZHI", "page_size": 1, "page": 2},
            ).json()["data"]
            self.assertNotEqual(data["items"][0]["id"], second["items"][0]["id"])
            self.assertEqual(data["items"][0]["entry_agent_code"], "QIMING")
            self.session.execute(update(foreign_knowledge_base).values(is_deleted=True))
            self.session.commit()
            revoked = client.get(
                "/api/v1/agent-runs", params={"executor_agent_code": "SUZHI"}
            ).json()["data"]
            self.assertTrue(
                all(
                    not item["knowledge_base_accessible"] and not item["summary"]
                    for item in revoked["items"]
                )
            )
            self.assertEqual(
                client.get(
                    "/api/v1/agent-runs", params={"executor_agent_code": "INVALID"}
                ).status_code,
                422,
            )

    def test_professional_create_routes_use_unified_payload(self) -> None:
        """三个专业入口接受新字段并保存 options，后台模型执行在此隔离。"""
        for index, code in enumerate(("ZHIJIAN", "SUZHI", "RONGTONG")):
            agent_id = str(index) * 32
            self.session.execute(
                insert(research_agent).values(
                    id=agent_id,
                    code=code,
                    name=code,
                    description=code,
                    model_config_id="m" * 32,
                    enabled=True,
                    is_deleted=False,
                )
            )
            self.session.execute(
                insert(research_prompt).values(
                    id=str(index + 3) * 32,
                    agent_id=agent_id,
                    version=1,
                    system_prompt="依据证据回答",
                    status="ACTIVE",
                    is_deleted=False,
                )
            )
        self.session.commit()
        app = FastAPI()

        @app.middleware("http")
        async def request_id(request, call_next):
            """为接口测试提供请求标识。"""
            request.state.request_id = "test-create"
            return await call_next(request)

        app.include_router(router, prefix="/api/v1")
        app.dependency_overrides[current_actor] = lambda: self.actor
        app.dependency_overrides[get_session] = lambda: self.session
        with (
            TestClient(app) as client,
            patch("app.foreign_integration.api._execute_run", new_callable=AsyncMock),
        ):
            for code, options in (
                ("zhijian", {}),
                ("suzhi", {"review_length": "SHORT"}),
                ("rongtong", {"target_language": "EN"}),
            ):
                response = client.post(
                    f"/api/v1/agents/{code}/runs",
                    json={
                        "request_id": f"create-{code}",
                        "knowledge_base_id": 9,
                        "query": "根据论文进行研究",
                        "options": options,
                    },
                )
                self.assertEqual(response.status_code, 202, response.text)
                run_id = response.json()["data"]["run_id"]
                run = (
                    self.session.execute(select(research_run).where(research_run.c.id == run_id))
                    .mappings()
                    .one()
                )
                self.assertEqual(run["executor_agent_code"], code.upper())
                self.assertEqual(run["request_payload"], options)

    def test_research_only_accepts_selected_collector_role(self) -> None:
        """即使账号拥有所有角色，研究接口也仅接受当前选择的收集者角色。"""
        secret = "research-role-test-secret-long-enough"
        self.session.execute(
            update(foreign_user).where(foreign_user.c.id == 7)
            .values(role=json.dumps([0, 1, 2, 3]))
        )
        self.session.commit()
        settings = SimpleNamespace(jwt_secret=secret)
        with patch("app.foreign_integration.auth.get_foreign_settings", return_value=settings):
            for role in [0, 1, 2, 3]:
                with self.subTest(role=role):
                    token = jwt.encode(
                        {"username": "collector", "role": str(role)}, secret, algorithm="HS256"
                    )
                    if role == 2:
                        self.assertEqual(current_actor("Bearer " + token, self.session).role, 2)
                    else:
                        with self.assertRaises(HTTPException) as error:
                            current_actor("Bearer " + token, self.session)
                        self.assertEqual(error.exception.status_code, 403)

    def test_run_scope_idempotency_and_soft_delete(self) -> None:
        """同一请求只建一次任务，不复制库成员，软删除仍保留消息事实。"""
        payload = dict(
            actor=self.actor,
            code="QIMING",
            request_id="request-123",
            knowledge_base_id=9,
            query="这篇文献讨论什么？",
            conversation_id=None,
            mode="FAST",
            options={},
        )
        accepted = create_run(self.session, **payload)
        self.assertTrue(accepted["created"])
        self.assertFalse(create_run(self.session, **payload)["created"])
        self.assertEqual(
            read_run(self.session, accepted["run_id"], self.actor)["knowledge_base_id"], 9
        )
        with self.assertRaisesRegex(ValueError, "request_id"):
            create_run(self.session, **{**payload, "knowledge_base_id": 10})
        scope = (
            self.session.execute(
                select(research_run_scope).where(research_run_scope.c.run_id == accepted["run_id"])
            )
            .mappings()
            .first()
        )
        self.assertIsNone(scope)
        with self.assertRaises(ValueError):
            create_run(self.session, **{**payload, "query": "不同问题"})
        self.session.execute(
            update(research_run)
            .where(research_run.c.id == accepted["run_id"])
            .values(status="SUCCESS")
        )
        self.session.commit()
        self.assertEqual(
            read_run(self.session, accepted["run_id"], self.actor)["status"], "SUCCESS"
        )
        soft_delete_run(self.session, accepted["run_id"], self.actor)
        run = (
            self.session.execute(
                select(research_run).where(research_run.c.id == accepted["run_id"])
            )
            .mappings()
            .one()
        )
        message = (
            self.session.execute(
                select(research_message).where(research_message.c.run_id == accepted["run_id"])
            )
            .mappings()
            .one()
        )
        self.assertTrue(run["is_deleted"])
        self.assertTrue(message["is_deleted"])
        with self.assertRaises(LookupError):
            read_run(self.session, accepted["run_id"], self.actor)

    def test_execution_loads_conversation_knowledge_base_and_rechecks_permission(self):
        """无 Run 知识库列时仍可恢复执行配置，撤权后禁止启动检索。"""
        accepted = self._history_for_delete("conversation-execution-context")
        executor = RunExecutor(accepted["run_id"])
        with patch(
            "app.foreign_integration.service.session_factory", return_value=lambda: self.session
        ):
            run, prompt, model, scope = executor._load_context()
            self.assertEqual(run["knowledge_base_id"], 9)
            self.assertEqual(prompt["id"], "p" * 32)
            self.assertEqual(model["id"], "m" * 32)
            self.assertIsNone(scope)
            self.session.execute(update(foreign_knowledge_base).values(owner_id=8))
            self.session.commit()
            with self.assertRaises(PermissionError):
                executor._load_context()

    def test_followup_cannot_switch_conversation_knowledge_base(self):
        """移除 Run 的绑定副本后，后续提问仍受会话固定知识库约束。"""
        accepted = self._history_for_delete("conversation-fixed-knowledge-base")
        self.session.execute(insert(foreign_knowledge_base).values(
            id=10, owner_id=7, is_deleted=False,
        ))
        self.session.commit()
        with self.assertRaisesRegex(ValueError, "不能切换知识库"):
            create_run(
                self.session, actor=self.actor, code="QIMING", request_id="switch-base-followup",
                knowledge_base_id=10, query="继续提问", conversation_id=accepted["conversation_id"],
                mode="FAST", options={},
            )

    def test_execution_rejects_deleted_conversation(self):
        """异步执行重新读取会话状态，不使用已经删除会话的绑定。"""
        accepted = self._history_for_delete("deleted-conversation-execution")
        self.session.execute(update(research_conversation).values(is_deleted=True))
        self.session.commit()
        with patch(
            "app.foreign_integration.service.session_factory", return_value=lambda: self.session
        ):
            with self.assertRaises(LookupError):
                RunExecutor(accepted["run_id"])._load_context()

    def _history_for_delete(self, request_id):
        """建立可用于删除权限回归的已结束运行。"""
        result = create_run(self.session, actor=self.actor, code="QIMING",
                            request_id=request_id, knowledge_base_id=9,
                            query="删除测试", conversation_id=None, mode="FAST", options={})
        self.session.execute(update(research_run).where(
            research_run.c.id == result['run_id']).values(status='SUCCESS'))
        self.session.commit()
        return result

    def test_delete_owned_history_after_base_deleted_or_revoked(self):
        """历史归属当前用户即可读取和清理，不依赖知识库状态。"""
        for scope in ('deleted', 'revoked'):
            for kind in ('run', 'conversation'):
                with self.subTest(scope=scope, kind=kind):
                    self.session.execute(update(foreign_knowledge_base).values(
                        owner_id=7, is_deleted=False))
                    self.session.commit()
                    result = self._history_for_delete(f'{scope}-{kind}')
                    self.session.execute(update(foreign_knowledge_base).values(
                        **({'is_deleted': True} if scope == 'deleted' else {'owner_id': 8})))
                    self.session.commit()
                    self.assertEqual(
                        read_run(self.session, result['run_id'], self.actor)['status'], 'SUCCESS')
                    if kind == 'run':
                        soft_delete_run(self.session, result['run_id'], self.actor)
                    else:
                        soft_delete_conversation(
                            self.session, result['conversation_id'], self.actor)
                    self.assertTrue(self.session.scalar(select(research_run.c.is_deleted).where(
                        research_run.c.id == result['run_id'])))
                    self.assertTrue(all(self.session.scalars(select(research_message.c.is_deleted).where(
                        research_message.c.run_id == result['run_id']))))

    def test_deleted_base_history_readable_but_followup_rejected(self):
        """删除知识库后仍可读取各模块历史和会话，但不能继续提问。"""
        result = self._history_for_delete('history-read-deleted-base')
        self.session.execute(update(research_run).values(
            final_answer='已保存的历史结果', result_payload={'result_type': 'qiming'}))
        self.session.execute(update(foreign_knowledge_base).values(is_deleted=True))
        self.session.commit()
        app = FastAPI()

        @app.middleware('http')
        async def request_id(request, call_next):
            """为已删知识库历史接口测试注入固定请求追踪 ID。"""
            request.state.request_id = 'test-deleted-base'
            return await call_next(request)

        app.include_router(router, prefix='/api/v1')
        app.dependency_overrides[current_actor] = lambda: self.actor
        app.dependency_overrides[get_session] = lambda: self.session
        with TestClient(app) as client:
            for code in ('QIMING', 'ZHIJIAN', 'SUZHI', 'RONGTONG'):
                with self.subTest(module=code):
                    self.session.execute(update(research_run).values(executor_agent_code=code))
                    self.session.commit()
                    response = client.get(f"/api/v1/agent-runs/{result['run_id']}")
                    self.assertEqual(response.status_code, 200)
                    self.assertEqual(response.json()['data']['final_answer'], '已保存的历史结果')
                    self.assertEqual(client.get(
                        f"/api/v1/agent-runs/{result['run_id']}/evidence").status_code, 200)
            response = client.get(f"/api/v1/conversations/{result['conversation_id']}")
            self.assertEqual(response.status_code, 200)
            self.assertTrue(response.json()['data']['messages'])
            with self.assertRaises(PermissionError):
                create_run(self.session, actor=self.actor, code='QIMING',
                           request_id='deleted-base-followup', knowledge_base_id=9,
                           query='继续提问', conversation_id=result['conversation_id'],
                           mode='FAST', options={})

    def test_evidence_preserves_rerank_score_separately_from_rrf(self):
        """保存并读取真正的相关性分，旧证据为空而不是复用 RRF。"""
        from app.foreign_integration.api import get_evidence
        from app.foreign_integration.service import RunExecutor
        result = self._history_for_delete('evidence-relevance-score')
        chunk = uuid.uuid4()
        source = SimpleNamespace(corpus_id=11, point_id=str(chunk), chunk_key='chunk',
                                 file_path='test.pdf', page_start=1, page_end=1,
                                 content='测试证据', score=0.023)
        executor = RunExecutor(result['run_id'])
        executor.retrieval = SimpleNamespace(sources={chunk: source})
        item = SimpleNamespace(chunk_id=chunk, document_id=uuid.UUID(int=11),
                               relevance_score=0.725, citation_index=1)
        with patch('app.foreign_integration.service.get_foreign_settings',
                   return_value=SimpleNamespace(qdrant_collection='test')):
            executor._save_evidence(self.session, {'evidence': [item]})
        self.session.commit()
        request = SimpleNamespace(state=SimpleNamespace(request_id='score-test'))
        evidence = get_evidence(result['run_id'], request, self.actor, self.session)['data'][0]
        self.assertAlmostEqual(evidence['retrieval_score'], 0.023)
        self.assertAlmostEqual(evidence['relevance_score'], 0.725)
        self.session.execute(update(research_evidence).values(relevance_score=None))
        self.session.commit()
        evidence = get_evidence(result['run_id'], request, self.actor, self.session)['data'][0]
        self.assertIsNone(evidence['relevance_score'])

    def test_evidence_score_column_upgrade_preserves_old_rows(self):
        """旧表补列可重复执行，保留原记录且不伪造相关性分。"""
        from app.foreign_integration.migrate import _ensure_evidence_relevance_score
        engine = create_engine('sqlite+pysqlite:///:memory:')
        with engine.begin() as connection:
            connection.exec_driver_sql(
                'CREATE TABLE research_evidence (id TEXT, retrieval_score FLOAT)')
            connection.exec_driver_sql("INSERT INTO research_evidence VALUES ('old', 0.023)")
        _ensure_evidence_relevance_score(engine)
        _ensure_evidence_relevance_score(engine)
        with engine.connect() as connection:
            row = connection.exec_driver_sql('SELECT * FROM research_evidence').mappings().one()
            self.assertEqual(row['id'], 'old')
            self.assertAlmostEqual(row['retrieval_score'], 0.023)
            self.assertIsNone(row['relevance_score'])
        engine.dispose()

    def test_delete_history_rejects_other_or_disabled_user(self):
        """取消知识库门槛不取消记录归属及账号有效性检查。"""
        result = self._history_for_delete('delete-authorization')
        for kind in ('run', 'conversation'):
            delete = soft_delete_run if kind == 'run' else soft_delete_conversation
            key = result['run_id' if kind == 'run' else 'conversation_id']
            with self.assertRaises(PermissionError):
                delete(self.session, key, SimpleNamespace(id=8))
            for values in ({'status': 0}, {'status': 1, 'is_deleted': True}):
                self.session.execute(update(foreign_user).values(**values))
                self.session.commit()
                with self.assertRaises(PermissionError):
                    delete(self.session, key, self.actor)
            self.session.execute(update(foreign_user).values(status=1, is_deleted=False))
            self.session.commit()
        self.assertFalse(self.session.scalar(select(research_run.c.is_deleted).where(
            research_run.c.id == result['run_id'])))

    def test_delete_inaccessible_history_still_rejects_active_runs(self):
        """知识库已删除也不允许删除排队或运行中的任务/会话。"""
        result = self._history_for_delete('delete-active')
        self.session.execute(update(foreign_knowledge_base).values(is_deleted=True))
        for status in ('PENDING', 'RUNNING'):
            self.session.execute(update(research_run).values(status=status))
            self.session.commit()
            with self.assertRaises(ValueError):
                soft_delete_run(self.session, result['run_id'], self.actor)
            with self.assertRaises(ValueError):
                soft_delete_conversation(self.session, result['conversation_id'], self.actor)

    def test_foreign_role_is_rechecked_in_database(self) -> None:
        """仅有正确 JWT 签名不足以使用不属于账号的角色。"""
        secret = "integration-test-secret-at-least-32-bytes"
        token = jwt.encode({"username": "collector", "role": "2"}, secret, algorithm="HS256")
        settings = SimpleNamespace(jwt_secret=secret)
        with patch("app.foreign_integration.auth.get_foreign_settings", return_value=settings):
            actor = current_actor("Bearer " + token, self.session)
            self.assertEqual(actor.id, 7)
            forged_role = jwt.encode(
                {"username": "collector", "role": "0"}, secret, algorithm="HS256"
            )
            with self.assertRaises(HTTPException) as context:
                current_actor("Bearer " + forged_role, self.session)
            self.assertEqual(context.exception.status_code, 403)

    def test_conversation_preserves_each_messages_professional_route(self) -> None:
        """混合会话的三个专业回答各自带正确跳转元数据，普通问答无专业结果。"""
        from fastapi import Request

        from app.foreign_integration.api import get_conversation

        conversation_id = None
        expected = {}
        for index, (code, result_type) in enumerate(
            (
                ("ZHIJIAN", "ZHIJIAN_COMPARISON"),
                ("SUZHI", "SUZHI_LITERATURE_REVIEW"),
                ("RONGTONG", "RONGTONG_POLICY_COMMUNICATION"),
                ("QIMING", None),
            )
        ):
            accepted = create_run(
                self.session,
                actor=self.actor,
                code="QIMING",
                request_id=f"route-test-{index}",
                knowledge_base_id=9,
                query="研究问题",
                conversation_id=conversation_id,
                mode="FAST",
                options={},
            )
            conversation_id = accepted["conversation_id"]
            run_id = accepted["run_id"]
            self.session.execute(
                update(research_run)
                .where(research_run.c.id == run_id)
                .values(
                    status="SUCCESS",
                    executor_agent_code=code,
                    result_payload={"result_type": result_type, "schema_version": 3, "payload": {}}
                    if result_type
                    else None,
                )
            )
            self.session.execute(
                insert(research_message).values(
                    id=str(index) * 32,
                    conversation_id=conversation_id,
                    run_id=run_id,
                    role="ASSISTANT",
                    content="已审核回答",
                    is_deleted=False,
                )
            )
            expected[run_id] = (code, result_type)
            self.session.commit()
        request = Request({"type": "http", "state": {"request_id": "test-routes"}})
        detail = get_conversation(conversation_id, request, self.actor, self.session)["data"]
        answers = [message for message in detail["messages"] if message["role"] == "ASSISTANT"]
        self.assertEqual(len(answers), 4)
        for message in answers:
            self.assertEqual(message["entry_agent_code"], "QIMING")
            self.assertEqual(
                (message["executor_agent_code"], message["result_type"]),
                expected[message["run_id"]],
            )

    def test_mineru_page_and_attachment_are_verified(self) -> None:
        """仅在附件匹配时按 MinerU 的 bbox 页码生成可引用原文页。"""
        hit = SimpleNamespace(
            id=11000000,
            score=0.6,
            payload={
                "file_path": "uploaded/11.pdf",
                "text": "真实段落",
                "page_start": 0,
                "page_end": 1,
                "bbox_blocks": [{"page_idx": 0}, {"page_idx": 1}],
                "chunk_id": "11_0",
            },
        )
        row = {"id": 11, "attachment": json.dumps([{"file_path": "uploaded/11.pdf"}])}
        source = _source_from_hit(hit, row)
        self.assertEqual((source.page_start, source.page_end), (1, 2))
        self.assertIsNone(_source_from_hit(hit, {**row, "attachment": "[]"}))
        hit.payload["bbox_blocks"] = [{"page_idx": 3}]
        self.assertEqual(_source_from_hit(hit, row).page_start, 4)

    def test_stream_replays_when_running_cursor_no_longer_exists(self) -> None:
        """步骤已结束导致旧 RUNNING 游标消失时，重连仍返回快照与终态。"""
        from app.foreign_integration.api import _events

        projection = {
            "steps": [{"sequence": 1, "node_name": "GenerateAnswer", "status": "SUCCESS"}],
            "status": "SUCCESS",
            "final_answer": "已完成",
            "error_code": None,
        }

        async def collect() -> list[str]:
            """读取一次已完成任务的重连事件。"""
            return [
                event
                async for event in _events(
                    "run-test", ForeignActor(7, "collector", "2"), "run-test:step:1:RUNNING"
                )
            ]

        with (
            patch(
                "app.foreign_integration.api.session_factory",
                return_value=lambda: Session(self.engine),
            ),
            patch("app.foreign_integration.api._authorized_run", return_value=projection),
        ):
            events = asyncio.run(collect())
        self.assertEqual(len(events), 2)
        self.assertIn('"sequence": 1', events[0])
        self.assertIn("event: complete", events[1])

    def test_original_reference_number_is_not_agent_citation(self) -> None:
        """论文自带的 [15] 不得冒充本次只有一条证据的 Citation。"""
        answer = ExtractiveModelGateway._compose(
            {
                "query": "教师反馈的影响",
                "evidence": [SimpleNamespace(content="教师反馈改善了写作能力[15]。")],
                "output_language": "ZH",
                "detected_language": "zh",
                "max_points": 1,
            }
        )
        self.assertIn("教师反馈改善了写作能力", answer)
        self.assertIn("[1]", answer)
        self.assertNotIn("[15]", answer)


if __name__ == "__main__":
    unittest.main()
