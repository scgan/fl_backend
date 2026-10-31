"""所选文献权限、会话复用、完整材料、预算和独立 graph 的隔离测试。"""

import asyncio
import hashlib
import json
import unittest
import uuid
from types import SimpleNamespace
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, insert, select, update
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from app.agents.literature_graph import LiteratureGraph, cited_answer, pack_materials
from app.agents.literature_schemas import ReadingAnswer, ReadingInnovation, ReadingOverview
from app.foreign_integration.api import _validate_options, router
from app.foreign_integration.auth import ForeignActor, current_actor
from app.foreign_integration.database import get_session
from app.foreign_integration.literature import (
    LiteratureMaterialTool,
    selection_rows,
    selection_version,
)
from app.foreign_integration.schema import (
    foreign_corpus,
    foreign_corpus_index_job,
    foreign_user,
    metadata,
    research_agent,
    research_citation,
    research_conversation,
    research_evidence,
    research_model_config,
    research_prompt,
    research_run,
    research_run_scope,
    research_step,
)
from app.foreign_integration.service import RunExecutor, create_run
from app.retrieval.schemas import CitationRead, EvidenceRead


def evidence(index, corpus_id):
    """生成稳定文献和切块 UUID 的证据，页码由输入序号确定以检验定位映射。"""
    return EvidenceRead(
        citation_index=index,
        document_id=uuid.UUID(int=corpus_id),
        chunk_id=uuid.UUID(int=index),
        document_title=f"文献{corpus_id}",
        authors=[],
        publication_year=2024,
        journal=None,
        language_code="zh",
        page_start=index,
        page_end=index,
        section=None,
        content=f"文献{corpus_id}研究指出教师工作发生变化。",
        relevance_score=1,
        retrieval_score=1,
        vector_rank=None,
        lexical_rank=None,
    )


def citation(item):
    """将同一证据转换为引用，保留编号、切块、原页和正文摘录。"""
    return CitationRead(
        citation_index=item.citation_index,
        document_id=item.document_id,
        chunk_id=item.chunk_id,
        document_title=item.document_title,
        page_start=item.page_start,
        page_end=item.page_end,
        section=None,
        quote=item.content[:360],
        file_url="/api/file",
    )


class FakeGateway:
    """按输出 Schema 返回确定性阅读结果，不调用模型。

    Attributes:
        calls: 记录 Schema 名及实际指令，用于检查路由和输入范围。
        wrong_citation: 保留的旧测试配置标记。
        unsupported: 控制语义支持检查替身的返回值。
    """
    def __init__(self, wrong_citation=False, unsupported=False):
        """初始化调用记录及兼容测试开关，使不同用例不会共享模型调用状态。"""
        self.calls = []
        self.wrong_citation = wrong_citation
        self.unsupported = unsupported

    async def structured(self, instruction, schema, **kwargs):
        """按阅读、概览、问题规划或回答 Schema 生成固定结果，未知 Schema 直接使测试失败。"""
        self.calls.append((schema.__name__, instruction))
        if schema.__name__ == "DocumentReading":
            items, _ = json.JSONDecoder().raw_decode(instruction.split("不可信原文：", 1)[1])
            return schema(
                document_type="理论文献",
                summary=[
                    {
                        "text": items[0]["text"],
                    }
                ],
                sections=[
                    {"key": key, "claims": [], "missing_reason": "当前材料未提供"}
                    for key in (
                        "background",
                        "ideas",
                        "methods",
                        "findings",
                        "contributions",
                        "limitations",
                    )
                ],
                recommended_questions=["教师工作如何变化？", "研究方法是什么？"],
            )
        if schema.__name__ == "ReadingOverview":
            documents = json.loads(instruction.split("分析：", 1)[1])
            return schema(
                overview=[item["analysis"]["summary"][0] for item in documents],
                common_themes=[],
                differences=[],
                potential_innovations=[
                    {"title": "教师协作与课后服务支持机制",
                     "text": "把教师协作与课后服务体验结合起来，使用访谈和工作日志分析支持条件。"},
                    {"title": "排班制度与教师工作体验",
                     "text": "比较不同学校的排班方式，用持续跟踪考察教师时间安排与工作体验。"},
                    {"title": "资源共享与备课改进路径",
                     "text": "结合教师资源共享记录和备课评价，探索协作过程中的具体改进机制。"},
                ],
                recommended_questions=["两篇有哪些差异？", "第二篇采用什么方法？"],
            )
        if schema.__name__ == "QuestionPlan":
            return schema(standalone_question="文献12的研究结论是什么？", corpus_ids=[12])
        if schema.__name__ == "ReadingAnswer":
            items = json.loads(instruction.split("不可信原文：", 1)[1])
            return schema(
                answer=[
                    {
                        "text": items[0]["text"],
                        "citation_ids": [items[0]["citation_id"]],
                    }
                ],
                insufficient_reason="",
                recommended_questions=[],
            )
        if schema.__name__ == "SupportReview":
            return schema(unsupported_claim_indices=[0] if self.unsupported else [])
        raise AssertionError(schema.__name__)


class FakeMaterials:
    """提供两篇可控文献及证据，使阅读测试不依赖数据库外部索引。

    Attributes:
        items: 当前可用证据，unavailable 场景为空。
        documents: 固定选文范围及其就绪状态。
        sources: 加载时建立的切块到真实来源映射，供引用验证。
    """
    def __init__(self, unavailable=False):
        """构造就绪或材料不足场景，保持相同的选中文献范围。"""
        self.items = [] if unavailable else [evidence(1, 11), evidence(2, 12)]
        self.documents = [
            {
                "corpus_id": key,
                "title": f"文献{key}",
                "author": "",
                "status": "UNAVAILABLE" if unavailable else "READY",
                "reason": "解析未就绪" if unavailable else "",
                "chunk_count": 1,
            }
            for key in (11, 12)
        ]

    async def load_materials(self):
        """建立切块来源映射，返回文献、证据及一一对应的引用。"""
        self.sources = {
            item.chunk_id: SimpleNamespace(
                corpus_id=item.document_id.int,
                point_id=str(item.chunk_id),
                chunk_key=str(item.chunk_id),
                file_path="paper.pdf",
                content=item.content,
                page_start=item.page_start,
                page_end=item.page_end,
                score=1.0,
            )
            for item in self.items
        }
        return self.documents, self.items, [citation(item) for item in self.items]

    async def recommend(self, readings):
        """默认关闭推荐结果，隔离主阅读及引用断言。"""
        return {"recommended_documents": [], "recommendation_status": "SUCCESS"}


class LiteratureTests(unittest.TestCase):
    """在隔离 SQLite 和确定性模型替身下验证阅读范围、会话、并发及引用完整性。"""
    def setUp(self):
        """创建独立内存数据库、两位用户、不同审核状态文献及阅读模型配置。"""
        self.engine = create_engine(
            "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
        )
        metadata.create_all(self.engine)
        self.session = Session(self.engine)
        self.actor = ForeignActor(7, "reader", 2)
        self.session.execute(
            insert(foreign_user),
            [
                {
                    "id": key,
                    "username": f"reader{key}",
                    "role": '["2"]',
                    "status": 1,
                    "is_deleted": False,
                }
                for key in (7, 8)
            ],
        )
        self.session.execute(
            insert(foreign_corpus),
            [
                {
                    "id": key,
                    "title_cn": f"文献{key}",
                    "attachment": "[]",
                    "status": 2 if key != 13 else 1,
                    "is_deleted": False,
                    "is_vectorized": True,
                }
                for key in (11, 12, 13)
            ],
        )
        self.session.execute(
            insert(research_model_config).values(
                id="m" * 32, provider="OPENAI_COMPATIBLE", model_name="test", enabled=True
            )
        )
        self.session.execute(
            insert(research_agent).values(
                id="a" * 32,
                code="LITERATURE",
                name="文献阅读",
                description="阅读",
                model_config_id="m" * 32,
                enabled=True,
            )
        )
        self.session.execute(
            insert(research_prompt).values(
                id="p" * 32, agent_id="a" * 32, version=1, system_prompt="证据优先", status="ACTIVE"
            )
        )
        self.session.commit()

    def tearDown(self):
        """关闭本例会话及内存数据库，避免状态泄漏到下个用例。"""
        self.session.close()
        self.engine.dispose()

    def create(self, **overrides):
        """按固定两篇选文构造阅读运行，允许覆盖字段测试权限与幂等边界。"""
        values = dict(
            actor=self.actor,
            code="LITERATURE",
            request_id=uuid.uuid4().hex,
            knowledge_base_id=None,
            query="分析所选文献",
            conversation_id=None,
            mode="FAST",
            options={"corpus_ids": [11, 12], "task": "ANALYZE", "template_version": 1},
        )
        values.update(overrides)
        return create_run(self.session, **values)

    def test_preview_hint_does_not_change_selected_scope(self):
        """验证预览焦点只是指代提示，不改变已冻结的选文范围。"""
        options = _validate_options(
            "LITERATURE", {"corpus_ids": [11, 12], "task": "ASK", "focus_corpus_id": 12}
        )
        self.assertEqual(options["corpus_ids"], [11, 12])
        self.assertEqual(options["focus_corpus_id"], 12)

    def test_long_recent_answer_remains_in_bounded_history(self):
        """验证历史预算裁剪仍保留最近一轮长回答的必要上下文。"""
        accepted = self.create()
        self.session.execute(
            update(research_run)
            .where(research_run.c.id == accepted["run_id"])
            .values(
                status="SUCCESS",
                final_answer="分析完成",
                result_payload={"payload": {"task": "ANALYZE", "overview": {}, "documents": []}},
            )
        )
        self.session.commit()
        latest = self.create(
            conversation_id=accepted["conversation_id"],
            query="展开最近的回答",
            options={"corpus_ids": [11, 12], "task": "ASK", "template_version": 1},
        )
        self.session.execute(
            update(research_run)
            .where(research_run.c.id == latest["run_id"])
            .values(
                status="SUCCESS",
                final_answer="长" * 20_000,
                result_payload={
                    "payload": {
                        "task": "ASK",
                        "answer": {"answer": [{"text": "段" * 1000} for _ in range(16)]},
                    }
                },
            )
        )
        self.session.commit()
        with patch(
            "app.foreign_integration.service.session_factory",
            return_value=lambda: Session(self.engine),
        ):
            history, _ = RunExecutor("not-a-run")._reading_history(
                {"id": "not-a-run", "conversation_id": accepted["conversation_id"], "user_id": 7}
            )
        self.assertEqual(history[-2]["content"], "展开最近的回答")
        self.assertLessEqual(len(json.dumps(history, ensure_ascii=False).encode()), 40_000)

    def test_selection_without_knowledge_base(self):
        """验证全局选文阅读不要求先加入私有知识库。"""
        accepted = self.create()
        run = self.session.execute(select(research_run)).mappings().one()
        conversation = self.session.execute(
            select(research_conversation).where(
                research_conversation.c.id == accepted["conversation_id"]
            )
        ).mappings().one()
        self.assertIsNone(conversation["knowledge_base_id"])
        self.assertEqual(run["conversation_id"], accepted["conversation_id"])
        self.assertEqual(run["output_language"], "ZH")
        self.assertEqual(
            set(self.session.scalars(select(research_run_scope.c.corpus_id))), set()
        )
        self.assertEqual(run["request_payload"]["corpus_ids"], [11, 12])

    def test_initial_analysis_deduplicates_refresh_and_tabs(self):
        """验证初次分析在刷新或多个标签页重复请求时保持幂等。"""
        first, second = self.create(), self.create()
        self.assertEqual(first["run_id"], second["run_id"])
        self.assertFalse(second["created"])

    def test_chinese_analysis_does_not_reuse_previous_foreign_language_result(self):
        """旧版外语分析仍可保留历史，新中文任务使用新身份且同规则重复提交仍幂等。"""
        first = self.create()
        old = self.session.execute(select(research_run).where(
            research_run.c.id == first["run_id"],
        )).mappings().one()
        signature = json.dumps([
            self.actor.id, old["request_payload"], old["query"],
            selection_version(selection_rows(self.session, self.actor.id, [11, 12])),
            old["prompt_id"], old["model_config_id"], "literature-with-titled-innovations-v6",
        ], ensure_ascii=False, sort_keys=True)
        self.session.execute(update(research_run).where(
            research_run.c.id == first["run_id"],
        ).values(
            request_id="literature:" + hashlib.sha256(signature.encode()).hexdigest(),
            status="SUCCESS", final_answer="Historical English analysis.", output_language="EN",
        ))
        self.session.commit()
        current = self.create()
        self.assertTrue(current["created"])
        self.assertNotEqual(current["run_id"], first["run_id"])
        self.assertEqual(self.create()["run_id"], current["run_id"])
        self.assertEqual(self.session.scalar(select(research_run.c.final_answer).where(
            research_run.c.id == first["run_id"],
        )), "Historical English analysis.")

    def test_unreviewed_document_rejected(self):
        """验证未审核文献不能进入固定阅读范围。"""
        with self.assertRaises(PermissionError):
            self.create(options={"corpus_ids": [13], "task": "ANALYZE", "template_version": 1})

    def test_personal_and_public_documents_share_one_ordered_reading_scope(self):
        """本人的个人材料与公共文献可以混合分析，运行保留明确的选择顺序。"""
        self.session.execute(update(foreign_corpus).where(foreign_corpus.c.id == 12)
                             .values(status=4, owner_id=7))
        self.session.commit()
        self.assertEqual(list(selection_rows(self.session, 7, [12, 11])), [12, 11])
        accepted = self.create(options={
            'corpus_ids': [12, 11], 'task': 'ANALYZE', 'template_version': 1,
        })
        run = self.session.execute(select(research_run).where(
            research_run.c.id == accepted['run_id'],
        )).mappings().one()
        self.assertEqual(run['request_payload']['corpus_ids'], [12, 11])

    def test_personal_selection_rejects_other_owner_ownerless_and_deleted_files(self):
        """知道个人文件 ID 也不能越权分析；缺少所有者或已删除时同样拒绝。"""
        for values in ({'status': 4, 'owner_id': 8, 'is_deleted': False},
                       {'status': 4, 'owner_id': None, 'is_deleted': False},
                       {'status': 4, 'owner_id': 7, 'is_deleted': True},
                       {'status': 2, 'owner_id': 8, 'is_deleted': False}):
            with self.subTest(values=values):
                self.session.execute(
                    update(foreign_corpus).where(foreign_corpus.c.id == 12).values(**values)
                )
                self.session.commit()
                with self.assertRaises(PermissionError):
                    selection_rows(self.session, 7, [11, 12])

    def test_personal_owner_is_part_of_frozen_material_version(self):
        """个人文件的所有权和状态改变会使旧材料指纹失效，撤权后不能继续读取。"""
        before = selection_version(selection_rows(self.session, 7, [11, 12]))
        self.session.execute(update(foreign_corpus).where(foreign_corpus.c.id == 12)
                             .values(status=4, owner_id=7))
        self.session.commit()
        after = selection_version(selection_rows(self.session, 7, [11, 12]))
        self.assertNotEqual(before, after)
        self.session.execute(
            update(foreign_corpus).where(foreign_corpus.c.id == 12).values(owner_id=8)
        )
        self.session.commit()
        with self.assertRaises(PermissionError):
            selection_rows(self.session, 7, [11, 12])

    def test_existing_public_reading_versions_remain_compatible(self):
        """公共材料沿用旧版指纹，部署个人文件支持不使既有公共阅读任务失效。"""
        rows = selection_rows(self.session, 7, [11, 12])
        old_payload = [(key, row['attachment'], row['index_build_id'],
                        row['index_attachment_digest'], row['title_cn'], row['title_en'])
                       for key, row in rows.items()]
        digest = hashlib.sha256(json.dumps(old_payload, ensure_ascii=False).encode()).digest()
        self.assertEqual(
            selection_version(rows), int.from_bytes(digest[:8], 'big') & ((1 << 63) - 1)
        )

    def test_another_user_cannot_continue_conversation(self):
        """验证其他用户不能复用已知会话 ID 继续阅读。"""
        first = self.create()
        with self.assertRaises(PermissionError):
            self.create(
                actor=ForeignActor(8, "reader8", 2), conversation_id=first["conversation_id"]
            )

    def test_scope_cannot_change_in_conversation(self):
        """验证同一会话不能通过后续请求替换已选文献范围。"""
        first = self.create()
        with self.assertRaises(ValueError):
            self.create(
                conversation_id=first["conversation_id"],
                options={"corpus_ids": [12], "task": "ANALYZE", "template_version": 1},
            )

    def test_ask_requires_completed_analysis(self):
        """验证初次分析未完成时不能直接继续追问。"""
        with self.assertRaises(ValueError):
            self.create(options={"corpus_ids": [11, 12], "task": "ASK", "template_version": 1})

    def test_material_version_detects_attachment_change(self):
        """验证阅读版本指纹能够发现分析后的附件变更。"""
        before = selection_version(selection_rows(self.session, 7, [11, 12]))
        self.session.execute(
            update(foreign_corpus)
            .where(foreign_corpus.c.id == 11)
            .values(attachment='[{"file_path":"changed.pdf"}]')
        )
        after = selection_version(selection_rows(self.session, 7, [11, 12]))
        self.assertNotEqual(before, after)

    def test_reading_history_excluded_from_original_conversation_list(self):
        """验证阅读历史使用独立入口，不混入原有研究会话列表。"""
        self.create()
        app = FastAPI()
        app.include_router(router)

        @app.middleware("http")
        async def trace(request, call_next):
            """向测试请求写入固定追踪 ID，满足统一响应契约。"""
            request.state.request_id = "test-reading"
            return await call_next(request)

        app.dependency_overrides[current_actor] = lambda: self.actor
        app.dependency_overrides[get_session] = lambda: self.session
        client = TestClient(app)
        self.assertEqual(client.get("/conversations").json()["data"]["items"], [])
        self.assertEqual(client.get("/literature/conversations").json()["data"]["total"], 1)

    def test_material_pagination_keeps_all_chunks(self):
        """验证原文加载跨页读取完整切块，不只保留首次返回的片段。"""
        path = "paper.pdf"
        attachments = json.dumps([{"file_path": path}])
        digest = hashlib.sha256(attachments.encode()).hexdigest()
        self.session.execute(
            update(foreign_corpus).where(foreign_corpus.c.id == 11).values(attachment=attachments)
        )
        self.session.execute(
            insert(foreign_corpus_index_job).values(
                corpus_id=11,
                status="SUCCESS",
                build_id="build",
                attachment_digest=digest,
                chunk_count=2,
                vector_status="SUCCESS",
                lexical_status="SUCCESS",
                is_deleted=False,
            )
        )
        self.session.commit()
        rows = selection_rows(self.session, 7, [11])
        settings = SimpleNamespace(
            qdrant_url="http://test", qdrant_collection="chunks", public_api_base="/api"
        )
        tool = LiteratureMaterialTool(
            actor_id=7,
            knowledge_base_id=0,
            fixed_scope=[11],
            fixed_version=selection_version(rows),
            settings=settings,
        )
        points = [
            SimpleNamespace(
                id=index,
                payload={
                    "text": f"原文{index}",
                    "content_hash": hashlib.sha256(f"原文{index}".encode()).hexdigest(),
                    "file_path": path,
                    "attachment_digest": digest,
                    "build_id": "build",
                    "chunk_index": index,
                    "page_start": index,
                    "page_end": index,
                },
            )
            for index in (1, 2)
        ]
        with (
            patch(
                "app.foreign_integration.literature.session_factory",
                return_value=lambda: Session(self.engine),
            ),
            patch("app.foreign_integration.literature.QdrantClient", autospec=True) as client,
        ):
            client.return_value.scroll.side_effect = [
                ([points[0]], 2),
                ([points[1]], None),
            ]
            documents, chunks, refs = tool._load_materials_sync()
            client.return_value.close.assert_called_once()
            client.return_value.reset_mock()
            client.return_value.scroll.side_effect = RuntimeError("索引读取中断")
            with self.assertRaisesRegex(RuntimeError, "索引读取中断"):
                tool._load_materials_sync()
            client.return_value.close.assert_called_once()
        self.assertEqual(documents[0]["status"], "READY")
        self.assertEqual(len(chunks), 2)
        self.assertEqual(len(refs), 2)

    def run_graph(self, gateway=None, materials=None, **state):
        """在独立事件循环执行阅读图，可注入材料、网关和图状态。"""
        async def step(name, state, operation):
            """直接等待本阶段操作，隔离步骤持久化对图行为测试的影响。"""
            return await operation()

        return asyncio.run(
            LiteratureGraph(
                materials or FakeMaterials(), gateway or FakeGateway(), "证据优先", step
            ).run({"task": "ANALYZE", **state})
        )

    def test_multi_document_fixed_template(self):
        """验证多篇阅读保留每篇固定六分区及综合概览结构。"""
        result = self.run_graph()
        self.assertEqual(len(result["literature_result"]["documents"]), 2)
        self.assertEqual(result["validated_citations"], [])
        self.assertEqual(
            result["literature_result"]["documents"][0]["analysis"]["sections"][2]["key"], "methods"
        )

    def test_analysis_concurrency_is_bounded_and_preserves_selection_order(self):
        """验证并发峰值受限、不可用文献跳过分析，最终输出仍按选文顺序。"""
        async def check():
            """用事件屏障控制完成先后，区分并发调度顺序与最终展示顺序。"""
            active, peak = 0, 0
            entered = asyncio.Event()
            release = asyncio.Event()
            completed = []

            class Gateway(FakeGateway):
                """延迟首批文献分析并记录活动数，综合阶段必须等全部有效文献完成。"""
                async def structured(inner, instruction, schema, **kwargs):
                    """按文献 ID 阻塞或放行分析，在 finally 回收活动计数以验证峰值。"""
                    nonlocal active, peak
                    if schema.__name__ != "DocumentReading":
                        self.assertEqual(active, 0)
                        self.assertEqual(len(completed), 5)
                        return await super().structured(instruction, schema, **kwargs)
                    items, _ = json.JSONDecoder().raw_decode(
                        instruction.split("不可信原文：", 1)[1]
                    )
                    key = items[0]["corpus_id"]
                    active += 1
                    peak = max(peak, active)
                    try:
                        if key in (11, 12, 13):
                            await release.wait()
                        else:
                            entered.set()
                        result = await super().structured(instruction, schema, **kwargs)
                        completed.append(key)
                        return result
                    finally:
                        active -= 1

            materials = FakeMaterials()
            materials.documents = [
                {**materials.documents[0], "corpus_id": key, "title": f"文献{key}"}
                for key in range(11, 17)
            ]
            materials.documents[3]["status"] = "UNAVAILABLE"
            materials.items = [evidence(key, key) for key in range(11, 17)]

            async def step(name, state, operation):
                """直接执行节点，不增加额外串行等待影响并发测试。"""
                return await operation()

            graph = LiteratureGraph(materials, Gateway(), "测试", step)
            task = asyncio.create_task(graph.run({"task": "ANALYZE"}))
            try:
                await asyncio.wait_for(entered.wait(), 2)
                # 前三篇还在等待时，后面的篇目已完成，证明没有串行执行。
                self.assertIn(15, completed)
                self.assertNotIn(11, completed)
                release.set()
                result = await asyncio.wait_for(task, 2)
            finally:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
            self.assertEqual(peak, 4)
            documents = result["literature_result"]["documents"]
            self.assertEqual([item["corpus_id"] for item in documents], list(range(11, 17)))
            self.assertIsNone(documents[3]["analysis"])

        asyncio.run(check())

    def test_failed_document_cancels_peers_and_persists_terminal_steps(self):
        """验证单篇失败或父任务取消时收尾其他任务，并持久化失败或取消终态。"""
        async def check(cancel_parent=False):
            """同步启动多个分析后注入失败或父取消，确认无遗留活跃协程。"""
            started = asyncio.Event()
            active = 0
            cancelled = []

            class Gateway(FakeGateway):
                """将分析挂起并按场景触发失败，记录同伴收到取消的情况。"""
                async def structured(inner, instruction, schema, **kwargs):
                    """在事件屏障后注入单篇错误，或等待父取消，并始终回收活动计数。"""
                    nonlocal active
                    self.assertEqual(schema.__name__, "DocumentReading")
                    active += 1
                    if active == 2:
                        started.set()
                    try:
                        await started.wait()
                        if "文献11研究" in instruction and not cancel_parent:
                            raise RuntimeError("测试文献失败")
                        await asyncio.Event().wait()
                    except asyncio.CancelledError:
                        cancelled.append(instruction)
                        raise
                    finally:
                        active -= 1

            executor = RunExecutor("cancel-test")
            graph = LiteratureGraph(FakeMaterials(), Gateway(), "测试", executor._run_step)
            task = asyncio.create_task(
                graph._analyze(
                    {
                        "documents": FakeMaterials().documents,
                        "materials": FakeMaterials().items,
                    }
                )
            )
            try:
                await asyncio.wait_for(started.wait(), 2)
                if cancel_parent:
                    task.cancel()
                    with self.assertRaises(asyncio.CancelledError):
                        await task
                else:
                    with self.assertRaisesRegex(RuntimeError, "测试文献失败"):
                        await task
            finally:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
            self.assertEqual(active, 0)
            self.assertEqual(len(cancelled), 2 if cancel_parent else 1)

        with patch(
            "app.foreign_integration.service.session_factory",
            return_value=lambda: Session(self.engine),
        ):
            asyncio.run(check())
            rows = self.session.execute(select(research_step)).mappings().all()
            self.assertEqual({row["status"] for row in rows}, {"FAILED", "CANCELLED"})
            self.assertTrue(all(row["finished_at"] for row in rows))
            self.assertEqual(len({row["sequence"] for row in rows}), 2)
            asyncio.run(check(cancel_parent=True))
            self.session.expire_all()
            rows = self.session.execute(select(research_step)).mappings().all()
            self.assertEqual(sum(row["status"] == "CANCELLED" for row in rows), 3)
            self.assertFalse(any(row["status"] == "RUNNING" for row in rows))

    def test_expanded_overview_keeps_all_paragraphs_for_single_and_multiple_documents(self):
        """验证单篇与多篇都完整保留概览段落和创新方向正文，不被展示层截短。"""
        paragraphs = [
            {"text": f"第{index}段综合介绍：" + "不同研究角度有助于理解教育实践。" * 8}
            for index in range(1, 7)
        ]
        innovations = [
            {"title": f"研究方向{index}的具体切入点",
             "text": "结合文献关注的教师协作问题，提出可进一步验证的研究设想。"
                     + "通过访谈、工作日志和持续观察分析支持条件及其适用范围。" * 8}
            for index in range(1, 4)
        ]

        class ExpandedGateway(FakeGateway):
            """提供包含长概览和长创新正文的模型替身，检验发布时的完整性。"""
            async def structured(self, instruction, schema, **kwargs):
                """只替换综合概览结果，其他阶段沿用固定网关行为。"""
                if schema.__name__ == "ReadingOverview":
                    return schema(
                        overview=paragraphs,
                        common_themes=[],
                        differences=[],
                        potential_innovations=innovations,
                        recommended_questions=["研究的主要启发是什么？", "有哪些适用限制？"],
                    )
                return await super().structured(instruction, schema, **kwargs)

        for count in (1, 2):
            with self.subTest(document_count=count):
                materials = FakeMaterials()
                materials.documents = materials.documents[:count]
                materials.items = materials.items[:count]
                result = self.run_graph(gateway=ExpandedGateway(), materials=materials)
                self.assertEqual(result["literature_result"]["overview"]["overview"], paragraphs)
                self.assertEqual(
                    result["literature_result"]["overview"]["potential_innovations"], innovations,
                )
                for index, innovation in enumerate(innovations, 1):
                    self.assertGreater(len(innovation["text"]), 200)
                    self.assertIn(f"### {index}. {innovation['title']}", result["final_answer"])
                    self.assertIn(innovation["text"], result["final_answer"])
                overview_text = result["final_answer"].split("## 文献", 1)[0]
                for paragraph in paragraphs:
                    self.assertIn(paragraph["text"], overview_text)

    def test_executor_publishes_followup_citations_and_restores_history(self):
        """验证公共和个人材料混合分析、追问引用发布与历史恢复，个人原文走授权链接。"""
        self.session.execute(
            update(foreign_corpus)
            .where(foreign_corpus.c.id.in_([11, 12]))
            .values(attachment='[{"file_path":"paper.pdf"}]')
        )
        self.session.execute(
            update(foreign_corpus).where(foreign_corpus.c.id == 12).values(status=4, owner_id=7)
        )
        self.session.commit()
        accepted = self.create()
        settings = SimpleNamespace(qdrant_collection="chunks", literature_analysis_concurrency=3)
        gateway = FakeGateway()
        materials = FakeMaterials()
        materials.items[1] = materials.items[1].model_copy(
            update={"content": "文献12研究指出教师工作发生变化。" * 30}
        )
        with (
            patch(
                "app.foreign_integration.service.session_factory",
                return_value=lambda: Session(self.engine),
            ),
            patch("app.foreign_integration.service.get_foreign_settings", return_value=settings),
            patch(
                "app.foreign_integration.service.LiteratureMaterialTool",
                return_value=materials,
            ),
            patch("app.foreign_integration.service.get_model_gateway", return_value=gateway),
        ):
            asyncio.run(RunExecutor(accepted["run_id"]).execute())
            self.session.expire_all()
            run = (
                self.session.execute(
                    select(research_run).where(research_run.c.id == accepted["run_id"])
                )
                .mappings()
                .one()
            )
            self.assertEqual(run["status"], "SUCCESS", run["error_message"])
            self.assertNotIn("[1]", run["final_answer"])
            innovations = run["result_payload"]["payload"]["overview"]["potential_innovations"]
            self.assertEqual(len(innovations), 3)
            self.assertIn("## 潜在创新点", run["final_answer"])
            self.assertIn(f"### 1. {innovations[0]['title']}", run["final_answer"])
            self.assertIn(innovations[0]["text"], run["final_answer"])
            followup = self.create(
                conversation_id=accepted["conversation_id"],
                query="第二篇呢？",
                options={"corpus_ids": [11, 12], "task": "ASK", "template_version": 1},
            )
            asyncio.run(RunExecutor(followup["run_id"]).execute())
            self.session.expire_all()
            reply = (
                self.session.execute(
                    select(research_run).where(research_run.c.id == followup["run_id"])
                )
                .mappings()
                .one()
            )
            self.assertEqual(reply["status"], "SUCCESS", reply["error_message"])
            self.assertEqual(reply["result_payload"]["payload"]["target_corpus_ids"], [12])
            self.assertTrue(reply["final_answer"].endswith("[1]"))
            saved = self.session.execute(select(research_citation)).mappings().all()
            self.assertEqual(len(saved), 1)
            self.assertEqual(saved[0]["citation_index"], 1)
            rows = (
                self.session.execute(
                    select(research_evidence).where(
                        research_evidence.c.run_id == followup["run_id"]
                    )
                )
                .mappings()
                .all()
            )
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["corpus_id"], 12)
            self.assertEqual(rows[0]["page_start"], 2)
            app = FastAPI()
            app.include_router(router)
            app.dependency_overrides[current_actor] = lambda: self.actor
            app.dependency_overrides[get_session] = lambda: self.session

            @app.middleware("http")
            async def trace(request, call_next):
                """为引用发布接口测试注入固定追踪 ID。"""
                request.state.request_id = "citation-test"
                return await call_next(request)

            restored = TestClient(app).get(f"/conversations/{accepted['conversation_id']}").json()
            # The public conversation API restores each answer's own citation mapping.
            self.assertEqual(restored["code"], "OK")
            replies = [item for item in restored["data"]["messages"] if item["role"] == "ASSISTANT"]
            self.assertEqual(replies[-1]["citations"][0]["file_path"], "paper.pdf")
            self.assertEqual(replies[-1]["citations"][0]["page_start"], 2)
            self.assertEqual(replies[-1]["citations"][0]["content"], rows[0]["content_snapshot"])
            self.assertGreater(len(replies[-1]["citations"][0]["content"]), 360)
            with patch('app.foreign_integration.api.get_foreign_settings', return_value=
                       SimpleNamespace(public_api_base='/api')):
                evidence_response = TestClient(app).get(
                    f"/agent-runs/{followup['run_id']}/evidence"
                )
            self.assertEqual(evidence_response.status_code, 200)
            personal_source = evidence_response.json()['data'][0]
            self.assertFalse(personal_source['source_unavailable'])
            self.assertIn('/literature/documents/12/file?', personal_source['file_url'])
            self.assertEqual(len(replies[-1]["citations"][0]["quote_text"]), 360)
            self.assertGreater(followup["run_id"], accepted["run_id"])
            answer_instructions = [text for name, text in gateway.calls if name == "ReadingAnswer"]
            self.assertIn(innovations[0]["text"], answer_instructions[-1])

    def test_old_overview_remains_readable_without_innovation_field(self):
        """验证没有创新方向字段的旧概览仍能读取。"""
        old_result = {
            "overview": [{"text": "已有整体概览"}],
            "common_themes": [],
            "differences": [],
            "recommended_questions": ["有哪些共同主题？", "方法有哪些差异？"],
        }
        self.assertEqual(ReadingOverview.model_validate(old_result).potential_innovations, [])

    def test_synthesis_keeps_expanded_paragraphs_and_ignores_legacy_references(self):
        """验证长正文和分段完整保留，旧文献列表字段不再进入发布结果及 Markdown。"""
        expanded_text = "从工作安排和协作过程理解教师体验。" * 65 + "\n\n第二段讨论适用范围。"

        class ThemedGateway(FakeGateway):
            """提供充分展开的正文及旧版文献列表，验证迁移期间的兼容行为。"""

            async def structured(self, instruction, schema, **kwargs):
                """替换综合结果，单篇分析仍复用已有替身。"""
                result = await super().structured(instruction, schema, **kwargs)
                if schema.__name__ == "ReadingOverview":
                    data = result.model_dump()
                    data["overview"] = [{
                        "title": "教师协作的支持机制",
                        "text": expanded_text,
                        "references": [
                            {"corpus_id": 11, "text": "文献11讨论协作条件。"},
                            {"corpus_id": 11, "text": "重复关联不应发布。"},
                            {"corpus_id": 999, "text": "越界文献不应发布。"},
                            {"corpus_id": 12, "text": "文献12提供排班视角。"},
                        ],
                    }]
                    data["common_themes"] = [data["overview"][0]]
                    data["differences"] = [data["overview"][0]]
                    data["potential_innovations"][0].update(
                        references=data["overview"][0]["references"],
                        validation_path="比较不同排班制度，跟踪教师工作日志并核对访谈。",
                    )
                    return schema.model_validate(data)
                return result

        result = self.run_graph(gateway=ThemedGateway())
        overview = result["literature_result"]["overview"]
        for key in ("overview", "common_themes", "differences", "potential_innovations"):
            self.assertNotIn("references", overview[key][0])
        self.assertGreater(len(expanded_text), 1000)
        self.assertEqual(overview["overview"][0]["text"], expanded_text)
        self.assertIn(expanded_text, result["final_answer"])
        self.assertIn("### 1. 教师协作的支持机制", result["final_answer"])
        self.assertNotIn("**核心思想**：", result["final_answer"])
        self.assertNotIn("**代表文献**：", result["final_answer"])
        self.assertNotIn("**文献依据**：", result["final_answer"])
        self.assertIn("**验证路径**：比较不同排班制度", result["final_answer"])
        self.assertNotIn("越界文献", result["final_answer"])
        self.assertNotIn("重复关联", result["final_answer"])
        restored = ReadingOverview.model_validate(overview)
        self.assertEqual(restored.overview[0].text, expanded_text)
        self.assertNotIn("ReadingReference", ReadingOverview.model_json_schema().get("$defs", {}))

    def test_answer_does_not_add_citation_markers(self):
        """验证分析展示不擅自追加不属于结构化追问契约的引用编号。"""
        result = self.run_graph()
        self.assertNotIn("[1]", result["final_answer"])
        self.assertNotIn("基于材料的判断：", result["final_answer"])

    def test_innovation_points_preserve_custom_angles_and_markdown_without_duplicate_text(self):
        """验证不同方向使用不同标签和数量，结构化结果与历史正文完整保留分项。"""
        directions = [
            {
                "title": "教师协作的过程机制",
                "text": "不应重复发布的旧正文。",
                "validation_path": "不应重复发布的旧验证路径。",
                "points": [
                    {"label": "观察切口", "text": "既有分析关注协作结果，可以继续观察形成过程。"},
                    {"label": "机制假设", "text": "关注**任务分配**与协作节奏。\n\n比较不同阶段。"},
                    {"label": "检验设计", "text": "结合持续日志和访谈检验变化。"},
                ],
            },
            {
                "title": "协作支持工具的改进路径",
                "points": [
                    {"label": label, "text": f"围绕{label}具体说明可探索的改进。"}
                    for label in ("改进目标", "实现思路", "数据条件", "评价方式", "适用范围")
                ],
            },
        ]

        class PointsGateway(FakeGateway):
            """为综合分析提供不同数量及标签的分项创新方向。"""

            async def structured(self, instruction, schema, **kwargs):
                """保留其他节点行为，只替换创新方向。"""
                result = await super().structured(instruction, schema, **kwargs)
                if schema.__name__ == "ReadingOverview":
                    return schema.model_validate({
                        **result.model_dump(), "potential_innovations": directions,
                    })
                return result

        result = self.run_graph(gateway=PointsGateway())
        innovations = result["literature_result"]["overview"]["potential_innovations"]
        self.assertEqual([len(item["points"]) for item in innovations], [3, 5])
        for index, direction in enumerate(directions):
            self.assertEqual(innovations[index]["points"], direction["points"])
            self.assertNotIn("text", innovations[index])
            self.assertNotIn("validation_path", innovations[index])
            for point in direction["points"]:
                self.assertIn(f"- **{point['label']}**：", result["final_answer"])
                self.assertIn(point["text"].replace("\n", "\n  "), result["final_answer"])
        self.assertNotIn("不应重复发布", result["final_answer"])
        self.assertNotIn("代表文献", result["final_answer"])
        restored = ReadingOverview.model_validate(result["literature_result"]["overview"])
        self.assertEqual(restored.potential_innovations[0].points[1].label, "机制假设")

    def test_innovation_points_reject_incomplete_content_and_accept_legacy_text(self):
        """验证新格式不发布残缺分项，旧正文和验证路径仍可正常恢复。"""
        points = [{"label": f"角度{index}", "text": "具体研究设想。"} for index in range(3)]
        for invalid in (
            {},
            {"points": points[:2]},
            {"points": [points[0], points[0], points[2]]},
            {"points": [{"label": "  ", "text": "正文。"}, *points[1:]]},
            {"points": [{"label": "角度零", "text": "  "}, *points[1:]]},
        ):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                ReadingInnovation(title="具体创新方向", **invalid)
        legacy = ReadingInnovation(
            title="旧创新方向", text="历史正文。", validation_path="历史检验方法。",
        )
        self.assertEqual(legacy.points, [])
        self.assertEqual(legacy.validation_path, "历史检验方法。")

    def test_graph_does_not_call_semantic_reviewer(self):
        """验证文献阅读图不额外执行语义审查节点。"""
        gateway = FakeGateway(unsupported=True)
        self.run_graph(gateway=gateway)
        self.assertNotIn("SupportReview", [name for name, _ in gateway.calls])

    def test_second_document_followup_uses_its_evidence(self):
        """验证指代第二篇的追问只使用对应文献证据。"""
        result = self.run_graph(
            task="ASK",
            query="第二篇呢？",
            history=[{"role": "user", "content": "第一篇的结论是什么？"}],
        )
        self.assertEqual(result["literature_result"]["target_corpus_ids"], [12])
        self.assertIn("文献12", result["final_answer"])

    def test_no_material_produces_explicit_unavailable_result(self):
        """验证无可用原文时明确发布不可用原因，不伪造阅读分析。"""
        result = self.run_graph(materials=FakeMaterials(unavailable=True))
        self.assertEqual(result["evidence_level"], "INSUFFICIENT")
        self.assertTrue(
            all(item["analysis"] is None for item in result["literature_result"]["documents"])
        )

    def test_budget_batches_cover_every_fragment(self):
        """验证预算分批覆盖全部原文片段，不因分批遗漏正文。"""
        items = [evidence(index, 11) for index in range(1, 7)]
        batches = pack_materials(items, budget=400)
        self.assertGreater(len(batches), 1)
        self.assertEqual(
            [item.citation_index for batch in batches for item in batch], list(range(1, 7))
        )

    def test_oversized_single_fragment_is_not_silently_truncated(self):
        """验证单片段超预算时明确失败，不静默截断材料。"""
        with self.assertRaises(ValueError):
            pack_materials([evidence(1, 11)], budget=10)

    def test_citations_compact_multiple_repeated_and_unused_chunks(self):
        """验证重复和未使用切块被整理后，发布引用从一开始连续编号。"""
        items = [evidence(index, 11) for index in (2, 7, 9)]
        answer = ReadingAnswer(
            answer=[
                {"text": "比较结果", "citation_ids": [7, 2, 7]},
                {"text": "进一步解释", "citation_ids": [2]},
                {"text": "建议访谈", "citation_ids": []},
            ],
            insufficient_reason="",
            recommended_questions=[],
        )
        text, used, citations, normalized = cited_answer(
            answer, items, [citation(i) for i in items]
        )
        self.assertEqual(text, "比较结果[1][2]\n\n进一步解释[2]\n\n建议访谈")
        self.assertEqual([i.chunk_id.int for i in used], [7, 2])
        self.assertEqual([i.citation_index for i in citations], [1, 2])
        self.assertEqual(normalized.answer[0].citation_ids, [1, 2])

    def test_citations_reject_unknown_or_unverifiable_locations(self):
        """验证未知引用或无法核验的原文定位不能发布。"""
        item = evidence(1, 11)
        answer = ReadingAnswer(
            answer=[{"text": "原文结论", "citation_ids": [1]}],
            insufficient_reason="",
            recommended_questions=[],
        )
        invalid = [
            [],
            [citation(item).model_copy(update={"quote": "不在原文"})],
            [citation(item).model_copy(update={"page_start": 999})],
            [citation(item).model_copy(update={"document_id": uuid.UUID(int=12)})],
        ]
        for sources in invalid:
            with self.subTest(sources=sources), self.assertRaises(ValueError):
                cited_answer(answer, [item], sources)
        with self.assertRaises(ValueError):
            cited_answer(answer, [], [])

    def test_citation_schema_rejects_handwritten_and_invalid_numbers(self):
        """验证正文手写引用、非正整数或非法类型被 Schema 拒绝。"""
        for part in [
            {"text": "结论[1]", "citation_ids": []},
            {"text": "结论", "citation_ids": [0]},
            {"text": "结论", "citation_ids": [True]},
        ]:
            with self.subTest(part=part), self.assertRaises(ValueError):
                ReadingAnswer(answer=[part], insufficient_reason="", recommended_questions=[])

    def test_missing_material_followup_has_no_citations(self):
        """验证材料不足的追问不携带虚构原文引用。"""
        result = self.run_graph(
            task="ASK", query="第二篇呢？", materials=FakeMaterials(unavailable=True)
        )
        self.assertEqual(result["validated_citations"], [])
        self.assertIn("暂无可阅读", result["final_answer"])

    def test_large_followup_uses_retrieved_sources_and_citations(self):
        """验证大材料追问转为定向召回，并基于实际命中重新生成引用。"""
        class SearchMaterials(FakeMaterials):
            """提供超大原文和定向召回替身，以区分整篇加载与追问命中来源。"""
            searched = False

            async def load_materials(inner):
                """返回超预算原文，触发追问使用检索路径。"""
                documents, _, _ = await super().load_materials()
                huge = evidence(5, 12).model_copy(update={"content": "大" * 130_000})
                return documents, [huge], [citation(huge)]

            async def search(inner, **kwargs):
                """断言只检索第二篇文献，并返回带独特页码的命中供引用核验。"""
                inner.searched = True
                self.assertEqual(kwargs["filters"].document_ids, [uuid.UUID(int=12)])
                item = evidence(91, 12)
                return SimpleNamespace(evidence=[item], citations=[citation(item)])

        materials = SearchMaterials()
        result = self.run_graph(task="ASK", query="第二篇呢？", materials=materials)
        self.assertTrue(materials.searched)
        self.assertEqual(result["evidence"][0].chunk_id.int, 91)
        self.assertEqual(result["validated_citations"][0].citation_index, 1)
        self.assertEqual(result["validated_citations"][0].page_start, 91)
        self.assertTrue(result["final_answer"].endswith("[1]"))

    def test_literature_prompt_migration_retains_old_record_and_is_idempotent(self):
        """验证提示词迁移保留旧版本且重复执行不重复创建活动版本。"""
        from app.agents.catalog import LITERATURE_SYSTEM_PROMPT
        from app.foreign_integration import migrate

        engine = create_engine("sqlite+pysqlite:///:memory:")
        metadata.create_all(engine)
        settings = SimpleNamespace(model_name="", model_base_url="")
        with (
            patch.object(migrate, "get_engine", return_value=engine),
            patch.object(migrate, "get_foreign_settings", return_value=settings),
        ):
            migrate.migrate()
            with engine.begin() as connection:
                connection.execute(
                    insert(research_prompt).values(
                        id=migrate.stable_id("prompt", "LITERATURE:1"),
                        agent_id=migrate.stable_id("agent", "LITERATURE"),
                        version=1,
                        system_prompt="historical",
                        status="ACTIVE",
                        is_deleted=False,
                    )
                )
            migrate.migrate()
            migrate.migrate()
        with engine.connect() as connection:
            rows = connection.execute(
                select(
                    research_prompt.c.version,
                    research_prompt.c.system_prompt,
                    research_prompt.c.status,
                )
                .where(research_prompt.c.agent_id == migrate.stable_id("agent", "LITERATURE"))
                .order_by(research_prompt.c.version)
            ).all()
        self.assertEqual(
            rows, [(1, "historical", "ARCHIVED"), (2, LITERATURE_SYSTEM_PROMPT, "ACTIVE")]
        )
        engine.dispose()
