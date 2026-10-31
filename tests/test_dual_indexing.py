"""用真实 SQLite 事务和模拟外部存储验证双路索引的发布、失败和版本撤销。"""

from __future__ import annotations

import json
import unittest
from datetime import datetime, timedelta
from unittest import mock

from flask import Flask, g
from types import SimpleNamespace
from sqlalchemy import Integer, inspect, text
from sqlalchemy.dialects.mysql import BIT, YEAR

from app.exts import db
from app.indexing.jobs import enqueue_corpus_index
from app.indexing.opensearch_store import OpenSearchStore
from app.indexing.service import build_indexes, delete_indexes
from app.models import Corpus, CorpusCategory, CorpusIndexJob, KnowledgeBase, KnowledgeBaseDocument
from app.api.knowledge_base import bp as knowledge_base_bp
from app.utils import qdrant_utils
from app.utils.ocr_errors import record_ocr_warning
from app.workers import corpus_index_worker


class DualIndexingTests(unittest.TestCase):
    """在隔离数据库内执行真实服务逻辑，不调用模型或远程业务数据库。"""

    def setUp(self):
        """建立文献和唯一任务表，外部引擎用记录调用的替身。"""
        self.app = Flask(__name__)
        self.app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite://"
        db.init_app(self.app)
        self.app.register_blueprint(knowledge_base_bp)

        @self.app.before_request
        def authenticated_owner():
            """离线路由测试使用已认证用户，不跳过资源级权限校验。"""
            g.current_user = SimpleNamespace(id=7)
            g.role = 2
        self.context = self.app.app_context()
        self.context.push()
        self.types = [(column, column.type)
                      for table in (Corpus.__table__, CorpusCategory.__table__)
                      for column in table.columns
                      if isinstance(column.type, (BIT, YEAR))]
        for column, kind in self.types:
            column.type = kind.with_variant(Integer(), "sqlite")
        for table in (Corpus.__table__, CorpusCategory.__table__, CorpusIndexJob.__table__,
                      KnowledgeBase.__table__, KnowledgeBaseDocument.__table__):
            table.create(db.engine)
        self.attachments = [{"file_path": "paper.pdf"}]
        db.session.add(Corpus(id=42, contributor=7, title_cn="学习证据", status=2,
                              attachment=json.dumps(self.attachments),
                              is_deleted=False, is_vectorized=True))
        db.session.commit()
        db.session.add_all([
            KnowledgeBase(id=9, owner_id=7, name='test', is_deleted=False),
            KnowledgeBase(id=10, owner_id=8, name='other', is_deleted=False),
            KnowledgeBaseDocument(knowledge_base_id=9, corpus_id=42, added_by=7, is_deleted=False),
        ])
        db.session.commit()
        self.client = mock.MagicMock()
        self.store = mock.MagicMock(index_name="foreign_chunks_v1")
        self.patches = [
            mock.patch.object(qdrant_utils, "get_qdrant_client", return_value=self.client),
            mock.patch.object(qdrant_utils, "_read_index_blocks", return_value=[
                {"text": "学习证据 learning evidence", "page_idx": 0,
                 "bbox": [0, 0, 1, 1], "block_id": 1},
            ]),
            mock.patch.object(qdrant_utils, "_get_embedding", return_value=[[0.1] * 1024]),
            mock.patch("app.indexing.service.get_lexical_store", return_value=self.store),
            mock.patch("app.indexing.opensearch_store.get_lexical_store", return_value=self.store),
        ]
        for patcher in self.patches:
            patcher.start()

    def test_versioned_queue_is_invisible_to_legacy_workers(self):
        """同一测试库的新请求不会被只领取 PENDING 的旧进程抢走。"""
        with mock.patch.dict('os.environ', {'FOREIGN_INDEX_QUEUE_STATUS':'PENDING_TOKENS'}):
            first = self.app.test_client().post('/knowledge-bases/9/documents/42/reindex')
            self.assertEqual(first.status_code, 202)
            self.assertEqual(first.json['data']['index']['status'], 'PENDING')
            job = db.session.get(CorpusIndexJob, 42)
            self.assertEqual(job.status, 'PENDING_TOKENS')
            self.assertEqual(db.session.query(CorpusIndexJob).filter_by(status='PENDING').count(), 0)
            generation = job.generation
            second = self.app.test_client().post('/knowledge-bases/9/documents/42/reindex')
            self.assertTrue(second.json['data']['reused'])
            self.assertEqual(db.session.get(CorpusIndexJob, 42).generation, generation)
            job.next_attempt_at = datetime.utcnow() - timedelta(seconds=1)
            db.session.commit()
            with mock.patch.object(corpus_index_worker, 'app', self.app):
                claim = corpus_index_worker.claim_next_job()
                self.assertEqual(claim[0], 42)
            db.session.expire_all()
            self.assertEqual(db.session.get(CorpusIndexJob, 42).status, 'RUNNING')

    def tearDown(self):
        """释放隔离事务和测试补丁。"""
        for patcher in reversed(self.patches):
            patcher.stop()
        db.session.remove()
        db.engine.dispose()
        self.context.pop()
        for column, kind in self.types:
            column.type = kind

    def test_filename_punctuation_is_preserved_in_both_index_payloads(self):
        """验证真实文件名中的标点在 Qdrant 与 OpenSearch 载荷中一致保留。"""
        filename = 'Example. Policy affecting higher education? #2; %3F.pdf'
        attachments = [{'file_path': 'uploaded_files/corpus/' + filename}]
        corpus = db.session.get(Corpus, 42)
        corpus.attachment = json.dumps(attachments)
        db.session.commit()
        build_indexes(42, attachments)
        vector_points = self.client.upsert.call_args.kwargs['points']
        lexical_points = self.store.write_points.call_args.args[0]
        for points in (vector_points, lexical_points):
            self.assertTrue(points)
            for point in points:
                self.assertEqual(point.payload['file_name'], filename)
                self.assertEqual(point.payload['file_ext'], 'pdf')
                self.assertEqual(point.payload['file_path'], 'uploaded_files/corpus/' + filename)

    def test_personal_file_publishes_both_indexes_with_private_payload(self):
        """有所有者的个人材料复用双路构建，载荷标记使旧公共检索先排除个人片段。"""
        corpus = db.session.get(Corpus, 42)
        corpus.status, corpus.owner_id = 4, 7
        db.session.commit()
        build_indexes(42, self.attachments)
        self.assertTrue(db.session.get(Corpus, 42).is_vectorized)
        self.assertEqual(db.session.get(CorpusIndexJob, 42).status, 'SUCCESS')
        point = self.client.upsert.call_args.kwargs['points'][0]
        self.assertTrue(point.payload['personal_upload'])
        self.assertTrue(corpus_index_worker._valid_source(corpus, db.session.get(CorpusIndexJob, 42).attachment_digest))

    def test_ownerless_personal_file_cannot_build(self):
        """无主个人文件不能绕过公共审核限制建立索引。"""
        corpus = db.session.get(Corpus, 42)
        corpus.status, corpus.owner_id = 4, None
        db.session.commit()
        with self.assertRaises(RuntimeError):
            build_indexes(42, self.attachments)
        self.client.upsert.assert_not_called()

    def test_success_only_after_both_writes_and_same_point_identity(self):
        """OpenSearch 写入期间任务仍 RUNNING，两个引擎使用完全相同的 point。"""
        def verify_unpublished(points):
            """在第二路写入边界检查状态与数据契约。"""
            job = db.session.get(CorpusIndexJob, 42)
            self.assertEqual(job.status, "RUNNING")
            self.assertFalse(db.session.get(Corpus, 42).is_vectorized)
            self.assertEqual(points[0].id, self.client.upsert.call_args.kwargs["points"][0].id)
            self.assertEqual(points[0].payload["build_id"], job.build_id)
        self.store.write_points.side_effect = verify_unpublished
        result = build_indexes(42, self.attachments)
        job = db.session.get(CorpusIndexJob, 42)
        self.assertEqual((job.status, job.vector_status, job.lexical_status), ("SUCCESS",) * 3)
        self.assertEqual(result["total_chunks"], job.chunk_count)
        self.assertTrue(db.session.get(Corpus, 42).is_vectorized)
        self.assertNotIn("corpus_search_index", inspect(db.engine).get_table_names())

    def test_knowledge_base_rebuild_is_async_and_idempotent(self):
        """按钮只入队，重复请求不会创建新 generation 或执行昂贵索引。"""
        client = self.app.test_client()
        endpoint = '/knowledge-bases/9/documents/42/reindex'
        first = client.post(endpoint)
        self.assertEqual(first.status_code, 202)
        self.assertEqual(first.json['data']['index']['status'], 'PENDING')
        generation = db.session.get(CorpusIndexJob, 42).generation
        second = client.post(endpoint)
        self.assertTrue(second.json['data']['reused'])
        self.assertEqual(db.session.get(CorpusIndexJob, 42).generation, generation)
        self.client.upsert.assert_not_called()
        self.assertFalse(db.session.get(Corpus, 42).is_vectorized)

    def test_retry_wait_is_distinguished_from_new_pending_job(self):
        """待重试任务仍不能重复提交，但必须与首次排队区分显示。"""
        client = self.app.test_client()
        first = client.post('/knowledge-bases/9/documents/42/reindex')
        self.assertFalse(first.json['data']['index']['retry_pending'])
        job = db.session.get(CorpusIndexJob, 42)
        job.attempts = 2
        job.error_message = 'text-embedding-v4 API failed'
        db.session.commit()
        state = client.get('/knowledge-bases/9/documents').json['data'][0]['index']
        self.assertEqual(state['status'], 'PENDING')
        self.assertTrue(state['retry_pending'])
        self.assertFalse(state['can_rebuild'])
        self.assertIn('2', state['description'])

    def test_rebuild_rejects_other_base_removed_document_and_empty_source(self):
        """服务端阻止他人知识库、无附件和已移除文献的重建请求。"""
        client = self.app.test_client()
        self.assertEqual(client.post('/knowledge-bases/10/documents/42/reindex').status_code, 404)
        corpus = db.session.get(Corpus, 42)
        corpus.attachment = '[]'
        db.session.commit()
        self.assertEqual(client.post('/knowledge-bases/9/documents/42/reindex').status_code, 400)
        db.session.get(KnowledgeBaseDocument, (9, 42)).is_deleted = True
        db.session.commit()
        self.assertEqual(client.post('/knowledge-bases/9/documents/42/reindex').status_code, 404)
        self.assertIsNone(db.session.get(CorpusIndexJob, 42))

    def test_document_list_requires_dual_success_and_current_version(self):
        """表格不把旧向量标志或词法未完成误显示为已就绪。"""
        client = self.app.test_client()
        endpoint = '/knowledge-bases/9/documents'
        self.assertEqual(client.get(endpoint).json['data'][0]['index']['status'], 'NOT_INDEXED')
        build_indexes(42, self.attachments)
        self.assertEqual(client.get(endpoint).json['data'][0]['index']['status'], 'SUCCESS')
        db.session.get(CorpusIndexJob, 42).lexical_status = 'PENDING'
        db.session.commit()
        self.assertEqual(client.get(endpoint).json['data'][0]['index']['status'], 'STALE')

    def test_lexical_failure_cleans_both_and_never_publishes(self):
        """向量已写但 BM25 失败时撤销整个快照，重试生成新构建令牌。"""
        self.store.write_points.side_effect = RuntimeError("bulk failed")
        with self.assertRaises(RuntimeError):
            build_indexes(42, self.attachments)
        job = db.session.get(CorpusIndexJob, 42)
        self.assertEqual(job.status, "FAILED")
        first_build = job.build_id
        self.assertFalse(db.session.get(Corpus, 42).is_vectorized)
        self.assertGreaterEqual(self.client.delete.call_count, 2)
        self.assertGreaterEqual(self.store.delete_corpus.call_count, 2)
        self.store.write_points.side_effect = None
        build_indexes(42, self.attachments)
        self.assertNotEqual(db.session.get(CorpusIndexJob, 42).build_id, first_build)
        self.assertEqual(db.session.get(CorpusIndexJob, 42).status, "SUCCESS")

    def test_worker_rechecks_generation_before_success(self):
        """Worker 统一发布双路 SUCCESS，不能在索引函数内部提前发布文献。"""
        corpus = db.session.get(Corpus, 42)
        enqueue_corpus_index(db.session, corpus)
        db.session.get(CorpusIndexJob, 42).next_attempt_at = datetime.utcnow() - timedelta(seconds=1)
        db.session.commit()
        with mock.patch.object(corpus_index_worker, "app", self.app):
            claim = corpus_index_worker.claim_next_job()
            corpus_index_worker.process_job(*claim)
        db.session.expire_all()
        self.assertEqual(db.session.get(CorpusIndexJob, 42).status, "SUCCESS")
        self.assertTrue(db.session.get(Corpus, 42).is_vectorized)

    def test_worker_ocr_failure_retries_and_never_publishes_partial_index(self):
        """验证 OCR 失败进入有限重试，双路残留清理且不发布部分就绪。"""
        from app.utils.mineru_utils import MinerUError

        enqueue_corpus_index(db.session, db.session.get(Corpus, 42))
        db.session.commit()
        with mock.patch.object(corpus_index_worker, "app", self.app), \
                mock.patch.object(qdrant_utils, "_read_index_blocks",
                                  side_effect=MinerUError("MinerU OCR 总处理时间超时")):
            for attempt in range(1, 4):
                job = db.session.get(CorpusIndexJob, 42)
                job.next_attempt_at = datetime.utcnow() - timedelta(seconds=1)
                db.session.commit()
                claim = corpus_index_worker.claim_next_job()
                with self.assertLogs(corpus_index_worker.LOGGER, level="ERROR"):
                    corpus_index_worker.process_job(*claim)
                db.session.expire_all()
                job = db.session.get(CorpusIndexJob, 42)
                self.assertEqual(job.attempts, attempt)
                self.assertEqual(job.status, "PENDING" if attempt < 3 else "FAILED")
                self.assertEqual((job.vector_status, job.lexical_status), ("FAILED", "FAILED"))
                self.assertIn("MinerU", job.error_message)
                self.assertFalse(db.session.get(Corpus, 42).is_vectorized)
        self.client.upsert.assert_not_called()
        self.store.write_points.assert_not_called()
        self.client.delete.assert_called()
        self.store.delete_corpus.assert_called()

    def test_worker_success_keeps_ocr_warning_and_clean_rebuild_clears_it(self):
        """跳页后真实完成双路发布，成功收尾保留摘要，下次完整重建清除旧警告。"""
        warning = "PDF OCR 已跳过: file=paper.pdf page=105 code=data_inspection_failed request_id=req-skip"
        blocks = [{"text": "Remaining readable evidence", "page_idx": 105,
                   "bbox": [0, 0, 1, 1], "block_id": 0}]

        def parse_with_warning(*_args):
            """模拟解析层在当前构建上下文记录一次跳页并返回其他页的真实正文。"""
            record_ocr_warning(warning)
            return blocks

        enqueue_corpus_index(db.session, db.session.get(Corpus, 42))
        job = db.session.get(CorpusIndexJob, 42)
        job.next_attempt_at = datetime.utcnow() - timedelta(seconds=1)
        job.error_message = "previous failure"
        db.session.commit()
        with mock.patch.object(corpus_index_worker, "app", self.app), \
                mock.patch.object(qdrant_utils, "_read_index_blocks", side_effect=parse_with_warning):
            claim = corpus_index_worker.claim_next_job()
            corpus_index_worker.process_job(*claim)
        db.session.expire_all()
        job = db.session.get(CorpusIndexJob, 42)
        self.assertEqual((job.status, job.vector_status, job.lexical_status), ("SUCCESS",) * 3)
        self.assertEqual(job.error_message, warning)
        self.assertEqual(job.attempts, 1)
        self.assertTrue(db.session.get(Corpus, 42).is_vectorized)
        self.client.upsert.assert_called_once()
        self.store.write_points.assert_called_once()

        build_indexes(42, self.attachments)
        self.assertEqual(db.session.get(CorpusIndexJob, 42).status, "SUCCESS")
        self.assertIsNone(db.session.get(CorpusIndexJob, 42).error_message)

    def test_direct_build_success_stores_ocr_warning_without_schema_change(self):
        """非 Worker 入口也保留跳页摘要，双路写入结果与统计契约保持一致。"""
        original_blocks = qdrant_utils._read_index_blocks.return_value

        def parse_with_warning(*_args):
            """模拟多页拒绝摘要，由构建服务统一写入既有错误字段。"""
            record_ocr_warning("page=105 code=data_inspection_failed")
            record_ocr_warning("page=108 code=data_inspection_failed")
            return original_blocks

        with mock.patch.object(qdrant_utils, "_read_index_blocks", side_effect=parse_with_warning):
            build_indexes(42, self.attachments)
        job = db.session.get(CorpusIndexJob, 42)
        self.assertEqual((job.status, job.vector_status, job.lexical_status), ("SUCCESS",) * 3)
        self.assertIn("page=105", job.error_message)
        self.assertIn("page=108", job.error_message)
        self.assertTrue(db.session.get(Corpus, 42).is_vectorized)

    def test_reenqueue_during_build_does_not_overwrite_new_task(self):
        """旧执行者失效后可清理自己的快照，但不能把新 PENDING 改成 FAILED。"""
        def reenqueue(_points):
            """模拟构建时用户重新提交审核任务。"""
            enqueue_corpus_index(db.session, db.session.get(Corpus, 42))
            db.session.commit()
        self.store.write_points.side_effect = reenqueue
        with self.assertRaisesRegex(RuntimeError, "新请求"):
            build_indexes(42, self.attachments)
        job = db.session.get(CorpusIndexJob, 42)
        self.assertEqual(job.status, "PENDING")
        self.assertEqual(job.generation, 2)
        self.assertIsNone(job.error_message)

    def test_missing_attachment_or_embedding_is_not_partial_success(self):
        """附件不完整和 embedding 数量不匹配都不能发布部分成功。"""
        with self.assertRaisesRegex(RuntimeError, "附件"):
            build_indexes(42, [{"file_path": "other.pdf"}])
        db.session.rollback()
        with mock.patch.object(qdrant_utils, "_get_embedding", return_value=[]):
            with self.assertRaisesRegex(RuntimeError, "向量数量"):
                build_indexes(42, self.attachments)
        self.assertFalse(db.session.get(Corpus, 42).is_vectorized)

    def test_cleanup_attempts_both_engines_when_one_fails(self):
        """清理一个引擎失败仍尝试另一个，数据库立即撤销可检索状态。"""
        self.client.delete.side_effect = RuntimeError("qdrant unavailable")
        with self.assertRaisesRegex(RuntimeError, "清理失败"):
            delete_indexes(42)
        self.store.delete_corpus.assert_called_once()
        self.assertFalse(db.session.get(Corpus, 42).is_vectorized)

    def test_expired_claim_gets_new_generation(self):
        """过期运行回收递增 generation，使仍存活的旧执行者无法发布。"""
        enqueue_corpus_index(db.session, db.session.get(Corpus, 42))
        job = db.session.get(CorpusIndexJob, 42)
        job.status = "RUNNING"
        job.started_at = datetime.utcnow() - timedelta(hours=1)
        db.session.commit()
        with mock.patch.object(corpus_index_worker, "app", self.app):
            claim = corpus_index_worker.claim_next_job()
        self.assertEqual(claim[2], 2)

    def test_old_table_migration_is_idempotent_and_not_hybrid_ready(self):
        """旧 SUCCESS 迁移后两路默认 PENDING，且第二次迁移不重复加列。"""
        db.session.remove()
        CorpusIndexJob.__table__.drop(db.engine)
        with db.engine.begin() as connection:
            connection.execute(text("CREATE TABLE corpus_index_job (corpus_id BIGINT PRIMARY KEY, status VARCHAR(16))"))
            connection.execute(text("INSERT INTO corpus_index_job VALUES (42, 'SUCCESS')"))
        corpus_index_worker._migrate_job_columns()
        corpus_index_worker._migrate_job_columns()
        with db.engine.connect() as connection:
            row = connection.execute(text("SELECT vector_status, lexical_status FROM corpus_index_job")).one()
        self.assertEqual(tuple(row), ("PENDING", "PENDING"))

    def test_opensearch_uses_point_id_and_two_analyzers(self):
        """实际适配器使用 point ID，单节点 mapping 不依赖额外中文插件。"""
        store = object.__new__(OpenSearchStore)
        store.index_name = "test"
        store.client = mock.MagicMock()
        store.client.indices.exists.return_value = False
        store.ensure_index()
        properties = store.client.indices.create.call_args.kwargs["body"]["mappings"]["properties"]
        self.assertEqual(properties["content"]["analyzer"], "cjk")
        self.assertEqual(properties["content_standard"]["analyzer"], "standard")
        build_indexes(42, self.attachments)
        points = self.store.write_points.call_args.args[0]
        with mock.patch("app.indexing.opensearch_store.bulk") as bulk:
            store.write_points(points)
        action = bulk.call_args.args[1][0]
        self.assertEqual(action["_id"], str(points[0].id))
        self.assertEqual(action["_source"]["content"], points[0].payload["text"])


if __name__ == "__main__":
    unittest.main()
