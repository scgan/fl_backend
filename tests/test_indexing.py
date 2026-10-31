"""离线验证 Foreign 批量与单篇索引的原文路径和定位元数据。"""

from __future__ import annotations

import os
import json
import unittest
from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest import mock

from flask import Flask

from app.app import app as foreign_app
from app.indexing.jobs import cancel_corpus_index, enqueue_corpus_index
from app.models import db
from app.models import CorpusIndexJob
from app.utils import qdrant_utils
from app.workers import corpus_index_worker
from scripts import batch_vectorize_db


class IndexingTests(unittest.TestCase):
    """覆盖两条入口共用的分块和文件来源契约。"""

    def test_worker_process_returns_after_one_completed_job(self):
        """一篇任务完成后不得领取第二篇，避免继续复用已经占用大量内存的进程。"""
        stop = mock.Mock()
        stop.is_set.side_effect = [False, False, True]
        claim = (42, "digest", 1)
        with mock.patch.object(corpus_index_worker.signal, "signal"), \
                mock.patch.object(corpus_index_worker, "claim_next_job", return_value=claim) as take, \
                mock.patch.object(corpus_index_worker, "process_job") as process:
            corpus_index_worker.run_worker(stop)
        take.assert_called_once()
        process.assert_called_once_with(*claim)
        stop.wait.assert_not_called()

    def test_idle_worker_polls_until_a_job_is_available(self):
        """空队列继续等待，领取一篇并完成后才退出，不因空轮询触发进程回收。"""
        stop = mock.Mock()
        stop.is_set.side_effect = [False, False, True]
        claim = (42, "digest", 1)
        with mock.patch.object(corpus_index_worker.signal, "signal"), \
                mock.patch.object(corpus_index_worker, "claim_next_job", side_effect=[None, claim]) as take, \
                mock.patch.object(corpus_index_worker, "process_job") as process:
            corpus_index_worker.run_worker(stop)
        self.assertEqual(take.call_count, 2)
        process.assert_called_once_with(*claim)
        stop.wait.assert_called_once_with(corpus_index_worker.POLL_SECONDS)

    def test_incompatible_collection_is_never_cached_as_a_valid_client(self):
        """验证维度或距离不兼容的集合客户端不会进入成功缓存。"""
        client = mock.MagicMock()
        client.get_collections.return_value.collections = [SimpleNamespace(name=qdrant_utils.COLLECTION_NAME)]
        client.get_collection.return_value.config.params.vectors = qdrant_utils.models.VectorParams(
            size=768, distance=qdrant_utils.models.Distance.COSINE)
        with mock.patch.object(qdrant_utils, "_qdrant_client", None), \
             mock.patch.object(qdrant_utils.qdrant_client, "QdrantClient", return_value=client):
            with self.assertRaisesRegex(RuntimeError, "无法连接 Qdrant"):
                qdrant_utils.get_qdrant_client(host="offline-host")
            self.assertIsNone(qdrant_utils._qdrant_client)

    def test_long_running_ocr_is_reclaimed_only_after_mysql_lock_is_released(self):
        """在 SQLite 注册 MySQL 锁函数替身，执行真实过期更新和重新领取。"""
        app = Flask(__name__)
        app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite://"
        db.init_app(app)
        held = {"foreign-corpus-index:42": 100}
        with app.app_context():
            CorpusIndexJob.__table__.create(bind=db.engine)
            connection = db.engine.raw_connection()
            connection.create_function("IS_USED_LOCK", 1, lambda key: held.get(key))
            connection.create_function("CONCAT", 2, lambda a, b: str(a) + str(b))
            connection.close()
            corpus = SimpleNamespace(id=42, attachment='[{"file_path":"scan.pdf"}]',
                                     is_vectorized=False)
            enqueue_corpus_index(db.session, corpus)
            # 领取器按整秒判断到期；明确构造已到期任务，不依赖测试执行速度。
            db.session.get(CorpusIndexJob, 42).next_attempt_at = datetime.utcnow() - timedelta(seconds=1)
            db.session.commit()
            with mock.patch.object(corpus_index_worker, "app", app):
                claim = corpus_index_worker.claim_next_job()
                job = db.session.get(CorpusIndexJob, 42)
                job.started_at = datetime.utcnow() - timedelta(hours=1)
                db.session.commit()
                with mock.patch.object(db.engine.dialect, "name", "mysql"):
                    self.assertIsNone(corpus_index_worker.claim_next_job())
                    self.assertEqual(db.session.get(CorpusIndexJob, 42).generation, claim[2])
                    held.clear()
                    reclaimed = corpus_index_worker.claim_next_job()
                    self.assertEqual(reclaimed[0], 42)
                    self.assertEqual(reclaimed[2], claim[2] + 1)
            db.session.remove()
            db.engine.dispose()

    def test_chunk_pages_and_character_offsets_reset_after_flush(self):
        """分片页偏移应进入 bbox，后续 chunk 不沿用上一页的起始位置。"""
        blocks = [
            {"text": "abc!", "page_idx": 0, "bbox": [0, 0, 1, 1], "block_id": 1},
            {"text": "def!", "page_idx": 1, "bbox": [0, 0, 1, 1], "block_id": 2},
        ]
        chunks = qdrant_utils._blocks_to_chunks(
            blocks, "part2.pdf", chunk_size=3, page_offset=200,
            max_chunk_size=3, chunk_overlap=0, min_chunk_size=1
        )
        self.assertEqual([item["metadata"]["page_start"] for item in chunks], [200, 201])
        self.assertEqual([item["metadata"]["char_start"] for item in chunks], [0, 4])
        self.assertEqual(chunks[1]["metadata"]["bbox_blocks"][0]["page_idx"], 201)

    def test_single_index_downloads_when_source_is_not_mounted(self):
        """单篇入口可从文件服务读取原文，临时文件随后删除。"""
        app = Flask(__name__)
        app.config["FILE_URL"] = "http://file-service"
        response = mock.MagicMock()
        response.__enter__.return_value = response
        response.iter_content.return_value = [b"%PDF-test"]
        seen = []

        def parse_file(path, **_kwargs):
            """记录下载临时文件，断言文件内容存在后返回可定位的原文块。"""
            seen.append(path)
            self.assertTrue(os.path.isfile(path))
            return [{"text": "text", "page_idx": 0}]

        with app.app_context(), mock.patch.object(
            qdrant_utils.requests, "get", return_value=response
        ) as download, mock.patch.object(
            qdrant_utils, "_read_blocks_locally", side_effect=parse_file
        ):
            blocks = qdrant_utils._read_index_blocks(
                "uploaded_files/corpus/paper.pdf", None, 42
            )

        self.assertEqual(blocks[0]["text"], "text")
        self.assertFalse(os.path.exists(seen[0]))
        self.assertEqual(download.call_args.kwargs["params"]["filename"],
                         "uploaded_files/corpus/paper.pdf")

    def test_local_pdf_blocks_keep_page_and_line_bbox(self):
        """本地 PDF 文字层提供逐行页码和 PDF 坐标。"""
        first = mock.Mock(height=800, edges=[])
        first.extract_words.return_value = [
            {"text": "第一行", "x0": 10, "x1": 70, "top": 20, "bottom": 35}
        ]
        second = mock.Mock(height=800, edges=[])
        second.extract_words.return_value = [
            {"text": "Second line", "x0": 12, "x1": 90, "top": 50, "bottom": 66}
        ]
        pdf = mock.MagicMock()
        pdf.__enter__.return_value.pages = [first, second]
        with mock.patch.object(qdrant_utils.pdfplumber, "open", return_value=pdf):
            blocks = qdrant_utils._read_blocks_locally("paper.pdf")

        self.assertEqual([item["page_idx"] for item in blocks], [0, 1])
        self.assertEqual(blocks[0]["bbox"], [10, 765, 70, 780])
        self.assertEqual(blocks[1]["text"], "Second line\n")

    def test_payload_keeps_original_attachment_and_split_page(self):
        """Qdrant 片段引用原附件，内部解析仍使用临时文件。"""
        client = mock.MagicMock()
        corpus = SimpleNamespace(is_vectorized=False)
        with mock.patch.object(qdrant_utils, "get_qdrant_client", return_value=client), \
             mock.patch.object(qdrant_utils, "_read_index_blocks", return_value=[
                 {"text": "proof", "page_idx": 1, "bbox": [0, 0, 1, 1], "block_id": 1}
             ]), \
             mock.patch.object(qdrant_utils, "_get_embedding", return_value=[[0.1] * 1024]), \
             mock.patch("app.indexing.opensearch_store.get_lexical_store") as lexical, \
             mock.patch.object(db.session, "query") as query, \
             mock.patch.object(db.session, "commit"):
            query.return_value.filter.return_value.first.return_value = corpus
            result = qdrant_utils._index_corpus_chunks(
                42,
                [{"file_path": "/tmp/tmp-part2.pdf",
                  "source_file_path": "uploaded_files/corpus/paper.pdf",
                  "file_name": "paper.pdf", "chunk_name": "paper_part2.pdf",
                  "page_offset": 200}],
                corpus_meta={"title_cn": "paper"},
                index_context={"build_id": "test-build", "attachment_digest": "test-digest"},
            )

        payload = client.upsert.call_args.kwargs["points"][0].payload
        self.assertEqual(result, {"indexed": 1, "total_chunks": 1})
        self.assertEqual(payload["file_path"], "uploaded_files/corpus/paper.pdf")
        self.assertEqual(payload["file_name"], "paper.pdf")
        self.assertEqual(payload["page_start"], 201)
        self.assertEqual(payload["bbox_blocks"][0]["page_idx"], 201)
        self.assertFalse(corpus.is_vectorized)
        lexical.return_value.write_points.assert_called_once()
        self.assertEqual(lexical.return_value.write_points.call_args.args[0][0].id, 42000000)

    def test_pdf_open_failure_uses_real_text_with_same_coordinates(self):
        """备用 PDF 解析保留真实页码和左下原点坐标，不凭空制造文字。"""
        import fitz
        page = mock.Mock()
        page.rect.height = 800
        page.get_text.side_effect = lambda kind: (
            [(10, 20, 70, 35, "真实原文", 0, 0, 0)] if kind == "words" else {"blocks": []})
        page.find_tables.return_value.tables = []
        document = mock.MagicMock()
        document.__enter__.return_value = [page]
        with mock.patch.object(qdrant_utils.pdfplumber, "open", side_effect=ValueError("No /Root")), \
             mock.patch.object(fitz, "open", return_value=document):
            blocks = qdrant_utils._read_blocks_locally("paper.pdf")
        self.assertEqual(blocks[0]["text"], "真实原文\n")
        self.assertEqual(blocks[0]["page_idx"], 0)
        self.assertEqual(blocks[0]["bbox"], [10, 765, 70, 780])

    def test_batch_preserves_attachment_and_part_offsets(self):
        """数据库批量入口为每个分片传入原路径及对应原文页偏移。"""
        split_files = []
        captured = []

        def split_file(path, max_pages):
            """生成可控分片路径，供原件引用及原页偏移断言使用。"""
            self.assertEqual(max_pages, 200)
            for index in (1, 2):
                part = f"{path}_part{index}.pdf"
                with open(part, "wb") as output:
                    output.write(b"pdf")
                split_files.append(part)
            return split_files

        def capture(_corpus_id, attachments, **_kwargs):
            """捕获批量入口传给双路服务的分片、原件路径及页偏移。"""
            captured.extend(attachments)
            return {"indexed": 2, "total_chunks": 2}

        with mock.patch.object(batch_vectorize_db, "download_file_from_service",
                               return_value=b"pdf"), \
             mock.patch.object(batch_vectorize_db, "split_pdf_if_needed",
                               side_effect=split_file), \
             mock.patch.object(batch_vectorize_db.qdrant_utils,
                               "index_corpus_chunks", side_effect=capture), \
             mock.patch.object(batch_vectorize_db.qdrant_utils,
                               "delete_corpus_chunks"), \
             mock.patch.object(db.session, "get",
                               return_value=SimpleNamespace(is_vectorized=True)), \
             mock.patch.object(db.session, "commit"):
            result = batch_vectorize_db._process_one(
                42,
                [{"file_path": "uploaded_files/corpus/paper.pdf",
                  "file_name": "paper.pdf"}],
                {"title_cn": "paper"},
                SimpleNamespace(force=False),
            )

        self.assertIsNone(result["error"])
        self.assertEqual([item["page_offset"] for item in captured], [0, 200])
        self.assertEqual([item["source_file_path"] for item in captured],
                         ["uploaded_files/corpus/paper.pdf"] * 2)
        self.assertEqual([item["file_name"] for item in captured], ["paper.pdf"] * 2)
        self.assertTrue(all(not os.path.exists(path) for path in split_files))

    def test_single_index_replaces_old_points_by_corpus_id(self):
        """单篇索引先清理该文献全部旧片段，并传递附件原路径。"""
        corpus = SimpleNamespace(
            id=42, is_vectorized=True,
            attachment=json.dumps([{"file_path": "uploaded_files/corpus/paper.pdf"}]),
        )
        with mock.patch.object(foreign_app, "before_request_funcs", {None: []}), \
             mock.patch.object(db.session, "query") as query, \
             mock.patch.object(db.session, "commit"), \
             mock.patch.object(qdrant_utils, "delete_corpus_chunks") as delete, \
             mock.patch.object(qdrant_utils, "index_corpus_chunks",
                               return_value={"indexed": 1, "total_chunks": 3}) as index:
            query.return_value.filter.return_value.first.return_value = corpus
            response = foreign_app.test_client().post(
                "/ai_search/internal/corpus/42/index"
            )

        self.assertEqual(response.status_code, 200)
        delete.assert_called_once_with(42)
        self.assertEqual(index.call_args.args[1][0]["source_file_path"],
                         "uploaded_files/corpus/paper.pdf")
        self.assertFalse(corpus.is_vectorized)

    def test_approval_enqueues_durable_job_and_change_cancels_it(self):
        """审核入队、领取和附件变更撤销共享同一条持久化任务。"""
        app = Flask(__name__)
        app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite://"
        db.init_app(app)
        with app.app_context():
            CorpusIndexJob.__table__.create(bind=db.engine)
            corpus = SimpleNamespace(
                id=42, attachment='[{"file_path":"paper.pdf"}]', is_vectorized=True
            )
            enqueue_corpus_index(db.session, corpus)
            db.session.get(CorpusIndexJob, 42).next_attempt_at = datetime.utcnow() - timedelta(seconds=1)
            db.session.commit()
            self.assertFalse(corpus.is_vectorized)
            with mock.patch.object(corpus_index_worker, "app", app):
                claim = corpus_index_worker.claim_next_job()
            self.assertEqual(claim[0], 42)
            job = db.session.get(CorpusIndexJob, 42)
            self.assertEqual(job.status, "RUNNING")
            cancel_corpus_index(db.session, 42)
            db.session.commit()
            self.assertEqual(job.status, "CANCELLED")
            self.assertNotEqual(job.generation, claim[2])

    def test_search_excludes_unapproved_and_unfinished_corpus(self):
        """共享 Qdrant 的旧片段必须经过已审核且索引完成的数据库复核。"""
        hits = [
            SimpleNamespace(payload={"corpus_id": 11, "text": "draft"}, score=0.9),
            SimpleNamespace(payload={"corpus_id": "42", "text": "approved"}, score=0.8),
            SimpleNamespace(payload={"corpus_id": 43, "text": "running"}, score=0.7),
        ]
        client = mock.Mock()
        client.query_points.return_value.points = hits
        with foreign_app.app_context(), \
             mock.patch.object(qdrant_utils, "get_qdrant_client", return_value=client), \
             mock.patch.object(qdrant_utils, "_get_embedding", return_value=[[0.1]]), \
             mock.patch.object(db.session, "query") as query:
            query.return_value.filter.return_value.all.return_value = [(42,)]
            results = qdrant_utils.search_chunks("question", top_k=2)

        self.assertEqual([item["text"] for item in results], ["approved"])
        self.assertEqual(client.query_points.call_args.kwargs["limit"], 10)


if __name__ == "__main__":
    unittest.main()
