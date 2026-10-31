"""串行化同篇文献的索引操作，并以 MySQL 状态发布完整双路快照。"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime
from contextlib import contextmanager
from contextvars import ContextVar

from sqlalchemy import text

from app.exts import db
from app.indexing.jobs import attachment_digest
from app.indexing.eligibility import can_index_corpus
from app.indexing.opensearch_store import get_lexical_store
from app.models import Corpus, CorpusIndexJob
from app.utils.ocr_errors import collect_ocr_warnings


LOGGER = logging.getLogger(__name__)
_held_locks = ContextVar("corpus_index_locks", default=frozenset())


class IndexBusy(RuntimeError):
    """另一进程仍持有同篇文献的索引锁。"""


@contextmanager
def corpus_lock(corpus_id):
    """用独立 MySQL 连接持锁，业务事务提交不会提前释放锁。

    Args:
        corpus_id: 需要串行处理的文献 ID；允许同线程重入。

    Raises:
        IndexBusy: 十秒内未获得索引锁。
    """
    if corpus_id in _held_locks.get():
        yield
        return
    # SQLite 只用于离线测试，生产 MySQL 使用跨进程命名锁。
    with db.engine.connect() as connection:
        lock_name = f"foreign-corpus-index:{corpus_id}"
        mysql = connection.dialect.name == "mysql"
        if mysql and connection.execute(
            text("SELECT GET_LOCK(:name, 10)"), {"name": lock_name}
        ).scalar() != 1:
            raise IndexBusy("该文献仍有索引任务执行中")
        token = _held_locks.set(_held_locks.get() | {corpus_id})
        try:
            yield
        finally:
            _held_locks.reset(token)
            if mysql:
                connection.execute(text("SELECT RELEASE_LOCK(:name)"), {"name": lock_name})


def delete_indexes(corpus_id, file_names=None, preserve_job=False):
    """在文献锁内先撤销就绪状态，再尽力清理双路副本。

    Args:
        corpus_id: 目标文献主键。
        file_names: 可选附件名列表，部分删除也使整篇混合索引失效。
        preserve_job: 内部清理保留任务代次；为假时取消当前任务并推进 generation。

    Raises:
        RuntimeError: 任一路远程清理失败，状态已撤销，不恢复可检索标记。
    """
    from app.utils.qdrant_utils import _delete_qdrant_chunks

    # 1. 与构建共用文献锁，先在数据库撤销就绪及按需取消任务，再允许远程删除
    with corpus_lock(corpus_id):
        corpus = db.session.query(Corpus).filter(Corpus.id == corpus_id).with_for_update().one_or_none()
        state = db.session.query(CorpusIndexJob).filter(
            CorpusIndexJob.corpus_id == corpus_id
        ).with_for_update().one_or_none()
        if state and not preserve_job:
            state.generation += 1
            state.status = "CANCELLED"
            state.vector_status = "PENDING"
            state.lexical_status = "PENDING"
        if corpus:
            corpus.is_vectorized = False
        db.session.commit()
        # 2. 分别尝试两路清理，即使一侧失败仍尝试另一侧；任何失败都向调用方报告
        errors = []
        for operation in (_delete_qdrant_chunks, get_lexical_store().delete_corpus):
            try:
                operation(corpus_id, file_names)
            except Exception as error:
                errors.append(error)
        if errors:
            raise RuntimeError("双路索引清理失败") from errors[0]


def build_indexes(corpus_id, attachments, corpus_meta=None, base_dir=None, mark_vectorized=True,
                  expected_generation=None):
    """从同一组分块构建两路索引，完整成功并复核当前原文后发布。

    Args:
        corpus_id: 文献 ID。
        attachments: 原附件或带 source_file_path 的临时分片。
        corpus_meta: 兼容既有批量入口；题录最终以当前数据库记录为准。
        base_dir: 本地原文基准目录。
        mark_vectorized: Worker 传 False，待任务 generation 复核后发布文献标志。
        expected_generation: Worker 领取的版本，防止领取后入队的新请求被覆盖。

    Returns:
        既有 indexed、total_chunks 统计；OCR 审核拒绝的跳页警告写入任务 error_message。

    Raises:
        RuntimeError: 原文失效、构建失败或生成空片段。
    """
    from app.utils.qdrant_utils import _index_corpus_chunks

    with corpus_lock(corpus_id):
        # 1. 读取当前审核快照，拒绝为旧附件或撤销审核的文献写索引。
        db.session.expire_all()
        corpus = db.session.query(Corpus).filter(Corpus.id == corpus_id).with_for_update().one_or_none()
        if not can_index_corpus(corpus) or not corpus.attachment:
            raise RuntimeError("文献未审核或原文已失效")
        digest = attachment_digest(corpus.attachment)
        current_attachments = json.loads(corpus.attachment)
        if not isinstance(current_attachments, list):
            current_attachments = [current_attachments]
        paths = {item.get("file_path") for item in current_attachments if isinstance(item, dict)}
        paths.discard(None)
        supplied = {item.get("source_file_path") or item.get("file_path") for item in attachments}
        if not paths or paths != supplied:
            raise RuntimeError("索引附件与当前文献原文不一致")
        build_id = uuid.uuid4().hex
        # 不相信调用方缓存的旧题录；后续分块器从当前文献补全。
        corpus_meta = None
        store = get_lexical_store()

        # 2. 先撤销可检索状态；旧快照删除与新快照写入受同一文献锁保护。
        state = db.session.query(CorpusIndexJob).filter(
            CorpusIndexJob.corpus_id == corpus_id
        ).with_for_update().one_or_none()
        if expected_generation is not None and (
            state is None or state.generation != expected_generation
        ):
            raise RuntimeError("索引任务已被新请求替换")
        if state is None:
            state = CorpusIndexJob(corpus_id=corpus_id, generation=1, attempts=0)
            db.session.add(state)
        elif mark_vectorized:
            state.generation += 1
        generation = state.generation
        state.status = "RUNNING"
        state.vector_status = "RUNNING"
        state.lexical_status = "RUNNING"
        state.is_deleted = False
        state.requested_at = state.requested_at or datetime.utcnow()
        state.started_at = datetime.utcnow()
        state.finished_at = None
        state.next_attempt_at = datetime.utcnow()
        state.build_id = build_id
        state.attachment_digest = digest
        state.lexical_index = store.index_name
        state.error_message = None
        state.chunk_count = 0
        corpus.is_vectorized = False
        db.session.commit()
        try:
            # 3. 一次解析和分块，同时写入 Qdrant 与 OpenSearch；等待两路可见。
            delete_indexes(corpus_id, preserve_job=True)
            store.ensure_index()
            # 页面警告在解析调用线程聚合，不让一次构建的缺页信息泄漏给后续文献。
            with collect_ocr_warnings() as ocr_warnings:
                result = _index_corpus_chunks(
                    corpus_id, attachments, corpus_meta, base_dir, mark_vectorized=False,
                    index_context={"build_id": build_id, "attachment_digest": digest,
                                   "personal_upload": corpus.status == 4},
                )
            if not result["indexed"] or not result["total_chunks"]:
                raise RuntimeError("原文未生成可检索片段")

            # 4. 长耗时解析后重新锁行核验，只有同一原文完整写入才允许发布 SUCCESS。
            db.session.expire_all()
            corpus = db.session.query(Corpus).filter(Corpus.id == corpus_id).with_for_update().one()
            if (not can_index_corpus(corpus)
                    or attachment_digest(corpus.attachment) != digest):
                raise RuntimeError("构建期间文献审核状态或附件发生变化")
            state = db.session.query(CorpusIndexJob).filter(
                CorpusIndexJob.corpus_id == corpus_id
            ).with_for_update().one()
            if state.generation != generation:
                raise RuntimeError("构建期间索引任务已被新请求替换")
            state.vector_status = "SUCCESS"
            state.lexical_status = "SUCCESS"
            state.status = "SUCCESS" if mark_vectorized else "RUNNING"
            state.finished_at = datetime.utcnow() if mark_vectorized else None
            state.chunk_count = result["total_chunks"]
            # 剩余正文双路写入成功即允许发布；沿用错误字段保留已脱敏的跳页摘要。
            state.error_message = "; ".join(sorted(ocr_warnings))[:2000] or None
            corpus.is_vectorized = bool(mark_vectorized)
            db.session.commit()
            return result
        except Exception:
            db.session.rollback()
            # 同篇文献锁仍持有，清理不会误删新执行者的快照；失败清理不掩盖原错。
            try:
                delete_indexes(corpus_id, preserve_job=True)
            except Exception:
                LOGGER.exception("清理部分双路索引失败: corpus_id=%s", corpus_id)
            db.session.rollback()
            state = db.session.query(CorpusIndexJob).filter(
                CorpusIndexJob.corpus_id == corpus_id
            ).with_for_update().one()
            if state.generation == generation:
                state.status = "FAILED"
                state.vector_status = "FAILED"
                state.lexical_status = "FAILED"
                state.error_message = "双路索引构建失败，请检查 Worker 日志"
                state.finished_at = datetime.utcnow()
            db.session.commit()
            raise
