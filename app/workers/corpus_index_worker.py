"""轮询唯一 MySQL 索引任务表，两路索引都成功后发布 Foreign 文献。"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import random
import time
from datetime import datetime, timedelta
from sqlalchemy import inspect, text
from sqlalchemy.exc import OperationalError

from app.app import app
from app.exts import db
from app.indexing.jobs import attachment_digest, pending_status
from app.indexing.eligibility import can_index_corpus
from app.indexing.personal_migration import migrate_personal_owner
from app.models import Corpus, CorpusIndexJob
from app.indexing.service import corpus_lock, IndexBusy
from app.utils.qdrant_utils import index_corpus_chunks
from app.workers.process_pool import run_processes


LOGGER = logging.getLogger(__name__)
# 每轮领取最多三次；进程内瞬时数据库冲突重试不等于文献索引尝试次数。
MAX_ATTEMPTS = 3
# 空队列或轮询异常时的等待秒数，停止事件可提前唤醒。
POLL_SECONDS = 5
# 只有超过此期限且 MySQL 文献锁已释放的运行任务才允许被回收。
STALE_AFTER = timedelta(minutes=30)


def _migrate_job_columns() -> None:
    """为既有任务表幂等增加双路状态；旧 SUCCESS 默认仍未完成 BM25。"""
    additions = {
        "vector_status": "VARCHAR(16) NOT NULL DEFAULT 'PENDING'",
        "lexical_status": "VARCHAR(16) NOT NULL DEFAULT 'PENDING'",
        "build_id": "VARCHAR(32) NULL",
        "lexical_index": "VARCHAR(128) NULL",
        "chunk_count": "INTEGER NOT NULL DEFAULT 0",
    }
    existing = {column["name"] for column in inspect(db.engine).get_columns("corpus_index_job")}
    with db.engine.begin() as connection:
        for name, definition in additions.items():
            if name not in existing:
                connection.execute(text(f"ALTER TABLE corpus_index_job ADD COLUMN {name} {definition}"))


def migrate_index_job_table() -> None:
    """幂等升级个人文件所有者列和唯一索引任务表，保留原有文献数据。"""
    with app.app_context():
        with corpus_lock(-1):
            migrate_personal_owner(db.engine)
            CorpusIndexJob.__table__.create(bind=db.engine, checkfirst=True)
            _migrate_job_columns()


def claim_next_job() -> tuple[int, str, int] | None:
    """领取一个任务，仅对 MySQL 锁等待超时或死锁最多尝试三次。

    Returns:
        文献 ID、附件指纹、任务代次组成的快照；暂无可领任务时返回 None。

    Raises:
        OperationalError: 不是可重试锁冲突，或三次尝试后仍然冲突。
    """
    for attempt in range(3):
        try:
            return _claim_next_job_once()
        except OperationalError as error:
            code = error.orig.args[0] if getattr(error.orig, 'args', ()) else None
            if code not in (1205, 1213) or attempt == 2:
                raise
            LOGGER.warning('任务领取事务冲突，重试: mysql_code=%s attempt=%s', code, attempt + 1)
            time.sleep(random.uniform(0.05, 0.15) * (attempt + 1))
    return None


def _claim_next_job_once() -> tuple[int, str, int] | None:
    """回收失去执行者的过期任务，再通过条件更新原子领取到期任务。

    Returns:
        已提交 RUNNING 的任务快照 (corpus_id, attachment_digest, generation)，
        没有候选或所有候选已被其他进程领取时返回 None。
    """
    # 1. 找出已过期且没有持锁执行者的任务，防止回收仍在进行长耗时 OCR 的任务
    # 旧 DATETIME 没有小数精度；去掉微秒避免 MySQL 向上舍入后出现尚未到期的假象。
    now = datetime.utcnow().replace(microsecond=0)
    with app.app_context():
        stale_jobs = db.session.query(CorpusIndexJob).filter(
            CorpusIndexJob.status == "RUNNING",
            CorpusIndexJob.started_at < now - STALE_AFTER,
            CorpusIndexJob.is_deleted.is_(False),
        )
        # 长文档 OCR 可能超过 30 分钟。仍持有文献锁的执行者不可被回收；
        # 进程退出/连接断开会自动释放 MySQL 命名锁，遗留任务仍可恢复。
        if db.engine.dialect.name == "mysql":
            stale_jobs = stale_jobs.filter(text(
                "IS_USED_LOCK(CONCAT('foreign-corpus-index:', corpus_index_job.corpus_id)) IS NULL"
            ))
        stale = stale_jobs.with_entities(CorpusIndexJob.corpus_id, CorpusIndexJob.generation).order_by(
            CorpusIndexJob.corpus_id).limit(32).all()
        # 候选读取后结束事务，后续逐条复核，避免长时间持有查询快照。
        db.session.commit()
        # 2. 逐主键复核并回收，同时增加代次，使失联的旧执行者不能发布结果
        # 不对整段 RUNNING 索引加范围锁，避免阻塞其他文献的完成更新并形成死锁。
        for cid, generation in stale:
            stale_row = db.session.query(CorpusIndexJob).filter(
                CorpusIndexJob.corpus_id == cid, CorpusIndexJob.generation == generation,
                CorpusIndexJob.status == 'RUNNING', CorpusIndexJob.started_at < now - STALE_AFTER,
                CorpusIndexJob.is_deleted.is_(False))
            if db.engine.dialect.name == 'mysql':
                stale_row = stale_row.filter(text(
                    "IS_USED_LOCK(CONCAT('foreign-corpus-index:', corpus_index_job.corpus_id)) IS NULL"))
            stale_row.update({CorpusIndexJob.status: pending_status(),
                              CorpusIndexJob.generation: CorpusIndexJob.generation + 1,
                              CorpusIndexJob.next_attempt_at: now}, synchronize_session=False)
            db.session.commit()

        # 3. 读取到期待处理候选后结束读事务，实际领取仍由条件更新竞争决定
        candidates = db.session.query(CorpusIndexJob.corpus_id).filter(
            CorpusIndexJob.status == pending_status(), CorpusIndexJob.next_attempt_at <= now,
            CorpusIndexJob.is_deleted.is_(False)).order_by(
                CorpusIndexJob.requested_at, CorpusIndexJob.corpus_id).limit(32).all()
        db.session.commit()
        for (cid,) in candidates:
            # 4. 原子竞争单条任务，并在外部解析前提交 RUNNING 与尝试次数
            # 只有条件更新命中的进程拥有任务；无需 MySQL 8 的 SKIP LOCKED。
            changed = db.session.query(CorpusIndexJob).filter(
                CorpusIndexJob.corpus_id == cid, CorpusIndexJob.status == pending_status(),
                CorpusIndexJob.next_attempt_at <= now, CorpusIndexJob.is_deleted.is_(False)
            ).update({CorpusIndexJob.status: 'RUNNING', CorpusIndexJob.started_at: now,
                      CorpusIndexJob.attempts: CorpusIndexJob.attempts + 1}, synchronize_session=False)
            if changed:
                job = db.session.get(CorpusIndexJob, cid, populate_existing=True)
                claim = (job.corpus_id, job.attachment_digest, job.generation)
                db.session.commit()
                return claim
            db.session.rollback()
        return None


def _valid_source(corpus: Corpus | None, digest: str) -> bool:
    """仅允许具备公共或个人索引资格的同版本附件，不改变检索读取权限。"""
    return bool(can_index_corpus(corpus)
                and corpus.attachment and attachment_digest(corpus.attachment) == digest)


def _mark_failed(corpus_id: int, digest: str, generation: int, error: Exception) -> None:
    """在当前文献锁下清理部分索引，仅为相同代次登记重试或最终失败。

    Args:
        corpus_id: 当前文献 ID。
        digest: 领取时的附件指纹，避免更改新附件的就绪标记。
        generation: 领取时的任务代次，过期执行者不能覆盖新任务。
        error: 索引失败原因，按任务错误字段长度截断保存。
    """
    # 1. 回滚失败事务并重新检查代次，旧执行者不再修改已经被替换的任务
    db.session.rollback()
    job = db.session.get(CorpusIndexJob, corpus_id)
    if not job or job.generation != generation:
        return
    # 2. 尽力删除双路残留；清理失败仍须记录任务错误，不能把任务留在 RUNNING
    try:
        from app.indexing.service import delete_indexes
        delete_indexes(corpus_id, preserve_job=True)
    except Exception:
        LOGGER.exception("清理失败索引片段出错: corpus_id=%s", corpus_id)
    # 3. 清理过程可能提交事务，因此重查当前快照后再写退避或最终失败状态
    with app.app_context():
        db.session.rollback()
        corpus = db.session.get(Corpus, corpus_id)
        if corpus and attachment_digest(corpus.attachment) == digest:
            corpus.is_vectorized = False
        job = db.session.get(CorpusIndexJob, corpus_id)
        if job and job.generation == generation:
            job.error_message = str(error)[:2000]
            if job.attempts < MAX_ATTEMPTS:
                job.status = pending_status()
                job.next_attempt_at = datetime.utcnow() + timedelta(seconds=30 * job.attempts)
            else:
                job.status = "FAILED"
                job.finished_at = datetime.utcnow()
        # 让前端与下一轮消费者同时看到错误摘要、就绪撤销和重试时间。
        db.session.commit()


def process_job(corpus_id: int, digest: str, generation: int) -> None:
    """持有文献锁直到发布或清理结束；锁冲突时退回队列且不消耗索引尝试。

    Args:
        corpus_id: 已领取的文献 ID。
        digest: 领取时的附件指纹。
        generation: 领取时的任务代次。
    """
    with app.app_context():
        try:
            with corpus_lock(corpus_id):
                _process_job(corpus_id, digest, generation)
        except IndexBusy:
            # 同篇文献仍由另一执行者持锁，不把等待锁计作文献解析失败。
            job = db.session.get(CorpusIndexJob, corpus_id)
            if job and job.generation == generation:
                job.status = pending_status()
                job.attempts = max(0, job.attempts - 1)
                job.next_attempt_at = datetime.utcnow() + timedelta(seconds=30)
                db.session.commit()


def _process_job(corpus_id: int, digest: str, generation: int) -> None:
    """持有文献锁构建并复核双路索引，成功时保留本次 OCR 跳页警告。

    Args:
        corpus_id: 已领取的文献 ID。
        digest: 领取时的附件清单指纹。
        generation: 领取时的任务代次，发布和失败处理均需再次复核。

    解析及外部索引异常在本层转入失败清理和有限重试，不对调用方发布部分成功。
    """
    with app.app_context():
        # 1. 复核领取快照，失效输入只取消本代任务，不覆盖后续入队的新任务
        corpus = db.session.get(Corpus, corpus_id)
        job = db.session.get(CorpusIndexJob, corpus_id)
        if not job or job.generation != generation or not _valid_source(corpus, digest):
            if job and job.generation == generation:
                job.status = "CANCELLED"
                job.finished_at = datetime.utcnow()
                db.session.commit()
            return
        try:
            # 2. 保留原附件路径并规范化清单，使临时解析结果仍可定位到真实原件
            attachments = json.loads(corpus.attachment)
            if not isinstance(attachments, list):
                attachments = [attachments]
            attachments = [
                {**item, "source_file_path": item.get("file_path")}
                for item in attachments if isinstance(item, dict) and item.get("file_path")
            ]
            if not attachments:
                raise ValueError("文献没有可索引的附件")

            # 3. 先提交不可用状态再解析正文并构建双路索引。
            corpus.is_vectorized = False
            db.session.commit()
            result = index_corpus_chunks(corpus_id, attachments, mark_vectorized=False,
                                         expected_generation=generation)
            if not result["indexed"] or not result["total_chunks"]:
                raise RuntimeError("原文未生成可检索片段")

            # 4. 清空 ORM 缓存并加行锁重查，防止解析期间的附件或审核变化被旧快照覆盖
            db.session.expire_all()
            corpus = (db.session.query(Corpus).filter(Corpus.id == corpus_id)
                      .with_for_update().one_or_none())
            job = (db.session.query(CorpusIndexJob)
                   .filter(CorpusIndexJob.corpus_id == corpus_id)
                   .with_for_update().one_or_none())
            if not job or job.generation != generation or not _valid_source(corpus, digest):
                if corpus:
                    corpus.is_vectorized = False
                if job and job.generation == generation:
                    job.status = "CANCELLED"
                    job.finished_at = datetime.utcnow()
                db.session.commit()
                from app.indexing.service import delete_indexes
                delete_indexes(corpus_id, preserve_job=True)
                return
            # 5. 两路均成功才将任务完成与文献就绪原子提交，避免单路成功被召回端误用
            if job.vector_status != "SUCCESS" or job.lexical_status != "SUCCESS":
                raise RuntimeError("双路索引尚未全部成功")
            job.status = "SUCCESS"
            job.finished_at = datetime.utcnow()
            # 构建服务已清理旧错误并写入本次跳页摘要，成功收尾不能再次清空。
            corpus.is_vectorized = True
            db.session.commit()
            LOGGER.info("文献索引完成: corpus_id=%s, chunks=%s", corpus_id, result["total_chunks"])
        except Exception as error:
            # 6. 任一阶段失败都进入同代次补偿与退避，保留原失败原因供任务查询
            LOGGER.exception("文献索引失败: corpus_id=%s", corpus_id)
            _mark_failed(corpus_id, digest, generation, error)


def run_worker(stop) -> None:
    """空闲时轮询，一篇任务完成收尾后退出，由父进程补位并释放驻留内存。

    Args:
        stop: 父进程共享的停止事件。
    """
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    while not stop.is_set():
        try:
            claim = claim_next_job()
            if claim:
                process_job(*claim)
                # 成功、取消或登记重试均已结束本次任务，退出以释放第三方库与分配器的高水位。
                # 文献锁和事务由 process_job 收尾；不能在索引写入中途回收进程。
                LOGGER.info("本篇任务结束，回收消费进程: corpus_id=%s pid=%s", claim[0], os.getpid())
                return
        except Exception:
            LOGGER.exception("文献索引 worker 轮询失败")
            with app.app_context():
                db.session.rollback()
        stop.wait(POLL_SECONDS)


def main() -> None:
    """先迁移任务表，再启动默认八个独立索引消费进程。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--migrate-only", action="store_true")
    parser.add_argument("--concurrency", type=int,
                        default=os.getenv("INDEX_WORKER_CONCURRENCY", "8"))
    args = parser.parse_args()
    if args.concurrency < 1:
        parser.error("--concurrency must be a positive integer")
    migrate_index_job_table()
    if args.migrate_only:
        return
    LOGGER.info("Foreign 文献索引 worker 已启动: concurrency=%s", args.concurrency)
    run_processes(run_worker, args.concurrency)


if __name__ == "__main__":
    main()
