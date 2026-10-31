"""在审核和附件变更事务中维护持久化的文献索引任务。"""

from __future__ import annotations

import hashlib
import os
from datetime import datetime

from app.models import CorpusIndexJob


def pending_status():
    """本地可使用独立排队状态，旧 Worker 只识别 PENDING，无法抢走新任务。"""
    return os.getenv('FOREIGN_INDEX_QUEUE_STATUS', 'PENDING')


def attachment_digest(attachment: str | None) -> str:
    """计算当前附件清单指纹，阻止后台发布过期文件的片段。

    Args:
        attachment: corpus.attachment 中的原始 JSON 字符串。

    Returns:
        附件清单的 SHA-256 十六进制摘要。
    """
    return hashlib.sha256((attachment or "").encode("utf-8")).hexdigest()


def enqueue_corpus_index(session, corpus) -> None:
    """在审核通过或个人文件上传的同一事务中登记或更新索引任务。

    Args:
        session: Foreign 当前数据库会话，由调用方提交。
        corpus: 已有真实附件且具备公共或个人索引资格的文献。
    """
    if not corpus.attachment:
        return
    # 1. 以文献主键复用唯一当前任务，重新请求推进代次，不额外插入多条任务
    job = session.get(CorpusIndexJob, corpus.id)
    now = datetime.utcnow()
    if job is None:
        job = CorpusIndexJob(corpus_id=corpus.id, generation=1)
        session.add(job)
    else:
        job.generation += 1
    # 2. 将任务重置为当前附件的待处理快照，并撤销文献就绪标记
    # 与审核或附件变更由调用方同一事务提交，外部 Worker 只消费已提交的新版本。
    job.attachment_digest = attachment_digest(corpus.attachment)
    job.status = pending_status()
    job.vector_status = "PENDING"
    job.lexical_status = "PENDING"
    job.build_id = None
    job.chunk_count = 0
    job.attempts = 0
    job.next_attempt_at = now
    job.requested_at = now
    job.started_at = None
    job.finished_at = None
    job.error_message = None
    job.is_deleted = False
    corpus.is_vectorized = False


def cancel_corpus_index(session, corpus_id: int) -> None:
    """附件更改后使未完成的旧任务失效。

    Args:
        session: Foreign 当前数据库会话，由调用方提交。
        corpus_id: 已修改附件的文献 ID。
    """
    job = session.get(CorpusIndexJob, corpus_id)
    if job is not None and not job.is_deleted:
        # 推进代次使在途旧执行者失效；与调用方附件修改一起提交，不能提前独立提交。
        job.generation += 1
        job.status = "CANCELLED"
        job.vector_status = "PENDING"
        job.lexical_status = "PENDING"
        job.finished_at = datetime.utcnow()
