"""预览或登记历史已审核文献的双路索引任务；不在脚本中执行模型调用。"""

from __future__ import annotations

import argparse
import os

from app.app import app
from app.exts import db
from app.indexing.jobs import attachment_digest, enqueue_corpus_index
from app.models import Corpus, CorpusIndexJob
from app.workers.corpus_index_worker import migrate_index_job_table


def main():
    """默认只预览候选，--apply 才升级任务表并登记待处理任务。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="确认登记任务，由 Worker 执行")
    parser.add_argument("--corpus-id", type=int, action="append", help="可重复指定文献 ID")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    if args.apply:
        migrate_index_job_table()
    with app.app_context():
        query = db.session.query(Corpus).filter(
            Corpus.is_deleted == 0, Corpus.status == 2, Corpus.attachment.isnot(None),
        ).order_by(Corpus.id)
        if args.corpus_id:
            query = query.filter(Corpus.id.in_(args.corpus_id))
        if args.limit > 0:
            query = query.limit(args.limit)
        rows = query.all()
        if not args.apply:
            print({"mode": "preview", "candidate_ids": [row.id for row in rows],
                   "note": "预览所有符合条件的文献；执行时跳过就绪及正在处理的任务"})
            return
        count = 0
        for corpus in rows:
            # 与审核入口统一锁顺序：先文献再任务，避免丢失并发入队。
            corpus = db.session.query(Corpus).filter(Corpus.id == corpus.id).with_for_update().one()
            job = db.session.query(CorpusIndexJob).filter(
                CorpusIndexJob.corpus_id == corpus.id
            ).with_for_update().one_or_none()
            if corpus.is_deleted or corpus.status != 2 or not corpus.attachment:
                db.session.commit()
                continue
            digest = attachment_digest(corpus.attachment)
            if job and not job.is_deleted and job.attachment_digest == digest:
                if job.status in {"PENDING", "RUNNING"} or (
                    job.status == "SUCCESS" and job.vector_status == "SUCCESS"
                    and job.lexical_status == "SUCCESS" and corpus.is_vectorized
                    and job.lexical_index == os.getenv("FOREIGN_OPENSEARCH_INDEX", "foreign_chunks_v1")
                    and job.build_id
                ):
                    db.session.commit()
                    continue
            enqueue_corpus_index(db.session, corpus)
            db.session.commit()
            count += 1
        print({"mode": "apply", "enqueued": count})


if __name__ == "__main__":
    main()
