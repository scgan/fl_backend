"""在 Backend 运行环境同步补齐明确账号和文献 ID 的题录，再排队索引；默认只预览。"""

import argparse
import json

from app.app import app
from app.exts import db
from app.indexing.jobs import pending_status
from app.services.personal_metadata import (
    has_personal_metadata, prepare_personal_index, extract_and_queue_personal_metadata,
    metadata_processing_active, fail_personal_metadata,
)
from app.models import Corpus, CorpusIndexJob, User


def backfill(username, corpus_ids, apply=False):
    """在 Backend 环境同步补齐指定账号的目标题录，再将完整材料交给 Worker。

    Args:
        username: 明确的文件所有者账号。
        corpus_ids: 明确指定的个人文献 ID 列表。
        apply: 是否真正提取并排队；False 只返回计划且不调用模型。

    Returns:
        每个目标的跳过、提取失败或排队结果。

    Raises:
        ValueError: 账号不存在或任一目标不属于该账号的有效个人文件。
    """
    owner = User.query.filter_by(username=username, is_deleted=False).one_or_none()
    if not owner:
        raise ValueError('指定账号不存在')
    targets = sorted(set(corpus_ids))
    query = Corpus.query.filter(Corpus.id.in_(targets), Corpus.owner_id == owner.id,
                                Corpus.status == 4, Corpus.is_deleted == 0)
    rows = query.order_by(Corpus.id).all()
    if [row.id for row in rows] != targets:
        raise ValueError('目标中存在无权处理或无效的个人文献')
    report = []
    for corpus in rows:
        # 每篇独立准备、提取和提交，模型期间不持有其他目标的行锁。
        if apply:
            corpus = Corpus.query.filter(Corpus.id == corpus.id, Corpus.owner_id == owner.id,
                                         Corpus.status == 4, Corpus.is_deleted == 0).populate_existing().with_for_update().one()
        query = CorpusIndexJob.query.filter_by(corpus_id=corpus.id)
        job = (query.with_for_update() if apply else query).one_or_none()
        if has_personal_metadata(corpus):
            action = 'SKIP_COMPLETE'
        elif job and not job.is_deleted and (job.status in {pending_status(), 'RUNNING'}
                                               or metadata_processing_active(job)):
            action = 'SKIP_ACTIVE'
        else:
            action = 'WOULD_EXTRACT_AND_QUEUE'
            if apply:
                prepare_personal_index(corpus)
                job = db.session.get(CorpusIndexJob, corpus.id)
                snapshot = (corpus.id, job.attachment_digest, job.generation)
                db.session.commit()
                try:
                    action = 'QUEUED' if extract_and_queue_personal_metadata(*snapshot) else 'SKIP_CHANGED'
                except Exception:
                    fail_personal_metadata(*snapshot, owner.id)
                    action = 'EXTRACTION_FAILED'
        report.append({'corpus_id': corpus.id, 'action': action})
    if apply:
        db.session.commit()
    else:
        db.session.rollback()
    return report


def main():
    """解析明确的处理范围，在现有 Backend 配置下执行预览或排队。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--username', required=True)
    parser.add_argument('--corpus-ids', required=True, type=int, nargs='+')
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    with app.app_context():
        print(json.dumps(backfill(args.username, args.corpus_ids, args.apply), ensure_ascii=False))


if __name__ == '__main__':
    main()
