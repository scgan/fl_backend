"""幂等补齐新版 Backend 的所有者、索引任务与知识库结构，不登记或消费索引任务。"""

from app.app import app
from app.exts import db
from app.indexing.service import corpus_lock
from app.models import KnowledgeBase, KnowledgeBaseDocument
from app.workers.corpus_index_worker import migrate_index_job_table


def main():
    """在共享迁移锁内补齐新版业务表，保留已有文献、索引状态及知识库数据。"""
    # API 与 Worker 使用同一把锁，避免首次部署同时增加列或创建表。
    # 已有迁移允许同线程重入；这里只建表，不回填历史文献或重置任务状态。
    with app.app_context(), corpus_lock(-1):
        migrate_index_job_table()
        KnowledgeBase.__table__.create(bind=db.engine, checkfirst=True)
        KnowledgeBaseDocument.__table__.create(bind=db.engine, checkfirst=True)


if __name__ == '__main__':
    main()
