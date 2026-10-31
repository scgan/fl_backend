"""部署时幂等补齐 corpus.owner_id；只迁移结构，不启动 Worker 或修改文献数据。"""

from app.app import app
from app.exts import db
from app.indexing.personal_migration import migrate_personal_owner
from app.indexing.service import corpus_lock


def main():
    """与 Worker 迁移共用锁，避免多个部署进程同时增加字段。"""
    with app.app_context():
        with corpus_lock(-1):
            migrate_personal_owner(db.engine)


if __name__ == '__main__':
    main()
