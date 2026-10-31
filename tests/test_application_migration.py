"""用独立 SQLite 验证首次上线补表、重复迁移及已有任务和文献的保留。"""

import unittest
from unittest import mock

from flask import Flask
from sqlalchemy import inspect, text

from app.exts import db
from app.models import CorpusIndexJob, KnowledgeBase
from scripts import migrate_application_schema


class ApplicationMigrationTests(unittest.TestCase):
    """以不含新版结构的数据库验证真实迁移，禁止依赖生产库或索引服务。"""

    def test_repeated_migration_preserves_existing_corpus_and_job(self):
        """从旧表补齐所有结构，两次执行后仍保留已有文献及双路构建状态。"""
        app = Flask(__name__)
        app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite://'
        db.init_app(app)
        with app.app_context():
            with db.engine.begin() as connection:
                connection.execute(text('CREATE TABLE corpus (id BIGINT PRIMARY KEY, status INTEGER)'))
                connection.execute(text('INSERT INTO corpus VALUES (11, 2)'))
            with mock.patch.object(migrate_application_schema, 'app', app), \
                    mock.patch('app.workers.corpus_index_worker.app', app):
                migrate_application_schema.main()
                db.session.add(CorpusIndexJob(
                    corpus_id=11, attachment_digest='existing-digest', status='SUCCESS',
                    generation=3, vector_status='SUCCESS', lexical_status='SUCCESS',
                    build_id='existing-build', lexical_index='foreign_chunks_prod_v1',
                    chunk_count=9,
                ))
                db.session.add(KnowledgeBase(id=1, owner_id=7, name='existing'))
                db.session.commit()
                migrate_application_schema.main()
            self.assertTrue({'corpus_index_job', 'knowledge_base', 'knowledge_base_document'}
                            .issubset(inspect(db.engine).get_table_names()))
            with db.engine.connect() as connection:
                self.assertEqual(tuple(connection.execute(text(
                    'SELECT status, owner_id FROM corpus WHERE id=11')).one()), (2, None))
                self.assertEqual(tuple(connection.execute(text(
                    'SELECT status, generation, build_id, lexical_index, chunk_count '
                    'FROM corpus_index_job WHERE corpus_id=11')).one()),
                    ('SUCCESS', 3, 'existing-build', 'foreign_chunks_prod_v1', 9))
                self.assertEqual(connection.execute(text(
                    'SELECT name FROM knowledge_base WHERE id=1')).scalar(), 'existing')
            db.session.remove()
            db.engine.dispose()
