"""用隔离 SQLite 和文件服务替身验证个人上传、越权、任务生命周期及公共 SQL 隔离。"""

import io
import json
import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest import mock

from flask import Flask, g, make_response
from sqlalchemy import BigInteger, Integer, create_engine, inspect, text
from sqlalchemy.dialects.mysql import BIT, YEAR

from app.api import literature_files
from app.api.AIsearch import bp as ai_search_bp
from app.api.corpus import bp as corpus_bp
from app.api.file import bp as file_bp
from app.api.public import bp as public_bp
from app.exts import db
from app.indexing.eligibility import can_index_corpus
from app.indexing.jobs import enqueue_corpus_index
from app.indexing.personal_migration import migrate_personal_owner
from app.models import Corpus, CorpusCategory, CorpusIndexJob
from app.utils.public_sql import public_corpus_sql
from app.workers import corpus_index_worker


class PersonalFileTests(unittest.TestCase):
    """执行真实 Flask 路由与事务，外部文件和索引操作全部离线。"""

    def setUp(self):
        """创建两位用户的材料和公共材料，供所有权与兼容性断言复用。"""
        self.app = Flask(__name__)
        self.app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite://'
        db.init_app(self.app)
        for blueprint, prefix in ((literature_files.bp, None), (corpus_bp, '/corpus'),
                                  (file_bp, '/file'), (public_bp, '/public'),
                                  (ai_search_bp, '/ai_search')):
            self.app.register_blueprint(blueprint, url_prefix=prefix)
        self.actor = 7

        @self.app.before_request
        def authenticate():
            """提供确定性的已登录账号，资源授权仍执行真实业务查询。"""
            g.current_user = SimpleNamespace(id=self.actor, username=f'user{self.actor}')
            g.role = 2

        self.context = self.app.app_context()
        self.context.push()
        tables = (Corpus.__table__, CorpusCategory.__table__, CorpusIndexJob.__table__)
        self.types = [(column, column.type) for table in tables for column in table.columns
                      if isinstance(column.type, (BIT, YEAR, BigInteger))]
        for column, kind in self.types:
            column.type = kind.with_variant(Integer(), 'sqlite')
        for table in tables:
            table.create(db.engine)
        for key, status, owner in ((11, 2, None), (12, 4, 7), (13, 4, 8), (14, 1, None)):
            corpus = Corpus(id=key, title_cn=f'材料{key}', contributor=owner or 7, status=status,
                            owner_id=owner, attachment=json.dumps([{'file_path': f'{key}.pdf'}]),
                            is_deleted=False, is_vectorized=False)
            db.session.add(corpus)
        db.session.commit()
        self.client = self.app.test_client()
        self.extractor = mock.patch('app.utils.literature_metadata.extract_personal_pdf_metadata',
            return_value={'title_cn': '测试论文', 'title_en': 'Test paper',
                          'keyword_cn': ['空间检索'], 'keyword_en': ['spatial search'],
                          'abstract_cn': '测试摘要'})
        self.model = self.extractor.start()

        @contextmanager
        def source(_path):
            """提供解析替身的本地路径，不读取真实文件或请求文件服务。"""
            yield '/tmp/test-personal.pdf'

        self.source = mock.patch('app.services.personal_metadata.personal_source_path', side_effect=source)
        self.source.start()

    def tearDown(self):
        """释放测试数据库并恢复方言字段，避免影响其它模块的测试。"""
        self.extractor.stop()
        self.source.stop()
        db.session.remove()
        db.engine.dispose()
        self.context.pop()
        for column, kind in self.types:
            column.type = kind

    def upload(self, content=b'%PDF-1.4\nmock', filename='本地论文.pdf', **values):
        """提交新的文件流，允许用例附带伪造字段验证服务端所有权赋值。"""
        return self.client.post('/literature/files', data={
            'file': (io.BytesIO(content), filename), **values,
        }, content_type='multipart/form-data')

    def test_upload_sets_owner_and_queues_atomically(self):
        """客户端不能伪造所有者或审核状态，文件服务接收可追踪且不冲突的名称。"""
        with mock.patch.object(literature_files, 'upload_file', return_value=(
            [{'file_path': 'personal-7/random.pdf'}], 200,
        )) as uploader:
            response = self.upload(owner_id='8', status='2')
        self.assertEqual(response.status_code, 202, response.json)
        data = response.json['data']
        self.assertEqual((data['status'], data['owner_id'], data['origin']), (4, 7, 'PERSONAL'))
        self.assertEqual(data['index']['status'], 'PENDING')
        row = db.session.get(Corpus, data['id'])
        self.assertEqual(json.loads(row.attachment)[0]['original_filename'], '本地论文.pdf')
        self.assertIsNotNone(db.session.get(CorpusIndexJob, data['id']))
        self.assertEqual(uploader.call_args.args[2], 'personal-7')

    def test_invalid_or_large_file_never_reaches_storage(self):
        """拒绝伪 PDF、错误扩展名、缺失和超限内容，不登记文献。"""
        with mock.patch.object(literature_files, 'upload_file') as uploader:
            self.assertEqual(self.upload(b'not pdf').status_code, 400)
            self.assertEqual(self.upload(filename='paper.txt').status_code, 400)
            self.assertEqual(self.client.post('/literature/files').status_code, 400)
            with mock.patch.object(literature_files, 'MAX_PDF_BYTES', 8):
                self.assertEqual(self.upload().status_code, 413)
            uploader.assert_not_called()
        self.assertEqual(Corpus.query.count(), 4)

    def test_backend_extracts_after_commit_before_worker_can_claim(self):
        """真实上传请求在 Backend 提取，已落库的准备任务不能被 Worker 抢走。"""
        def extracting(_path):
            """模拟同步模型调用期间，检查持久化状态与 Worker 领取边界。"""
            row = Corpus.query.filter(Corpus.id > 14).one()
            job = db.session.get(CorpusIndexJob, row.id)
            self.assertEqual(job.status, 'EXTRACTING')
            with mock.patch.object(corpus_index_worker, 'app', self.app):
                self.assertIsNone(corpus_index_worker.claim_next_job())
            return {'title_cn': '空间关键词查询', 'keyword_cn': ['空间检索'],
                    'keyword_en': ['spatial search'], 'abstract_cn': '摘要'}

        self.model.side_effect = extracting
        with mock.patch.object(literature_files, 'upload_file', return_value=(
            [{'file_path': 'personal-7/random.pdf'}], 200,
        )):
            response = self.upload()
        self.assertEqual(response.status_code, 202, response.json)
        row = db.session.get(Corpus, response.json['data']['id'])
        self.assertEqual(row.title_cn, '空间关键词查询')
        self.assertEqual(response.json['data']['index']['status'], 'PENDING')

    def test_failed_backend_extraction_keeps_file_and_retry_extracts_in_backend(self):
        """提取失败返回可重试材料，重复上传无需承担已有文件丢失风险。"""
        self.model.side_effect = RuntimeError('模型不可用')
        with mock.patch.object(literature_files, 'upload_file', return_value=(
            [{'file_path': 'personal-7/random.pdf'}], 200,
        )), mock.patch.object(literature_files, 'delete_file') as cleanup:
            response = self.upload()
        self.assertEqual(response.status_code, 502)
        cleanup.assert_not_called()
        row_id = response.json['data']['id']
        self.assertEqual(db.session.get(CorpusIndexJob, row_id).status, 'FAILED')
        self.model.side_effect = None
        retried = self.client.post(f'/literature/files/{row_id}/retry')
        self.assertEqual(retried.status_code, 202, retried.json)
        self.assertEqual(db.session.get(CorpusIndexJob, row_id).status, 'PENDING')

    def test_active_backend_extraction_is_reused_and_stale_one_can_retry(self):
        """并发重试复用尚未过期的 Backend 请求，崩溃遗留准备任务允许手动恢复。"""
        from datetime import datetime, timedelta
        job = CorpusIndexJob(corpus_id=12, generation=1, attachment_digest='old',
                             status='EXTRACTING', started_at=datetime.utcnow())
        db.session.add(job)
        db.session.commit()
        self.assertEqual(self.client.post('/literature/files/12/retry').status_code, 202)
        self.model.assert_not_called()
        self.assertEqual(job.generation, 1)
        job.started_at = datetime.utcnow() - timedelta(minutes=16)
        db.session.commit()
        self.assertEqual(self.client.post('/literature/files/12/retry').status_code, 202)
        self.model.assert_called_once()
        self.assertEqual(job.generation, 2)

    def test_storage_and_database_failure_do_not_leave_orphan_records(self):
        """上传失败回滚主表，提交失败时补偿删除已上传的随机附件。"""
        with mock.patch.object(literature_files, 'upload_file', return_value=('failed', 500)):
            self.assertEqual(self.upload().status_code, 502)
        self.assertEqual(Corpus.query.count(), 4)
        with mock.patch.object(literature_files, 'upload_file', return_value=([{'file_path': 'new.pdf'}], 200)), \
                mock.patch.object(db.session, 'commit', side_effect=RuntimeError('database unavailable')), \
                mock.patch.object(literature_files, 'delete_file') as cleanup:
            self.assertEqual(self.upload().status_code, 500)
            cleanup.assert_called_once_with(['new.pdf'])
        self.assertEqual(Corpus.query.count(), 4)

    def test_lists_and_detail_isolate_two_owners_and_public_library(self):
        """私有列表、公共候选和统一详情均不泄漏其他用户或待审材料。"""
        self.assertEqual([row['id'] for row in self.client.get('/literature/files').json['data']['items']], [12])
        self.assertEqual([row['id'] for row in self.client.get('/literature/library').json['data']['items']], [11])
        for key in (11, 12):
            self.assertEqual(self.client.get(f'/literature/documents/{key}').status_code, 200)
        for key in (13, 14):
            self.assertEqual(self.client.get(f'/literature/documents/{key}').status_code, 404)
        self.actor = 8
        self.assertEqual(self.client.get('/literature/documents/12').status_code, 404)

    def test_preview_requires_owner_and_exact_attachment(self):
        """原文路径只能来自本人材料的当前附件清单，返回结果禁止公共缓存。"""
        with mock.patch.object(literature_files, 'get_file', side_effect=lambda path: (make_response(b'%PDF'), 200)) as reader:
            self.assertEqual(self.client.get('/literature/documents/13/file?attachment_path=13.pdf').status_code, 404)
            self.assertEqual(self.client.get('/literature/documents/12/file?attachment_path=13.pdf').status_code, 404)
            reader.assert_not_called()
            response = self.client.get('/literature/documents/12/file?attachment_path=12.pdf')
            self.assertEqual(response.status_code, 200)
            self.assertIn('no-store', response.headers['Cache-Control'])

    def test_legacy_read_and_download_reject_personal_files(self):
        """旧管理、公共详情与附件接口不能通过 ID 读取个人文件。"""
        for path in ('/corpus/get/detail?corpus_id=12', '/file/corpus?corpus_id=12&attachment_path=12.pdf',
                     '/public/corpus_detail?corpus_id=12', '/public/file/corpus?corpus_id=12&attachment_path=12.pdf'):
            self.assertIn(self.client.get(path).status_code, (400, 404), path)
        response = self.client.get('/corpus/get?status=4')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json['data']['total'], 0)

    def test_source_snippet_requires_a_readable_attachment(self):
        """旧片段接口不能凭路径读取他人文件或任意服务器文件。"""
        with mock.patch('app.utils.qdrant_utils.get_source_snippet', return_value={'text': '证据'}) as reader:
            for path in ('13.pdf', '/etc/passwd'):
                response = self.client.post('/ai_search/internal/source_snippet', json={'file_path': path})
                self.assertEqual(response.status_code, 404)
            reader.assert_not_called()
            self.assertEqual(self.client.post('/ai_search/internal/source_snippet', json={
                'file_path': '12.pdf', 'char_start': 0, 'char_end': 2,
            }).status_code, 200)
            # 覆盖原有 ASCII 转义附件 JSON，避免新增授权筛选使中文和特殊字符文件名失效。
            path = '个人材料/研究 "结果" 100%_v2.pdf'
            corpus = db.session.get(Corpus, 12)
            corpus.attachment = json.dumps([{'file_path': path}], ensure_ascii=True)
            db.session.commit()
            self.assertEqual(self.client.post('/ai_search/internal/source_snippet', json={
                'file_path': path, 'char_start': 0, 'char_end': 2,
            }).status_code, 200)

    def test_retry_reuses_active_task_and_rejects_other_owner(self):
        """重复点击重试不重置正在执行的任务代次，其他用户不能操作。"""
        first = self.client.post('/literature/files/12/retry')
        self.assertEqual(first.status_code, 202)
        generation = db.session.get(CorpusIndexJob, 12).generation
        self.assertEqual(self.client.post('/literature/files/12/retry').status_code, 202)
        self.assertEqual(db.session.get(CorpusIndexJob, 12).generation, generation)
        self.assertEqual(self.client.post('/literature/files/13/retry').status_code, 404)

    def test_delete_revokes_running_task_before_remote_cleanup(self):
        """清理失败也不能继续读取文件，旧任务版本被撤销且再次请求可继续清理。"""
        corpus = db.session.get(Corpus, 12)
        enqueue_corpus_index(db.session, corpus)
        db.session.commit()
        job = db.session.get(CorpusIndexJob, 12)
        generation = job.generation
        job.status = 'RUNNING'
        db.session.commit()
        with mock.patch.object(literature_files, 'delete_indexes', side_effect=RuntimeError('busy')):
            response = self.client.delete('/literature/files/12')
        self.assertTrue(response.json['data']['cleanup_pending'])
        self.assertEqual(self.client.get('/literature/documents/12').status_code, 404)
        job = db.session.get(CorpusIndexJob, 12)
        self.assertEqual(job.status, 'CANCELLED')
        self.assertGreater(job.generation, generation)
        with mock.patch.object(literature_files, 'delete_indexes'), \
                mock.patch.object(literature_files, 'delete_file', return_value=({}, 200)):
            self.assertFalse(self.client.delete('/literature/files/12').json['data']['cleanup_pending'])
        self.assertEqual(self.client.delete('/literature/files/13').status_code, 404)

    def test_public_sql_excludes_private_rows_even_in_counts_and_subqueries(self):
        """旧模型生成的宽查询、聚合和多表别名都只能读取公开审核文献。"""
        for sql in ('SELECT id FROM corpus', 'SELECT c.id FROM `corpus` c',
                    'SELECT c.id FROM `corpus`c',
                    'SELECT c.id FROM corpus c,corpus d WHERE c.id=d.id',
                    'SELECT corpus.id FROM corpus WHERE 1=1',
                    'SELECT id FROM (SELECT * FROM corpus) x'):
            self.assertEqual([row[0] for row in db.session.execute(text(public_corpus_sql(sql)))], [11])
        self.assertEqual(db.session.execute(text(public_corpus_sql('SELECT COUNT(*) FROM corpus'))).scalar(), 1)
        for sql in ('DELETE FROM corpus', 'SELECT * FROM test.corpus', 'SELECT 1; SELECT * FROM corpus',
                    'SELECT * FROM (corpus)'):
            with self.assertRaises(ValueError):
                public_corpus_sql(sql)

    def test_eligibility_requires_personal_owner_and_never_implies_read_access(self):
        """公共与有主个人文件可索引，无主个人文件、待审或已删除材料不能构建。"""
        self.assertTrue(can_index_corpus(db.session.get(Corpus, 11)))
        self.assertTrue(can_index_corpus(db.session.get(Corpus, 13)))
        self.assertFalse(can_index_corpus(db.session.get(Corpus, 14)))
        personal = db.session.get(Corpus, 12)
        personal.owner_id = None
        self.assertFalse(can_index_corpus(personal))

    def test_owner_migration_is_idempotent_and_preserves_existing_data(self):
        """旧文献的所有者默认为空，重复迁移不增加重复列或索引。"""
        engine = create_engine('sqlite://')
        with engine.begin() as connection:
            connection.execute(text('CREATE TABLE corpus (id BIGINT PRIMARY KEY, status INTEGER)'))
            connection.execute(text('INSERT INTO corpus VALUES (11, 2)'))
        migrate_personal_owner(engine)
        migrate_personal_owner(engine)
        with engine.connect() as connection:
            self.assertEqual(tuple(connection.execute(text('SELECT status, owner_id FROM corpus')).one()), (2, None))
        self.assertEqual([item['name'] for item in inspect(engine).get_indexes('corpus')], ['ix_corpus_owner_id'])
        engine.dispose()
