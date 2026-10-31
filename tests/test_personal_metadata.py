"""离线验证个人文件题录提取、持久化复用及任务撤销竞态，模型与文件读取使用替身。"""

import json
import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest import mock

from flask import Flask
from sqlalchemy import BigInteger, Integer
from sqlalchemy.dialects.mysql import BIT, YEAR

from app.exts import db
from app.indexing.jobs import enqueue_corpus_index
from app.services.personal_metadata import (
    extract_and_queue_personal_metadata, normalize_metadata, prepare_personal_index,
    fail_personal_metadata,
)
from app.models import Corpus, CorpusIndexJob
from app.utils import literature_metadata
from app.workers import corpus_index_worker


class PersonalMetadataTests(unittest.TestCase):
    """在隔离数据库中验证模型结果只写回当前个人任务，不影响公共题录。"""

    def setUp(self):
        """建立个人材料、Backend 准备任务及可控的模型结果。"""
        self.app = Flask(__name__)
        self.app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite://'
        db.init_app(self.app)
        self.context = self.app.app_context()
        self.context.push()
        tables = (Corpus.__table__, CorpusIndexJob.__table__)
        self.types = [(col, col.type) for table in tables for col in table.columns
                      if isinstance(col.type, (BIT, YEAR, BigInteger))]
        for column, kind in self.types:
            column.type = kind.with_variant(Integer(), 'sqlite')
        for table in tables:
            table.create(db.engine)
        self.corpus = Corpus(id=42, title_cn='含备注的文件名', status=4, owner_id=7,
                             contributor=7, is_deleted=False, is_vectorized=False,
                             attachment=json.dumps([{'file_path': 'personal-7/paper.pdf',
                                                     'original_filename': '含备注的文件名.pdf'}]))
        db.session.add(self.corpus)
        enqueue_corpus_index(db.session, self.corpus)
        self.job = db.session.get(CorpusIndexJob, 42)
        self.job.status = 'EXTRACTING'
        db.session.commit()
        self.snapshot = (42, self.job.attachment_digest, self.job.generation)
        self.data = {'title_cn': '空间关键词查询', 'title_en': 'Spatial keyword queries',
                     'keyword_cn': ['空间查询', '空间查询', '关键词检索'],
                     'keyword_en': ['spatial keyword query'], 'abstract_cn': '研究空间与文本联合检索。',
                     'publish_time': '2014', 'owner_id': 99, 'status': 2}

        @contextmanager
        def source(path):
            """只接受当前附件清单中的真实路径，提供隔离测试路径。"""
            self.assertEqual(path, 'personal-7/paper.pdf')
            yield '/tmp/test.pdf'

        self.source = mock.patch('app.services.personal_metadata.personal_source_path', side_effect=source)
        self.extract = mock.patch.object(literature_metadata, 'extract_personal_pdf_metadata',
                                         return_value=self.data)
        self.source.start()
        self.extractor = self.extract.start()

    def tearDown(self):
        """清理数据库与替身，恢复模型字段的原方言类型。"""
        self.extract.stop()
        self.source.stop()
        db.session.remove()
        db.engine.dispose()
        self.context.pop()
        for column, kind in self.types:
            column.type = kind

    def test_metadata_is_saved_without_changing_identity_attachment_or_job(self):
        """只写题录白名单；原始文件名、私有归属和任务指纹保持有效。"""
        self.assertTrue(extract_and_queue_personal_metadata(*self.snapshot))
        self.assertEqual(self.corpus.title_cn, '空间关键词查询')
        self.assertEqual(self.corpus.title_en, 'Spatial keyword queries')
        self.assertEqual(json.loads(self.corpus.keyword_cn), ['空间查询', '关键词检索'])
        self.assertEqual(self.corpus.publish_time, 2014)
        self.assertEqual((self.corpus.owner_id, self.corpus.status), (7, 4))
        self.assertEqual(self.job.status, 'PENDING')
        self.assertEqual(json.loads(self.corpus.attachment)[0]['original_filename'], '含备注的文件名.pdf')
        self.assertFalse(prepare_personal_index(self.corpus))
        db.session.commit()
        self.extractor.assert_called_once()

    def test_public_material_does_not_call_model(self):
        """审核文献已有采集题录，不在索引时重复提取或覆盖。"""
        self.corpus.status, self.corpus.owner_id = 2, None
        db.session.commit()
        self.assertFalse(extract_and_queue_personal_metadata(*self.snapshot))
        self.extractor.assert_not_called()

    def test_worker_indexes_without_calling_metadata_model(self):
        """题录由 Backend 准备完成后，真实 Worker 编排不得调用题录模型。"""
        self.assertTrue(extract_and_queue_personal_metadata(*self.snapshot))
        self.extractor.reset_mock()
        self.job.status = 'RUNNING'
        db.session.commit()

        def build(corpus_id, attachments, **kwargs):
            """替代外部双路写入，确认 Worker 消费的是已保存的题录。"""
            self.assertEqual(db.session.get(Corpus, corpus_id).title_cn, '空间关键词查询')
            job = db.session.get(CorpusIndexJob, corpus_id)
            job.vector_status = job.lexical_status = 'SUCCESS'
            db.session.commit()
            return {'indexed': 1, 'total_chunks': 2}

        with mock.patch.object(corpus_index_worker, 'app', self.app), \
                mock.patch.object(corpus_index_worker, 'index_corpus_chunks', side_effect=build), \
                mock.patch('app.indexing.service.delete_indexes'):
            corpus_index_worker.process_job(*self.snapshot)
        db.session.expire_all()
        self.assertEqual(self.job.status, 'SUCCESS')
        self.assertTrue(self.corpus.is_vectorized)
        self.extractor.assert_not_called()

    def test_backend_model_failure_preserves_file_for_manual_retry(self):
        """Backend 提取失败保留原件和失败任务，不让 Worker 领取缺题录文件。"""
        self.extractor.side_effect = RuntimeError('个人文件题录提取失败')
        with self.assertRaises(RuntimeError):
            extract_and_queue_personal_metadata(*self.snapshot)
        fail_personal_metadata(*self.snapshot, 7)
        self.assertEqual(self.job.status, 'FAILED')
        self.assertFalse(self.corpus.is_vectorized)
        self.assertTrue(self.corpus.attachment)

    def test_failed_or_empty_model_result_cannot_publish_metadata(self):
        """模型错误或缺少关键词必须失败，不能缓存为完整材料或进入索引。"""
        self.extractor.side_effect = RuntimeError('模型不可用')
        with self.assertRaises(RuntimeError):
            extract_and_queue_personal_metadata(*self.snapshot)
        self.assertIsNone(self.corpus.keyword_cn)
        self.extractor.side_effect = None
        self.extractor.return_value = {'title_cn': '只有标题'}
        with self.assertRaisesRegex(ValueError, '关键词'):
            extract_and_queue_personal_metadata(*self.snapshot)
        self.assertEqual(self.corpus.title_cn, '含备注的文件名')

    def test_stale_generation_deletion_owner_or_attachment_cannot_write_back(self):
        """模型等待期间四种失效均撤销回填，旧任务不覆盖更新后的用户材料。"""
        for change in ('generation', 'deleted', 'owner', 'attachment'):
            with self.subTest(change=change):
                self.job.generation = self.snapshot[2]
                self.corpus.is_deleted, self.corpus.owner_id = False, 7
                self.corpus.attachment = json.dumps([{'file_path': 'personal-7/paper.pdf',
                                                     'original_filename': '含备注的文件名.pdf'}])
                db.session.commit()

                def replace_during_call(_path):
                    """模拟模型请求期间的独立业务更新。"""
                    if change == 'generation':
                        self.job.generation += 1
                    elif change == 'deleted':
                        self.corpus.is_deleted = True
                    elif change == 'owner':
                        self.corpus.owner_id = 8
                    else:
                        self.corpus.attachment = '[{"file_path":"new.pdf"}]'
                    db.session.commit()
                    return self.data

                self.extractor.side_effect = replace_during_call
                self.assertFalse(extract_and_queue_personal_metadata(*self.snapshot))
                self.assertIsNone(self.corpus.keyword_cn)

    def test_keywords_fit_database_without_breaking_json(self):
        """超长关键词按完整词项截取，年份无法确认时不写无效 YEAR。"""
        values = normalize_metadata({**self.data, 'keyword_cn': ['词' * 80 + str(n) for n in range(30)],
                                     'publish_time': '未知'})
        self.assertLessEqual(len(values['keyword_cn']), 510)
        self.assertTrue(json.loads(values['keyword_cn']))
        self.assertIsNone(values['publish_time'])

    def test_existing_extraction_rejects_non_json_and_uses_long_fallback(self):
        """复用原采集提取时验证响应为对象；扫描 PDF 仍走既有整文件提取。"""
        client = mock.MagicMock()
        client.chat.completions.create.return_value = SimpleNamespace(choices=[
            SimpleNamespace(message=SimpleNamespace(content='模型输出不是 JSON'))])
        page = mock.Mock()
        page.extract_text.return_value = '论文标题与摘要'
        pdf = mock.MagicMock()
        pdf.__enter__.return_value.pages = [page]
        with mock.patch.object(literature_metadata.pdfplumber, 'open', return_value=pdf):
            self.assertFalse(literature_metadata.extract_pdf_info_turbo(client, 'test.pdf')['success'])
        page.extract_text.return_value = ''
        with mock.patch.object(literature_metadata.pdfplumber, 'open', return_value=pdf), \
                mock.patch.object(literature_metadata, 'extract_pdf_info_long',
                                  return_value={'success': True, 'extracted_data': self.data}) as fallback:
            self.assertTrue(literature_metadata.extract_pdf_info_turbo(client, 'test.pdf')['success'])
            fallback.assert_called_once()
