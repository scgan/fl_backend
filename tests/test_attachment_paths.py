"""附件下载、解析与引用保留文件名标点；所有 HTTP 和 PDF/OCR 调用均离线。"""
import io
import os
import tempfile
import unittest
from unittest import mock

import fitz
from docx import Document
from flask import Flask

from app.utils import qdrant_utils as q


class AttachmentPathTests(unittest.TestCase):
    """验证真实附件名的标点、编码、扩展名及本地和下载解析的一致性。"""
    @staticmethod
    def sample_documents():
        """生成 PDF、DOCX 及多编码 TXT 字节与预期段落，供同一组解析断言复用。"""
        with fitz.open() as pdf:
            page = pdf.new_page()
            page.insert_text((40, 50), 'Ordinary PDF evidence')
            pdf_content = pdf.tobytes()
        docx = Document()
        docx.add_paragraph('Ordinary DOCX evidence')
        docx.add_paragraph('Second paragraph')
        buffer = io.BytesIO()
        docx.save(buffer)
        return (
            ('paper.pdf', pdf_content, ['Ordinary PDF evidence\n']),
            ('论文.v2.PDF', pdf_content, ['Ordinary PDF evidence\n']),
            ('paper.docx', buffer.getvalue(), ['Ordinary DOCX evidence\n', 'Second paragraph\n']),
            ('paper.txt', 'Ordinary TXT evidence\n正文\n'.encode('utf-8'),
             ['Ordinary TXT evidence\n', '正文\n']),
            ('utf16.txt', 'UTF16 evidence\n正文\n'.encode('utf-16'), ['UTF16 evidence\n', '正文\n']),
            ('gb18030.txt', 'GB evidence\n正文\n'.encode('gb18030'), ['GB evidence\n', '正文\n']),
        )

    def test_ordinary_pdf_docx_txt_keep_real_local_and_downloaded_parse_results(self):
        """验证常规格式的本地与下载解析返回一致原文。"""
        app = Flask(__name__)
        app.config['FILE_URL'] = 'https://files.example'
        with tempfile.TemporaryDirectory() as base, app.app_context():
            for filename, content, expected in self.sample_documents():
                with self.subTest(filename=filename):
                    local_path = os.path.join(base, filename)
                    with open(local_path, 'wb') as target:
                        target.write(content)
                    local_blocks = q._read_index_blocks(filename, base, 42)
                    self.assertEqual([b['text'] for b in local_blocks], expected)
                    self.assertEqual(q._read_index_blocks(local_path, base, 42), local_blocks)
                    response = mock.MagicMock()
                    response.__enter__.return_value = response
                    response.iter_content.return_value = [content]
                    remote_path = 'uploaded_files/corpus/' + filename
                    with mock.patch.object(q.requests, 'get', return_value=response) as get:
                        self.assertEqual(q._read_index_blocks(remote_path, base, 42), local_blocks)
                        self.assertEqual(get.call_args.kwargs['params'], {'filename': remote_path})

    def test_plain_paths_preserve_punctuation_and_literal_percent(self):
        """验证普通文件路径中的问号、井号、分号和百分号按文件名原样保留。"""
        for path in (
            'uploaded_files/corpus/Example. Policy affecting higher education?.pdf',
            'uploaded_files/corpus/И. Пирогов ; педагогика.pdf',
            'uploaded_files/corpus/Title. Chapter #2.pdf',
            'uploaded_files/corpus/Literal%3Fname.pdf',
            '/foreign-files/uploaded_files/corpus/Question?.PDF',
            r'C:\files\Title. Chapter #2.pdf',
        ):
            with self.subTest(path=path):
                self.assertEqual(q._attachment_path(path), path)
                self.assertEqual(os.path.splitext(q._attachment_path(path))[1].lower(), '.pdf')

    def test_http_urls_decode_path_and_preserve_semicolons(self):
        """验证仅对完整 HTTP URL 的路径进行解码，并保留真实分号。"""
        url = 'https://files.example/uploaded_files/Question%3F%20%23%202%3B.pdf?token=value#page=3'
        self.assertEqual(q._attachment_path(url), 'uploaded_files/Question? # 2;.pdf')
        self.assertEqual(q._attachment_path('http://files.example/book;part.pdf?v=1'), 'book;part.pdf')

    def test_normalized_reference_keeps_original_filename(self):
        """验证规范化文件引用不会截断或重写实际文件名。"""
        relative = 'uploaded_files/corpus/Title. Question? #2; %3F.pdf'
        self.assertEqual(q._normalize_file_path(relative), relative)
        self.assertEqual(q._normalize_file_path(os.path.join('/foreign-files', relative),
                                              '/foreign-files'), relative)

    def test_download_selects_parser_by_real_extension_and_cleans_temp_file(self):
        """验证下载临时文件保留真实扩展名，并在解析完成后清理。"""
        app = Flask(__name__)
        app.config['FILE_URL'] = 'https://files.example'
        for filename, extension in (
            ('Example. Policy affecting higher education?.pdf', '.pdf'),
            ('Title. Section; subtitle.pdf', '.pdf'),
            ('Title. Chapter #2.PDF', '.PDF'),
            ('Question?.txt', '.txt'),
            ('Notes; #2?.docx', '.docx'),
        ):
            path = 'uploaded_files/corpus/' + filename
            downloaded = []

            def parse(temp_path):
                """检查临时路径存在、后缀与下载字节一致，记录路径供退出后清理断言。"""
                downloaded.append(temp_path)
                self.assertTrue(os.path.isfile(temp_path))
                self.assertEqual(os.path.splitext(temp_path)[1], extension)
                with open(temp_path, 'rb') as source:
                    self.assertEqual(source.read(), b'file bytes')
                return [{'text': 'parsed'}]

            response = mock.MagicMock()
            response.__enter__.return_value = response
            response.iter_content.return_value = [b'file bytes']
            with self.subTest(path=path), tempfile.TemporaryDirectory() as base, \
                    app.app_context(), mock.patch.object(q.requests, 'get', return_value=response) as get, \
                    mock.patch.object(q, '_read_blocks_locally', side_effect=parse):
                self.assertEqual(q._read_index_blocks(path, base, 42), [{'text': 'parsed'}])
                self.assertEqual(get.call_args.kwargs['params'], {'filename': path})
                response.raise_for_status.assert_called_once()
                self.assertFalse(os.path.exists(downloaded[0]))

    def test_download_parse_failure_still_cleans_temp_file(self):
        """验证解析异常时临时下载文件仍然清理。"""
        app = Flask(__name__)
        app.config['FILE_URL'] = 'https://files.example'
        response = mock.MagicMock()
        response.__enter__.return_value = response
        response.iter_content.return_value = [b'invalid pdf']
        with tempfile.TemporaryDirectory() as base, app.app_context(), \
                mock.patch.object(q.requests, 'get', return_value=response), \
                mock.patch.object(q, '_read_blocks_locally', side_effect=ValueError('invalid pdf')) as parse:
            with self.assertRaisesRegex(ValueError, 'invalid pdf'):
                q._read_index_blocks('Title. Question?.pdf', base, 42)
            self.assertFalse(os.path.exists(parse.call_args.args[0]))

    def test_downloaded_pdf_with_question_mark_reaches_real_native_parser(self):
        """验证含问号的 PDF 下载后进入真实原生解析器。"""
        app = Flask(__name__)
        app.config['FILE_URL'] = 'https://files.example'
        with fitz.open() as pdf:
            page = pdf.new_page()
            page.insert_text((40, 50), 'Higher education policy evidence')
            content = pdf.tobytes()
        response = mock.MagicMock()
        response.__enter__.return_value = response
        response.iter_content.return_value = [content]
        with tempfile.TemporaryDirectory() as base, app.app_context(), \
                mock.patch.object(q.requests, 'get', return_value=response):
            blocks = q._read_index_blocks(
                'uploaded_files/corpus/Example. Policy affecting higher education?.pdf', base, 42)
        self.assertEqual([b['text'] for b in blocks], ['Higher education policy evidence\n'])


if __name__ == '__main__':
    unittest.main()
