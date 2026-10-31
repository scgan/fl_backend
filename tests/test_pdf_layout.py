"""真实 PDF 回归：原生文字双栏、备用解析、跨栏标题、表格及定位。"""
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import fitz

from app.utils import qdrant_utils as q
from app.utils.pdf_layout import order_positioned_lines, words_to_lines


class PDFLayoutTests(unittest.TestCase):
    """用真实生成的 PDF 验证双栏、跨栏标题、表格和引用切片的阅读顺序。"""
    def setUp(self):
        """建立本例临时目录与 PDF 路径，清理动作注册到测试结束。"""
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.path = Path(self.folder.name) / 'columns.pdf'

    def make_pdf(self, mid_heading=False, table=False, single=False):
        """生成可选跨栏标题、有线表格或单栏正文的测试页面。"""
        with fitz.open() as pdf:
            page = pdf.new_page(width=600, height=800)
            page.insert_text((40, 35), 'Full width document heading across the page', fontsize=20)
            for index in range(8):
                y = 90 + index * 18 + (50 if mid_heading and index >= 4 else 0)
                if single:
                    page.insert_text((40, y), f'Single {index} ordinary English sentence that continues on the same line.', fontsize=10)
                else:
                    page.insert_text((40, y), f'Left {index} sentence continues in its own column.', fontsize=10)
                    # 故意按行交替写入 PDF，不能依赖对象写入顺序。
                    page.insert_text((330, y), f'Right {index} different column of evidence.', fontsize=9)
            if mid_heading:
                page.insert_text((40, 178), 'Another full width heading across both columns', fontsize=18)
            if table:
                for y in (280, 300, 320, 340, 360):
                    page.draw_line((40, y), (555, y))
                for x in (40, 300, 555):
                    page.draw_line((x, 280), (x, 360))
                for index in range(4):
                    page.insert_text((50, 294 + index * 20), f'CellL{index}', fontsize=10)
                    page.insert_text((320, 294 + index * 20), f'CellR{index}', fontsize=10)
            page.insert_text((280, 770), 'Page 1', fontsize=10)
            pdf.save(self.path)

    def body_labels(self, blocks):
        """只提取正文标签，避免页眉页脚影响阅读顺序断言。"""
        return [' '.join(block['text'].split()[:2]) for block in blocks
                if block['text'].startswith(('Left ', 'Right ', 'Single '))]

    def test_native_columns_read_left_then_right_and_keep_real_line_boxes(self):
        """验证原生双栏先左后右读取，并保留实际行框。"""
        self.make_pdf()
        blocks = q._read_blocks_locally(str(self.path))
        self.assertEqual(self.body_labels(blocks), [f'Left {i}' for i in range(8)] + [f'Right {i}' for i in range(8)])
        self.assertTrue(blocks[0]['text'].startswith('Full width'))
        self.assertEqual(blocks[-1]['text'], 'Page 1\n')
        for block in blocks:
            self.assertEqual(block['page_idx'], 0)
            if block['text'].startswith(('Left ', 'Right ')):
                self.assertLess(block['bbox'][2] - block['bbox'][0], 280)
                self.assertFalse('Left ' in block['text'] and 'Right ' in block['text'])
        self.assertEqual([b['block_id'] for b in blocks], list(range(len(blocks))))
        source = ''.join(b['text'] for b in blocks)
        for chunk in q._blocks_to_chunks(blocks, self.path.name, chunk_size=64, max_chunk_size=96,
                                          chunk_overlap=8, min_chunk_size=16):
            meta = chunk['metadata']
            self.assertEqual(chunk['text'], source[meta['char_start']:meta['char_end']])
            self.assertTrue(all(b['page_idx'] == 0 for b in meta['bbox_blocks']))

    def test_pymupdf_fallback_also_orders_interleaved_pdf_objects_by_column(self):
        """验证备用解析器也能从交错 PDF 对象中恢复分栏顺序。"""
        self.make_pdf()
        with mock.patch.object(q.pdfplumber, 'open', side_effect=ValueError):
            blocks = q._read_blocks_locally(str(self.path))
        self.assertEqual(self.body_labels(blocks), [f'Left {i}' for i in range(8)] + [f'Right {i}' for i in range(8)])

    def test_midpage_spanning_heading_splits_reading_regions(self):
        """验证中部跨栏标题形成独立区域，前后双栏分别排序。"""
        self.make_pdf(mid_heading=True)
        blocks = q._read_blocks_locally(str(self.path))
        self.assertEqual(self.body_labels(blocks), [f'{side} {i}' for start in (0, 4)
                         for side in ('Left', 'Right') for i in range(start, start + 4)])
        heading = next(i for i, b in enumerate(blocks) if b['text'].startswith('Another full'))
        self.assertTrue(blocks[heading - 1]['text'].startswith('Right 3'))
        self.assertTrue(blocks[heading + 1]['text'].startswith('Left 4'))

    def test_single_column_is_not_split(self):
        """验证普通单栏正文不被误判为双栏。"""
        self.make_pdf(single=True)
        self.assertEqual(self.body_labels(q._read_blocks_locally(str(self.path))), [f'Single {i}' for i in range(8)])

    def test_ruled_table_remains_in_row_order_after_double_column_prose(self):
        """验证双栏正文之后的有线表格保持逐行顺序。"""
        self.make_pdf(table=True)
        for reader in (q._read_blocks_locally, q._read_pdf_blocks_with_pymupdf):
            blocks = reader(str(self.path))
            rows = [b['text'].strip() for b in blocks if b['text'].startswith('CellL')]
            self.assertEqual(rows, [f'CellL{i} CellR{i}' for i in range(4)])

    def test_sparse_aligned_words_do_not_turn_into_false_columns(self):
        """验证少量对齐词不会形成虚假的栏间隙。"""
        words = [{'text': value, 'x0': x, 'x1': x + 20, 'top': y, 'bottom': y + 10}
                 for y in range(0, 200, 20) for x, value in ((10, 'Label'), (200, 'Value'))]
        self.assertEqual([line['text'] for line in words_to_lines(words)], ['Label Value'] * 10)

    def test_positioned_ocr_lines_reorder_without_changing_text_or_coordinates(self):
        """验证 OCR 行只调整顺序，不改变正文和原坐标。"""
        lines = [(f'{side} {i}', [x, i * 20, x + 180, i * 20 + 10], 'line')
                 for i in range(6) for side, x in (('Left', 20), ('Right', 300))]
        ordered = order_positioned_lines(lines[::-1])
        self.assertEqual([text for text, _, _ in ordered], [f'{side} {i}' for side in ('Left', 'Right') for i in range(6)])
        self.assertCountEqual(ordered, lines)
        fallback = [('Page text with no line boxes', [0, 0, 600, 800], 'page')]
        self.assertEqual(order_positioned_lines(fallback), fallback)

    def test_source_snippet_uses_the_same_native_order_and_character_offsets(self):
        """验证引用正文与入库阅读顺序一致，字符偏移可复现切片。"""
        self.make_pdf()
        source = ''.join(b['text'] for b in q._read_blocks_locally(str(self.path)))
        start = source.index('Right 0')
        with mock.patch('app.utils.utils.get_file_content', return_value=(self.path.read_bytes(), 'application/pdf')):
            snippet = q.get_source_snippet('columns.pdf', start, start + 30)
        self.assertTrue(snippet['full_text_available'])
        self.assertEqual(snippet['text'], source[start:start + 30])


if __name__ == '__main__':
    unittest.main()
