"""离线验证多语言分块、覆盖范围和重叠区定位，不访问数据库或模型。"""

import unittest
from app.utils import document_chunking as chunking
from app.utils.embedding_tokens import count_tokens



class DocumentChunkingTests(unittest.TestCase):
    """所有语言采用相同 token 预算，分块不得遗漏或伪造原文。"""

    def verify_coverage(self, blocks, **options):
        """逐块检查原文偏移、硬上限、单调推进及全文覆盖。"""
        source = "".join(block.get("text", "") for block in blocks)
        chunks = chunking.blocks_to_chunks(blocks, "source.pdf", **options)
        covered = 0
        previous_start = -1
        for index, chunk in enumerate(chunks):
            meta = chunk["metadata"]
            start, end = meta["char_start"], meta["char_end"]
            self.assertEqual(chunk["text"], source[start:end])
            self.assertLessEqual(count_tokens(chunk["text"]), options.get('max_chunk_size', 768))
            self.assertEqual(chunk['metadata']['token_count'], count_tokens(chunk['text']))
            self.assertNotIn('\ufffd', chunk['text'])
            self.assertLessEqual(start, covered)
            self.assertGreater(start, previous_start)
            self.assertGreater(end, covered)
            self.assertEqual(meta["chunk_id"], f"source.pdf_{index}")
            previous_start, covered = start, end
        self.assertEqual(covered, len(source))
        return chunks

    def test_languages_and_long_unbroken_blocks(self):
        """中、英、阿拉伯、日、俄文以及无标点文本都受同一上限约束。"""
        for phrase in ["教育研究结果。", "Education research. ", "دراسة التعليم؟ ",
                       "教育の研究。", "Исследование образования. ", "无", "😊",
                       "การวิจัยการศึกษา", "शिक्षा का अध्ययन।"]:
            with self.subTest(phrase=phrase):
                self.verify_coverage([{"text": phrase * 700}])

    def test_exact_target_and_overlap(self):
        """没有边界时按 512 token 输入预算切分并重叠约 64 token。"""
        chunks = self.verify_coverage([{"text": "字" * 3000}])
        self.assertEqual(count_tokens(chunks[0]["text"]), 512)
        self.assertEqual(chunks[1]["metadata"]["char_start"], 447)

    def test_sentence_boundary(self):
        """目标附近存在句尾时优先保留完整句子。"""
        chunks = self.verify_coverage([{"text": ("字" * 99 + "。") * 30}])
        self.assertTrue(chunks[0]["text"].endswith("。"))
        start = chunks[1]['metadata']['char_start']
        self.assertLessEqual(count_tokens(chunks[0]['text'][start:]), 96)

    def test_short_tail(self):
        """短尾部能合并时合并，无法合并时保留原文覆盖。"""
        for length in [901, 1000, 1201, 1690, 1801, 2500]:
            chunks = self.verify_coverage([{"text": "字" * length}])
            self.assertGreaterEqual(count_tokens(chunks[-1]["text"]), 128)
        self.assertEqual(len(chunking.blocks_to_chunks([{"text": "字" * 590}], "a")), 1)

    def test_tokens_are_not_characters(self):
        """验证分块预算按本地 token 估算而非字符数计算。"""
        chunks = self.verify_coverage([{'text': 'Education research evidence. ' * 900}])
        self.assertGreater(len(chunks[0]['text']), 768)
        self.assertLessEqual(abs(count_tokens(chunks[0]['text']) - 512), 8)

    def test_paragraph_without_sentence_punctuation(self):
        """验证没有句末标点的长段落仍满足硬上限且覆盖原文。"""
        paragraph = '无句号内容' * 70 + '\n\n'
        chunks = self.verify_coverage([{'text': paragraph * 20}])
        self.assertTrue(chunks[0]['text'].endswith('\n\n'))

    def test_sentence_overlap(self):
        """验证重叠窗口保持原文顺序并尽量保留自然句子边界。"""
        sentence = '研究表明，教育政策需要结合不同语言的证据进行评估。'
        source = sentence * 250
        chunks = self.verify_coverage([{'text': source}])
        for prev, current in zip(chunks, chunks[1:]):
            self.assertTrue(prev['text'].endswith('。'))
            start = current['metadata']['char_start']
            self.assertEqual(source[start - 1], '。')
            self.assertTrue(32 <= count_tokens(source[start:prev['metadata']['char_end']]) <= 96)

    def test_overlap_bbox_and_page_offset(self):
        """重叠区携带原行坐标，页偏移只应用一次且不污染输入。"""
        blocks = [{"text": "字" * 450, "page_idx": i, "bbox": [0, i, 10, i + 1],
                   "block_id": i} for i in range(6)]
        chunks = self.verify_coverage(blocks, page_offset=200)
        for chunk in chunks:
            meta = chunk["metadata"]
            expected = [i for i in range(6)
                        if i * 450 < meta["char_end"] and (i + 1) * 450 > meta["char_start"]]
            self.assertEqual([b["block_id"] for b in meta["bbox_blocks"]], expected)
            self.assertEqual(meta["page_start"], 200 + min(expected))
            self.assertEqual(meta["page_end"], 200 + max(expected))
        self.assertEqual(blocks[0]["page_idx"], 0)

    def test_no_pages_and_empty_input(self):
        """空文档不生成空块，纯文本不虚构 PDF 页码。"""
        for blocks in [[], [{"text": ""}], [{"text": " \n\t"}]]:
            self.assertEqual(chunking.blocks_to_chunks(blocks, "a"), [])
        chunks = self.verify_coverage([{"text": "short"}])
        self.assertIsNone(chunks[0]["metadata"]["page_start"])
        self.assertEqual(chunks[0]["metadata"]["bbox_blocks"], [])

    def test_invalid_parameters(self):
        """非法配置立即报错，避免无限循环。"""
        for options in [{"chunk_overlap": 512}, {"max_chunk_size": 500},
                        {"min_chunk_size": 0}]:
            with self.assertRaises(ValueError):
                chunking.blocks_to_chunks([], "a", **options)


if __name__ == "__main__":
    unittest.main()
