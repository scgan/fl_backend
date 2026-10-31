"""覆盖跨语言召回、Citation、证据不足和知识库权限隔离验收。"""

from __future__ import annotations

from app.retrieval.query_processing import BilingualQueryProcessor


def test_bilingual_query_processor_removes_redundant_expressions() -> None:
    """验证规范查询、同语言术语和包含式上位词不会被重复拼接。"""
    processor = BilingualQueryProcessor()

    chinese = processor.process("教师如何使用人工智能？")
    assert chinese.chinese == "教师如何使用人工智能"
    assert chinese.english == ("教师如何使用人工智能 artificial intelligence teacher")
    assert chinese.bilingual == chinese.english
    assert chinese.bilingual.count(chinese.rewritten) == 1

    english = processor.process("teacher feedback")
    assert english.chinese == "teacher feedback 教师 反馈"
    assert english.english == "teacher feedback"
    assert english.bilingual == english.chinese
    assert english.bilingual.count(english.rewritten) == 1

    specific = processor.process("生成式人工智能赋能学习设计")
    assert specific.bilingual == (
        "生成式人工智能赋能学习设计 generative artificial intelligence learning"
    )
    assert specific.bilingual.count(specific.rewritten) == 1


class FakeQueryEmbedding:
    """根据测试查询返回可区分的固定向量标记。"""

    def embed_query(self, text: str) -> list[float]:
        """把中查英、英查中和无结果查询编码为不同首值。

        Args:
            text: 已双语扩展的查询。

        Returns:
            测试向量标记。
        """
        if "教师如何使用人工智能" in text:
            return [1.0]
        if "teacher feedback" in text.casefold():
            return [2.0]
        return [0.0]
