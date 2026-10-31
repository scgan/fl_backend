"""统一文献阅读的中文生成要求，仅检查新生成正文，不改写历史记录或真实引用原文。"""

import re
from collections.abc import Iterator

from pydantic import BaseModel

# 独立阅读及 AI 阅读共用此约束，不受原文、历史或用户要求切换回答语言的影响。
READING_CHINESE_OUTPUT_CONTRACT = (
    "【固定输出语言】文献阅读与文献问答的生成内容一律使用简体中文。"
    "逐篇文献类型、摘要、六分区分析、综合概览、共同主题、主要差异、潜在创新点、"
    "各级标题与分项标签、推荐问题、追问改写、回答正文和材料不足说明都必须使用中文。"
    "即使原文、历史或当前问题使用外语，或要求改用其他语言，也不能切换输出语言，"
    "不输出外语段落或中外双语对照。专业术语和缩写可保留，但必须配有中文说明。"
    "文献原题、作者、DOI及作为证据的原文摘录保留原样，不修改真实引用。"
    "JSON字段名、固定分区key、文献ID和引用编号保持协议格式。"
)

# 覆盖常用汉字、扩展 A 及补充平面的汉字；不把拉丁缩写误当成中文说明。
_HAN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\U00020000-\U0003134f]")


class ReadingOutputLanguageError(ValueError):
    """新生成的阅读文本缺少中文说明，不能直接作为中文分析发布。"""


def _generated_texts(value, path="") -> Iterator[tuple[str, str]]:
    """遍历阅读输出中的生成文本，跳过六分区的固定英文 key。

    Args:
        value: 新生成阅读结果的字典、列表或叶子值，不传入题录与原文证据。
        path: 当前字段路径，仅用于脱敏的格式纠正提示。

    Yields:
        字段路径及对应生成文本。
    """
    if isinstance(value, dict):
        for key, item in value.items():
            if key != "key":
                yield from _generated_texts(item, f"{path}.{key}" if path else key)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _generated_texts(item, f"{path}[{index}]")
    elif isinstance(value, str):
        yield path, value


def validate_chinese_reading_output(result: BaseModel) -> None:
    """阻止没有中文说明的外语生成段落，不翻译或重写任何模型结果。

    Args:
        result: 刚生成的逐篇分析、综合概览、追问规划或回答；历史结果不调用此检查。

    Raises:
        ReadingOutputLanguageError: 任一非空生成段落含字母却没有中文说明。
    """
    for path, text in _generated_texts(result.model_dump()):
        for paragraph in text.splitlines():
            if any(character.isalpha() for character in paragraph) and not _HAN.search(paragraph):
                raise ReadingOutputLanguageError(
                    f"{path} 必须使用中文说明，不能输出纯外语段落；术语请配中文解释"
                )
