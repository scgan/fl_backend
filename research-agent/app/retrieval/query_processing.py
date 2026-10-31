"""实现可审计的查询规范化、语言识别与教育研究术语双语扩展。

查询处理刻意不调用 LLM 改写，避免不可复现的改写引入查询中不存在的事实或意图。
这里的规则只移除常见问句外壳，并依据显式词表补充中英文等价表达；跨语言语义
召回仍主要依赖多语种 Embedding。处理结果同时保留原查询、规范查询和两种语言
表达式，便于日志、测试和后续 AgentState 审计。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# 这是可审计的教育研究领域词表，不生成事实；跨语言语义召回仍以多语种 embedding 为主。
EDUCATION_TERMS: tuple[tuple[str, str], ...] = (
    ("人工智能", "artificial intelligence"),
    ("生成式人工智能", "generative artificial intelligence"),
    ("教师", "teacher"),
    ("学生", "student"),
    ("课堂", "classroom"),
    ("学习", "learning"),
    ("教学", "teaching"),
    ("教育", "education"),
    ("反馈", "feedback"),
    ("评估", "assessment"),
    ("素养", "literacy"),
    ("高等教育", "higher education"),
    ("职业教育", "vocational education"),
    ("教师发展", "teacher development"),
    ("学业成绩", "academic achievement"),
    ("数据探索", "data exploration"),
    ("区域搜索", "region search"),
    ("最佳区域", "best region"),
)


@dataclass(frozen=True)
class ProcessedQuery:
    """承载同一次查询的原文、语言判断与双语检索表达式。

    该对象不可变，防止多个召回分支在执行过程中各自修改查询而失去审计基线。
    ``rewritten`` 仅做确定性清理，``chinese`` 和 ``english`` 则分别服务于跨语言
    扩展；它们都不是研究结论，也不会被当作 Evidence。
    """

    # 用户提交的原始查询，除首尾空白外不做改写。
    original: str
    # 自动识别的 zh、en 或 mixed。
    language: str
    # 去除问句外壳后的可审计查询表达式。
    rewritten: str
    # 中文检索分支的扩展表达式。
    chinese: str
    # 英文检索分支的扩展表达式。
    english: str
    # 去除原查询和扩展词重叠后的统一双语检索表达式。
    bilingual: str


class BilingualQueryProcessor:
    """为中英文教育研究查询生成确定性、可测试的双语检索表达式。

    处理器只负责查询理解的最低可用基线：语言识别、问句外壳清理和显式词表扩展。
    它不翻译整句、不推断用户未表达的限定条件，也不访问数据库或外部模型，因此
    同一输入始终得到同一输出，适合在检索审计和回归测试中复现。
    """

    def process(self, query: str) -> ProcessedQuery:
        """规范化查询并根据显式术语表生成中英文扩展。

        原查询先用于语言检测和确定性重写，规范表达式再同时放入两个分支，确保词表
        未覆盖的专有名词不会丢失。只有命中的术语对才会追加对应表达，最后按大小写不敏感
        规则稳定去重；该过程不创造事实，也不会扩大结构化检索过滤范围。

        Args:
            query: 用户原始检索问题。

        Returns:
            保留原文且包含中英文表达式的处理结果。
        """
        original = query.strip()
        language = self._detect_language(original)
        rewritten = self._rewrite(original, language)
        zh_expansions: list[str] = []
        en_expansions: list[str] = []
        folded = rewritten.casefold()
        for chinese, english in EDUCATION_TERMS:
            if chinese in rewritten or re.search(rf"\b{re.escape(english.casefold())}\b", folded):
                zh_expansions.append(chinese)
                en_expansions.append(english)
        return ProcessedQuery(
            original=original,
            language=language,
            rewritten=rewritten,
            chinese=self._merge_nonredundant(rewritten, zh_expansions),
            english=self._merge_nonredundant(rewritten, en_expansions),
            bilingual=self._merge_nonredundant(
                rewritten, [*zh_expansions, *en_expansions]
            ),
        )

    @staticmethod
    def _detect_language(query: str) -> str:
        """根据是否出现 CJK 与拉丁字符识别中文、英文或混合输入。

        当前仅需要为查询扩展选择分支，不承担自然语言学意义上的精确语种识别。
        同时出现 CJK 与拉丁字母时保留 ``mixed``，避免把论文缩写或英文术语误删。

        Args:
            query: 已去除首尾空白的查询。

        Returns:
            ``zh``、``en`` 或 ``mixed``。
        """
        cjk = len(re.findall(r"[\u3400-\u9fff]", query))
        latin = len(re.findall(r"[A-Za-z]", query))
        if cjk and latin:
            return "mixed"
        return "zh" if cjk else "en"

    @staticmethod
    def _rewrite(query: str, language: str) -> str:
        """移除不承载主题的问句外壳，同时保留研究关键词。

        规则只匹配字符串开头的常见请求语，不做同义改写和实体替换；若清理后为空，
        会退回原查询，保证下游永远不会因规则过度匹配而收到空表达式。

        Args:
            query: 原始查询。
            language: 已识别语言。

        Returns:
            适合检索的简洁表达式。
        """
        normalized = re.sub(r"\s+", " ", query).strip(" ?？。")
        if language in {"zh", "mixed"}:
            normalized = re.sub(r"^(请问|请帮我|帮我|查找|检索|有哪些|如何)", "", normalized)
        else:
            normalized = re.sub(
                r"^(please\s+)?(find|search for|show me|what are|how does)\s+",
                "",
                normalized,
                flags=re.IGNORECASE,
            )
        return normalized.strip() or query

    @classmethod
    def _merge_nonredundant(cls, rewritten: str, expansions: list[str]) -> str:
        """保留一次规范查询，并合并不与原文或彼此重叠的扩展词。

        原查询始终作为第一个完整表达式保留。若扩展词已经出现在原查询中则不再
        追加；多个扩展存在包含关系时保留更具体的长表达式，使 BM25 和 Embedding
        不会因重复词项获得非预期权重，同时仍保留词表提供的跨语言语义。

        Args:
            rewritten: 去除问句外壳后的规范查询。
            expansions: 按词表命中顺序产生的同语言或双语扩展表达式。

        Returns:
            使用空格连接且不存在包含式冗余的稳定查询表达式。
        """
        selected: list[str] = []
        for expansion in expansions:
            normalized = expansion.strip()
            if not normalized or cls._contains_expression(rewritten, normalized):
                continue
            if any(cls._contains_expression(item, normalized) for item in selected):
                continue

            # 更具体的后续短语可以取代已选上位词，例如 teacher development
            # 取代 teacher；具体短语仍包含上位词 token，不会降低词法召回。
            selected = [
                item for item in selected if not cls._contains_expression(normalized, item)
            ]
            selected.append(normalized)
        return " ".join([rewritten, *selected])

    @staticmethod
    def _contains_expression(container: str, expression: str) -> bool:
        """判断完整英文短语或连续中文短语是否已包含在另一表达式中。

        英文使用单词边界，避免 ``teacher`` 误命中 ``teachers``；中文没有空格词界，
        因此按连续子串判断。该规则只用于查询扩展去重，不承担自然语言分词。

        Args:
            container: 可能包含目标短语的表达式。
            expression: 待检查的扩展短语。

        Returns:
            目标短语已经完整出现时返回 ``True``。
        """
        if re.search(r"[\u3400-\u9fff]", expression):
            return expression in container
        return bool(
            re.search(
                rf"\b{re.escape(expression.casefold())}\b",
                container.casefold(),
            )
        )
