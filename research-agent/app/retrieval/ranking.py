"""实现向量与 BM25 候选的 RRF 融合、可解释重排和词项覆盖计算。

Qdrant 相似度与 OpenSearch BM25 得分不在同一尺度，不能直接相加或平均。本模块
先只使用各分支名次进行 Reciprocal Rank Fusion，再用归一化 RRF、向量信号、
词项覆盖和双路一致性生成 0 到 1 的解释性分值。该分值用于候选排序，最终能否
成为 Evidence 仍由服务层的绝对门槛判断。
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass


@dataclass(frozen=True)
class RankedHit:
    """表示单个外部召回分支返回的真实切块 ID 与原始得分。

    ``score`` 只允许在产生它的分支内部比较；向量相似度和 BM25 分数语义不同，
    融合时必须转化为名次，不能把该字段跨分支直接计算。
    """

    # 外部索引命中的真实切块主键。
    chunk_id: uuid.UUID
    # 当前检索分支的原始分数，仅在分支内比较。
    score: float


@dataclass(frozen=True)
class FusedHit:
    """表示 RRF 去重后仍保留双路来源信号的候选。

    除统一的 ``rrf_score`` 外，结构继续保存各分支原始分数和 1-based 名次，供
    后续重排、绝对 Evidence 门槛和审计使用。某分支未命中时对应字段保持为空，
    而不是伪造零名次。
    """

    # 去重后的真实切块主键。
    chunk_id: uuid.UUID
    # 只依赖分支名次计算的 RRF 分数。
    rrf_score: float
    # Qdrant 余弦相似度；未命中向量分支时为空。
    vector_score: float | None
    # 向量分支名次，起始值为 1。
    vector_rank: int | None
    # OpenSearch BM25 原始分数；未命中词法分支时为空。
    lexical_score: float | None
    # 词法分支名次，起始值为 1。
    lexical_rank: int | None


def reciprocal_rank_fusion(
    vector_hits: list[RankedHit], lexical_hits: list[RankedHit], *, k: int
) -> list[FusedHit]:
    """按名次而非不可比的原始分值融合两路结果。

    每个分支中第 ``rank`` 名贡献 ``1 / (k + rank)``；同一 Chunk 在两路命中时
    累加贡献，从而奖励跨方法一致性，同时避免某个引擎分值范围支配结果。最终用
    Chunk UUID 作为同分稳定排序键，保证测试和重复请求可复现。

    Args:
        vector_hits: 向量相似度降序结果。
        lexical_hits: BM25 得分降序结果。
        k: RRF 平滑常数。

    Returns:
        以 RRF 得分降序排列的去重候选。
    """
    # 1. 按分支内 1-based 名次累计每个 Chunk 的 RRF 贡献
    # 同一 Chunk 在两路命中时复用一条记录并累加分数，同时保留各分支原始分和名次，
    # 供后续可解释重排与绝对质量门槛使用。
    records: dict[uuid.UUID, dict[str, float | int | None]] = {}
    for branch, hits in (("vector", vector_hits), ("lexical", lexical_hits)):
        for rank, hit in enumerate(hits, start=1):
            item = records.setdefault(
                hit.chunk_id,
                {
                    "rrf": 0.0,
                    "vector_score": None,
                    "vector_rank": None,
                    "lexical_score": None,
                    "lexical_rank": None,
                },
            )
            item["rrf"] = float(item["rrf"] or 0.0) + 1.0 / (k + rank)
            item[f"{branch}_score"] = hit.score
            item[f"{branch}_rank"] = rank
    # 2. 将中间字典转换为强类型候选，不对不同引擎原始分做归一化或混算
    fused = [
        FusedHit(
            chunk_id=chunk_id,
            rrf_score=float(item["rrf"] or 0.0),
            vector_score=_optional_float(item["vector_score"]),
            vector_rank=_optional_int(item["vector_rank"]),
            lexical_score=_optional_float(item["lexical_score"]),
            lexical_rank=_optional_int(item["lexical_rank"]),
        )
        for chunk_id, item in records.items()
    ]
    # 3. 按融合分降序、UUID 升序稳定排序，保证重复请求和测试结果可复现
    return sorted(fused, key=lambda item: (-item.rrf_score, str(item.chunk_id)))


def rerank_evidence(
    query: str,
    hits: list[FusedHit],
    contents: dict[uuid.UUID, str],
) -> list[tuple[FusedHit, float]]:
    """用查询覆盖率、双路一致性和语义相似度重排候选。

    此实现不声称是神经 reranker；它是可解释的确定性基线，避免引入未经配置的
    在线模型。跨语言能力主要来自多语种 Embedding，词项覆盖只用于补充字面支持。
    权重固定为 RRF 0.45、向量 0.25、覆盖率 0.20、双路一致性 0.10，且输出截断
    到 0..1。这里仅排序，不决定证据是否充分，避免相对排名把低质量候选强行抬成
    Evidence。

    Args:
        query: 规范化后的检索表达式。
        hits: RRF 候选。
        contents: 候选切块正文映射。

    Returns:
        候选及 0 到 1 的统一相关性分值，按分值降序。
    """
    if not hits:
        return []
    max_rrf = max(item.rrf_score for item in hits) or 1.0
    ranked: list[tuple[FusedHit, float]] = []
    for hit in hits:
        content = contents.get(hit.chunk_id, "").casefold()
        coverage = term_coverage(query, content)
        branch_agreement = 1.0 if hit.vector_rank and hit.lexical_rank else 0.0
        vector_signal = max(0.0, min(1.0, hit.vector_score or 0.0))
        score = (
            0.45 * (hit.rrf_score / max_rrf)
            + 0.25 * vector_signal
            + 0.20 * coverage
            + 0.10 * branch_agreement
        )
        ranked.append((hit, round(min(score, 1.0), 6)))
    return sorted(ranked, key=lambda item: (-item[1], str(item[0].chunk_id)))


def term_coverage(query: str, content: str) -> float:
    """计算查询词项在候选正文中的覆盖率。

    覆盖率是简单、可解释的字面证据，不替代语义相似度。英文按长度至少三字符的
    token，中文按二元片段计算，以兼顾未分词文本并降低单字噪声。

    Args:
        query: 规范化或双语扩展查询。
        content: 候选切块正文。

    Returns:
        0 到 1 的词项覆盖率。
    """
    terms = _terms(query)
    if not terms:
        return 0.0
    folded = content.casefold()
    return sum(1 for term in terms if term in folded) / len(terms)


def _terms(query: str) -> set[str]:
    """提取可用于覆盖率计算的英文词和中文二字片段。

    返回集合而不是词频列表，防止双语扩展中重复词项提高覆盖率；短英文词和中文
    单字不参与计算，以减少冠词、缩写或高频虚词造成的误判。

    Args:
        query: 规范化查询。

    Returns:
        去除过短噪声后的词项集合。
    """
    english = {item.casefold() for item in re.findall(r"[A-Za-z][A-Za-z-]{2,}", query)}
    chinese_runs = re.findall(r"[\u3400-\u9fff]+", query)
    chinese = {
        run[index : index + 2] for run in chinese_runs for index in range(max(0, len(run) - 1))
    }
    return english | chinese


def _optional_float(value: float | int | None) -> float | None:
    """把内部联合类型安全转换为可选浮点数。

    Args:
        value: 内部记录值。

    Returns:
        浮点数或空值。
    """
    return float(value) if value is not None else None


def _optional_int(value: float | int | None) -> int | None:
    """把内部联合类型安全转换为可选整数。

    Args:
        value: 内部记录值。

    Returns:
        整数或空值。
    """
    return int(value) if value is not None else None
