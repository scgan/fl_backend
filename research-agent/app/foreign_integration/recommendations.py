"""库内题录推荐：真实双路检索、按文献融合及题录补充，不调用生成模型。"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing

from opensearchpy import OpenSearch
from qdrant_client import QdrantClient, models
from sqlalchemy import or_, select

from app.agents.web_literature import MAX_WEB_RECOMMENDATIONS, extract_doi, same_title
from app.common.bailian_client import embed_texts, embedding_config
from app.foreign_integration.database import session_factory
from app.foreign_integration.retrieval import _verified_candidates
from app.foreign_integration.schema import foreign_corpus, foreign_corpus_index_job, foreign_user
from app.retrieval.ranking import term_coverage

# 前端阅读结果的推荐上限；允许不足十篇，不为凑数放宽来源和相关性校验。
MAX_RECOMMENDATIONS = 10


def keywords(value):
    """兼容数据库中的关键词数组、JSON 字符串和分隔文本。

    Args:
        value: 中英文关键词字段的原始值；非列表且无法规范化时视为空。

    Returns:
        去除空项、每项最多八十字符的关键词列表，保留原始顺序。
    """
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            value = re.split(r"[;,，；、]", value)
    return (
        [str(item).strip()[:80] for item in value if str(item).strip()]
        if isinstance(value, list)
        else []
    )


def search_queries(readings):
    """按所选文献顺序组合标题、关键词和首段分析摘要，最多生成五条查询。

    Args:
        readings: 逐篇分析节点返回的题录及 analysis；未解析文献也可使用题录。

    Returns:
        每条最多六百字符的查询列表。标题字面覆盖率大于 0.8 时跳过近似主题，
        不再次调用生成模型扩展问题，也不把整篇分析模板送去向量编码。
    """
    queries = []
    for item in readings:
        title = str(item.get("title") or "").strip()[:200]
        if not title or any(term_coverage(title, old) > 0.8 for old in queries):
            continue
        terms = keywords(item.get("keyword_cn")) + keywords(item.get("keyword_en"))
        summary = (item.get("analysis") or {}).get("summary") or []
        text = str(summary[0].get("text", ""))[:160] if summary else ""
        queries.append(" ".join([title, *terms[:6], text]).strip()[:600])
        if len(queries) == 5:
            break
    return queries


def metadata_terms(readings):
    """提取仅用于题录补位的检索词，不复用正文检索的长查询。

    Args:
        readings: 含中英文关键词和标题的所选文献记录。

    Returns:
        按出现频次排列的至多十二个词，来自关键词、标题引号短语和英文词。
        固定泛词会被排除；未带关键词的普通中文标题不会自动做完整分词。
    """
    terms = []
    for item in readings:
        terms.extend(keywords(item.get("keyword_cn")) + keywords(item.get("keyword_en")))
        title = str(item.get("title") or "")
        terms.extend(re.findall(r'[“「"]([^”」"]{2,16})[”」"]', title))
        terms.extend(re.findall(r"[A-Za-z][A-Za-z-]{3,}", title))
    stop = {"policy", "research", "study", "analysis", "from", "with", "研究", "政策", "教育"}
    return [
        term for term, _ in Counter(t.casefold() for t in terms).most_common() if term not in stop
    ][:12]


def public_record(row):
    """把最终复核的数据库记录投影为可公开展示的推荐题录。

    Args:
        row: 已审核且未删除的文献行，附件字段兼容字符串与旧版字典列表。

    Returns:
        文献 ID、标题、作者、年份、来源和去重附件路径，不包含索引片段或评分。
        附件字段损坏时返回空路径；实际打开原文仍由公开文件接口重新校验。
    """
    try:
        attachments = json.loads(row["attachment"] or "[]")
    except (TypeError, ValueError):
        attachments = []
    paths = []
    for item in attachments if isinstance(attachments, list) else []:
        path = item.get("file_path") if isinstance(item, dict) else item
        if isinstance(path, str) and path and path not in paths:
            paths.append(path)
    return {
        "corpus_id": int(row["id"]),
        "title": row["title_cn"] or row["title_en"],
        "author": row["author"] or "",
        "year": row["publish_time"],
        "source": row["source"] or "",
        "attachment_paths": paths,
    }


class LiteratureRecommender:
    """在公开文献池内融合双路全文检索，并用题录命中补充推荐空位。

    推荐依赖真实索引和数据库题录；不调用生成模型挑选名单，也不把推荐候选
    注册为本轮回答 Evidence。每次调用的候选、积分和失败状态均为局部数据。
    """

    def __init__(self, actor_id, settings):
        """绑定当前账号与本次调用使用的外部检索配置。

        Args:
            actor_id: 当前 Foreign 账号 ID，回读题录时复查账号有效性。
            settings: Qdrant、OpenSearch、Embedding 凭据及最低向量分数配置。
        """
        # 推荐公开文献仍要求调用账号有效，不能绕开账号停用检查。
        self.actor_id = actor_id
        # 保留配置对象供各检索分支使用，不把凭据写入推荐结果。
        self.settings = settings

    def filter_external(self, papers, readings):
        """对联网候选查全库去重，而不只排除本次选择或已推荐的文献。

        包含全部未删除题录，不受审核/索引状态和知识库范围限制；只据此排除重复，
        不向调用方返回这些题录。比对中英文题名、轻微题名差异及原始来源中的 DOI。
        """
        candidates = [dict(paper) for paper in papers]
        with session_factory()() as session:
            if not session.scalar(select(foreign_user.c.id).where(
                foreign_user.c.id == self.actor_id,
                foreign_user.c.status == 1, foreign_user.c.is_deleted == 0,
            )):
                raise PermissionError("账号已停用")
            if not candidates:
                return []
            with session.execute(select(
                foreign_corpus.c.title_cn, foreign_corpus.c.title_en,
                foreign_corpus.c.original_source,
            ).where(foreign_corpus.c.is_deleted == 0).execution_options(yield_per=1000)) as rows:
                for row in rows.mappings():
                    doi = extract_doi(row["original_source"])
                    candidates = [paper for paper in candidates if not (
                        (doi and doi == paper.get("doi"))
                        or same_title(paper["title"], row["title_cn"])
                        or same_title(paper["title"], row["title_en"])
                    )]
                    if not candidates:
                        break
        result = []
        for paper in candidates:
            if any(same_title(paper["title"], item.get("title", "")) for item in readings):
                continue
            if any((paper.get("doi") and paper["doi"] == item.get("doi"))
                   or (paper.get("external_url")
                       and paper["external_url"] == item.get("external_url"))
                   or same_title(paper["title"], item["title"])
                   for item in result):
                continue
            result.append(paper)
        return result[:MAX_WEB_RECOMMENDATIONS]

    def _rows(self, ids):
        """按 ID 回查公开题录及索引版本，供检索后校验和发布前复核。

        Args:
            ids: 待核对的文献 ID 集合或有序列表；空输入不访问数据库。

        Returns:
            以整数文献 ID 为键的记录，只含已审核且未删除的文献。
            索引使用外连接，允许题录补位候选尚未完成全文索引。

        Raises:
            PermissionError: 当前账号已停用、删除或不存在。
        """
        if not ids:
            return {}
        job = foreign_corpus_index_job.c
        with session_factory()() as session:
            if not session.scalar(
                select(foreign_user.c.id).where(
                    foreign_user.c.id == self.actor_id,
                    foreign_user.c.status == 1,
                    foreign_user.c.is_deleted == 0,
                )
            ):
                raise PermissionError("账号已停用")
            rows = session.execute(
                select(
                    foreign_corpus,
                    job.build_id.label("index_build_id"),
                    job.attachment_digest.label("index_attachment_digest"),
                    job.status.label("index_status"),
                    job.vector_status,
                    job.lexical_status,
                    job.lexical_index,
                    job.is_deleted.label("index_deleted"),
                )
                .outerjoin(foreign_corpus_index_job, job.corpus_id == foreign_corpus.c.id)
                .where(
                    foreign_corpus.c.id.in_(ids),
                    foreign_corpus.c.is_deleted == 0,
                    foreign_corpus.c.status == 2,
                    foreign_corpus.c.owner_id.is_(None),
                )
            ).mappings()
            return {int(row["id"]): dict(row) for row in rows}

    def _metadata(self, readings, excluded):
        """从标题和关键词字段查找补位候选，不读取或分析候选全文。

        Args:
            readings: 用于提取题录检索词的所选文献。
            excluded: 本次已选文献 ID，不能再次出现在推荐中。

        Returns:
            按 ID 截取的最多一百条公开题录。这里采用任一词项子串命中，
            后续才按命中词数排序，因此不能视为严格的主题相关性判断。
        """
        terms = metadata_terms(readings)
        if not terms:
            return []
        columns = (
            foreign_corpus.c.title_cn,
            foreign_corpus.c.title_en,
            foreign_corpus.c.keyword_cn,
            foreign_corpus.c.keyword_en,
        )
        with session_factory()() as session:
            return [
                dict(row)
                for row in session.execute(
                    select(foreign_corpus)
                    .where(
                        foreign_corpus.c.status == 2,
                    foreign_corpus.c.owner_id.is_(None),
                        foreign_corpus.c.is_deleted == 0,
                        foreign_corpus.c.id.not_in(excluded),
                        or_(
                            *(
                                column.contains(term, autoescape=True)
                                for term in terms
                                for column in columns
                            )
                        ),
                    )
                    .order_by(foreign_corpus.c.id)
                    .limit(100)
                ).mappings()
            ]

    def _lexical(self, query, excluded):
        """在 OpenSearch 的标题和正文两组字段中召回关键词匹配片段。

        Args:
            query: 一条由所选文献构造的主题查询。
            excluded: 必须排除的所选文献 ID。

        Returns:
            最多八十条原始命中，保留分数及版本指纹供 Qdrant 回查后交叉验证。

        Raises:
            RuntimeError: 查询超时或分片失败，不能把不完整响应标记为完整成功。
        """
        client = OpenSearch(hosts=[self.settings.opensearch_url], timeout=8)
        try:
            response = client.search(
                index=self.settings.opensearch_index,
                body={
                    "size": 80,
                    "_source": [
                        "point_id",
                        "corpus_id",
                        "build_id",
                        "attachment_digest",
                        "content_hash",
                    ],
                    "query": {
                        "bool": {
                            "must_not": [{"terms": {"corpus_id": sorted(excluded)}}],
                            "must": [
                                {
                                    "multi_match": {
                                        "query": query,
                                        "fields": [
                                            "title^3",
                                            "title_standard^3",
                                            "content",
                                            "content_standard",
                                        ],
                                        "type": "best_fields",
                                    }
                                }
                            ],
                        }
                    },
                },
            )
            if response.get("timed_out") or response.get("_shards", {}).get("failed", 0):
                raise RuntimeError("推荐词法检索未完整完成")
            return response["hits"]["hits"]
        finally:
            client.close()

    def _vector(self, client, vector, excluded):
        """在当前 Embedding 模型和维度对应的向量点中召回语义相似片段。

        Args:
            client: 本轮复用的 Qdrant 客户端。
            vector: 当前查询的向量；编码失败时调用方直接跳过本方法。
            excluded: 必须排除的所选文献 ID。

        Returns:
            最多八十条带 payload 的命中。最低相似度及来源校验在回查后执行。
        """
        config = embedding_config()
        return client.query_points(
            collection_name=self.settings.qdrant_collection,
            query=vector,
            query_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="embedding_model", match=models.MatchValue(value=config.model)
                    ),
                    models.FieldCondition(
                        key="embedding_dimension", match=models.MatchValue(value=config.dimension)
                    ),
                ],
                must_not=[
                    models.FieldCondition(
                        key="corpus_id", match=models.MatchAny(any=sorted(excluded))
                    )
                ],
            ),
            limit=80,
            with_payload=True,
        ).points

    def recommend(self, readings):
        """生成最多十篇推荐，并返回检索是否完整完成的状态。

        Args:
            readings: 逐篇阅读结果，包含真实题录、关键词和可选的分析摘要。

        Returns:
            recommended_documents 为全文融合结果优先、题录候选补位的去重名单。
            检索和来源回查未记录失败时为 SUCCESS，即使结果为空；部分失败且有结果为 PARTIAL，
            存在检索失败且无结果为 FAILED。各路积分不作为相关性百分比返回。

        Raises:
            PermissionError: 回读题录时账号失效；图节点将异常降级为推荐失败。
        """
        # 1. 固定排除集合并生成主题查询；不推荐用户本次已经选中的文献。
        excluded = {int(item["corpus_id"]) for item in readings}
        queries = search_queries(readings)
        if not queries:
            return {"recommended_documents": [], "recommendation_status": "SUCCESS"}
        scores, available, failed = {}, set(), set()
        # 2. 先获取题录补位候选；最终仍需重新核对公开状态，不能直接发布此快照。
        metadata = self._metadata(readings, excluded)

        # 两个分支可独立降级；向量批量编码，词法检索无需向量服务。
        def vectors():
            """在线程池中批量编码查询，结果顺序与 queries 一一对应。"""
            return embed_texts(queries, text_type="query", key=self.settings.embedding_api_key)

        # 3. 让批量向量编码与词法召回重叠执行；向量查询本身在取得编码后逐条进行。
        # 任一分支异常只记录降级，保留另一分支或题录补位产生可用结果的机会。
        with ThreadPoolExecutor(max_workers=2) as pool:
            future = pool.submit(vectors)
            try:
                lexical = [self._lexical(query, excluded) for query in queries]
                available.add("lexical")
            except Exception:
                lexical = [[] for _ in queries]
                failed.add("lexical")
            try:
                embedded = future.result()
            except Exception:
                embedded = [None for _ in queries]
                failed.add("vector")
        # 4. 逐条查询向量候选，再回取真实片段，校验来源后按文献累计排名积分。
        with closing(
            QdrantClient(url=self.settings.qdrant_url, timeout=8, check_compatibility=False)
        ) as client:
            for query, vector, words in zip(queries, embedded, lexical, strict=True):
                try:
                    hits = self._vector(client, vector, excluded) if vector is not None else []
                    if vector is not None:
                        available.add("vector")
                except Exception:
                    hits = []
                    failed.add("vector")
                ids = {int(hit.id) for hit in hits}
                ids.update(int(hit["_id"]) for hit in words)
                if not ids:
                    continue
                # 词法命中也要从 Qdrant 回查，因此仅 OpenSearch 可用并不保证可发布全文推荐。
                try:
                    points = client.retrieve(
                        self.settings.qdrant_collection,
                        ids=sorted(ids),
                        with_payload=True,
                        with_vectors=False,
                    )
                except Exception:
                    failed.add("sources")
                    continue
                corpus_ids = {int(point.payload.get("corpus_id", 0)) for point in points}
                rows = {
                    key: row
                    for key, row in self._rows(corpus_ids - excluded).items()
                    if row["is_vectorized"]
                    and row["index_status"] == "SUCCESS"
                    and row["vector_status"] == "SUCCESS"
                    and row["lexical_status"] == "SUCCESS"
                    and not row["index_deleted"]
                    and row["lexical_index"] == self.settings.opensearch_index
                    and row["index_attachment_digest"]
                    == hashlib.sha256((row["attachment"] or "").encode()).hexdigest()
                }
                # 共同验证构建版本、附件摘要、正文哈希及可定位来源，剔除陈旧或损坏索引。
                vector_hits, lexical_hits, sources = _verified_candidates(hits, words, points, rows)
                # 每条查询的每个分支中，一篇文献最多贡献一次，长文献不会刷分。
                for branch in (vector_hits, lexical_hits):
                    seen = set()
                    for hit in branch:
                        source = sources[hit.chunk_id]
                        if source.corpus_id in seen:
                            continue
                        if branch is vector_hits and hit.score < self.settings.min_vector_score:
                            continue
                        if branch is lexical_hits:
                            row = rows[source.corpus_id]
                            title_text = " ".join(
                                str(row.get(key) or "")
                                for key in ("title_cn", "title_en", "keyword_cn", "keyword_en")
                            )
                            if (
                                term_coverage(query, title_text) < 0.12
                                and term_coverage(query, source.content) < 0.25
                            ):
                                continue
                        seen.add(source.corpus_id)
                        # len(seen) 是本分支通过门槛后的文献名次，不是片段原始排名。
                        # 同一文献跨查询或跨分支可累计贡献，单分支内重复片段不重复加分。
                        scores[source.corpus_id] = scores.get(source.corpus_id, 0) + 1 / (
                            60 + len(seen)
                        )
        # 5. 题录仅追加在全文融合名单之后；按命中词数排序，可补充尚未索引的文献。
        # 这一补位没有全文相关性门槛，标题含泛化关键词的文章也可能进入名单。
        terms = metadata_terms(readings)
        metadata.sort(
            key=lambda row: (
                -sum(
                    term
                    in " ".join(
                        str(row.get(key) or "").casefold()
                        for key in ("title_cn", "title_en", "keyword_cn", "keyword_en")
                    )
                    for term in terms
                ),
                row["id"],
            )
        )
        order = sorted(scores, key=lambda key: (-scores[key], key))
        order.extend(row["id"] for row in metadata if row["id"] not in scores)
        # 6. 发布前再次核对题录状态并截取上限；推荐原文不进入本次回答的 evidence/sources。
        current = self._rows(order)
        documents = [public_record(current[key]) for key in order if key in current][
            :MAX_RECOMMENDATIONS
        ]
        status = "PARTIAL" if failed and documents else "FAILED" if failed else "SUCCESS"
        return {"recommended_documents": documents, "recommendation_status": status}
