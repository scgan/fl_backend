"""只读访问 Foreign 共享 OpenSearch 索引，返回带版本信息的 BM25 候选。"""

from __future__ import annotations

from opensearchpy import OpenSearch

from app.foreign_integration.config import ForeignSettings


def lexical_search(
    settings: ForeignSettings, query: str, corpus_ids: list[int] | None,
    build_ids: list[str] | None, limit: int
) -> list[dict]:
    """召回待核验的词法候选；全库不添加成员过滤，返回后统一校验权限和版本。

    Args:
        settings: 共享词法索引配置。
        query: 双语查询表达式。
        corpus_ids: 当前资源文献集合，None 表示全库，空列表表示无候选。
        build_ids: 可选的已发布构建令牌，全库由命中后的回库校验确认。
        limit: 分支候选上限。

    Returns:
        保留 ID、BM25 分和来源哈希的原始命中列表。
    """
    if corpus_ids is not None and not corpus_ids:
        return []
    filters = []
    if corpus_ids is not None:
        filters.append({"terms": {"corpus_id": corpus_ids}})
    if build_ids is not None:
        filters.append({"terms": {"build_id": build_ids}})
    client = OpenSearch(hosts=[settings.opensearch_url], timeout=30, verify_certs=True)
    try:
        response = client.search(
            index=settings.opensearch_index,
            body={
                "size": limit,
                "_source": [
                    "point_id",
                    "corpus_id",
                    "build_id",
                    "attachment_digest",
                    "content_hash",
                ],
                "query": {
                    "bool": {
                        "filter": filters,
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
            raise RuntimeError("词法召回未完整完成")
        return response["hits"]["hits"]
    finally:
        client.close()
