"""管理 Foreign 2.9 兼容词法索引；与 Qdrant 共用 point ID 和构建版本。"""

from __future__ import annotations

import os
from functools import lru_cache

from opensearchpy import OpenSearch, RequestError
from opensearchpy.helpers import bulk


class OpenSearchStore:
    """提供版本化 mapping、同步可见的批量写入和幂等删除。"""

    def __init__(self):
        """从环境读取连接配置，HTTPS 默认校验证书。"""
        # 当前写入和删除共用的词法索引名，必须与研究召回端配置一致。
        self.index_name = os.getenv("FOREIGN_OPENSEARCH_INDEX", "foreign_chunks_v1")
        # 当前进程复用的同步 HTTP 客户端，负责索引建模、批写和清理请求。
        self.client = OpenSearch(
            hosts=[os.getenv("FOREIGN_OPENSEARCH_URL", "http://localhost:19200")],
            timeout=30, verify_certs=True,
        )

    def ensure_index(self):
        """缺少目标索引时创建严格双分析器 mapping，容忍并发创建同名索引。

        Raises:
            RequestError: 创建失败且并非另一执行者已创建相同索引。
        """
        if self.client.indices.exists(index=self.index_name):
            return
        properties = {
            name: {"type": "keyword"} for name in (
                "point_id", "build_id", "attachment_digest", "content_hash", "file_name",
            )
        }
        properties.update({
            "corpus_id": {"type": "long"},
            "title": {"type": "text", "analyzer": "cjk"},
            "title_standard": {"type": "text", "analyzer": "standard"},
            "content": {"type": "text", "analyzer": "cjk"},
            "content_standard": {"type": "text", "analyzer": "standard"},
        })
        try:
            self.client.indices.create(index=self.index_name, body={
                "settings": {"number_of_shards": 1, "number_of_replicas": 0},
                "mappings": {"dynamic": "strict", "properties": properties},
            })
        except RequestError as error:
            if error.error != "resource_already_exists_exception":
                raise

    def write_points(self, points):
        """把同一批 Qdrant point 转为词法文档，任何 bulk 子项失败均抛错。

        Args:
            points: 携带共享版本、正文和哈希的 PointStruct 列表。
        """
        # 每个词法文档复用向量点 ID、构建令牌、附件指纹与正文哈希，召回时才能交叉核验。
        actions = []
        for point in points:
            payload = point.payload
            title = " ".join(filter(None, [payload.get("title_cn"), payload.get("title_en")]))
            source = {name: payload[name] for name in (
                "corpus_id", "build_id", "attachment_digest", "content_hash", "file_name",
            )}
            source.update(point_id=str(point.id), title=title, title_standard=title,
                          content=payload["text"], content_standard=payload["text"])
            actions.append({"_index": self.index_name, "_id": str(point.id), "_source": source})
        if actions:
            bulk(self.client, actions, refresh="wait_for", raise_on_error=True)

    def delete_corpus(self, corpus_id, file_names=None):
        """删除文献或指定附件的词法副本，不因清理创建索引。

        Args:
            corpus_id: Foreign 文献 ID。
            file_names: 可选附件文件名列表。

        Raises:
            RuntimeError: 服务报告子项失败、超时或版本冲突，删除不能视为完整成功。
        """
        if not self.client.indices.exists(index=self.index_name):
            return
        filters = [{"term": {"corpus_id": corpus_id}}]
        if file_names:
            filters.append({"terms": {"file_name": file_names}})
        result = self.client.delete_by_query(
            index=self.index_name, body={"query": {"bool": {"filter": filters}}},
            refresh=True, conflicts="proceed",
        )
        if result.get("failures") or result.get("timed_out") or result.get("version_conflicts"):
            raise RuntimeError("词法索引清理未完成")


@lru_cache
def get_lexical_store():
    """返回进程级词法适配器。"""
    return OpenSearchStore()
