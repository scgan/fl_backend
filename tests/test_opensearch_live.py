"""可选的 OpenSearch 真机契约测试，只创建并清理本次专属临时索引。"""

import hashlib
import os
import unittest
import uuid
from types import SimpleNamespace
from unittest.mock import patch

from app.indexing.opensearch_store import OpenSearchStore


@unittest.skipUnless(os.getenv("FOREIGN_TEST_OPENSEARCH_URL"), "未指定独立 OpenSearch 验证地址")
class OpenSearchLiveTests(unittest.TestCase):
    """验证 2.9.0 的 mapping、bulk 可见性、中文/英文召回和定向删除。"""

    def test_write_search_and_delete(self):
        """临时测试数据不得写入业务索引或访问业务文献。"""
        index = "foreign_test_" + uuid.uuid4().hex
        with patch.dict(os.environ, {
            "FOREIGN_OPENSEARCH_URL": os.environ["FOREIGN_TEST_OPENSEARCH_URL"],
            "FOREIGN_OPENSEARCH_INDEX": index,
        }):
            store = OpenSearchStore()
        try:
            self.assertEqual(store.client.info()["version"]["number"], "2.9.0")
            store.ensure_index()
            store.ensure_index()
            content = "教师反馈改善学习效果 teacher feedback improves learning"
            payload = dict(corpus_id=42, title_cn="教师反馈", title_en="Teacher feedback",
                           text=content, build_id="test-build", attachment_digest="test-digest",
                           content_hash=hashlib.sha256(content.encode()).hexdigest(), file_name="test.pdf")
            store.write_points([SimpleNamespace(id=42000000, payload=payload)])
            for query in ("教师反馈", "teacher feedback"):
                response = store.client.search(index=index, body={"query": {"bool": {
                    "filter": [{"terms": {"corpus_id": [42]}}, {"terms": {"build_id": ["test-build"]}}],
                    "must": [{"multi_match": {"query": query, "fields": ["title^3", "content", "content_standard"]}}],
                }}})
                self.assertEqual(response["hits"]["hits"][0]["_id"], "42000000")
            store.delete_corpus(999)
            self.assertEqual(store.client.count(index=index)["count"], 1)
            store.delete_corpus(42, ["test.pdf"])
            self.assertEqual(store.client.count(index=index)["count"], 0)
        finally:
            if store.client.indices.exists(index=index):
                store.client.indices.delete(index=index)
            store.client.close()


if __name__ == "__main__":
    unittest.main()
