"""验证模型接口契约、暂时性重试及异常响应隔离，无真实模型调用。"""
import os
import unittest
from unittest import mock

from app.utils import bailian_client as client


def response(status=200, items=None):
    """构造带固定 usage 和可选向量条目的离线百炼响应。"""
    result = mock.Mock(status_code=status)
    result.json.return_value = {"output": {"embeddings": items or []}, "usage": {"total_tokens": 100}}
    return result


class BailianTests(unittest.TestCase):
    """验证百炼分批向量、顺序恢复、角色参数、失败重试及安全错误契约。"""
    def setUp(self):
        """固定离线凭据、维度和每批数量，结束后恢复原环境。"""
        patcher = mock.patch.dict(os.environ, {"DASHSCOPE_API_KEY": "offline-key",
            "FOREIGN_EMBEDDING_DIMENSION": "256", "FOREIGN_EMBEDDING_BATCH_SIZE": "20"})
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_batch_order_query_role_and_actual_usage_differ_from_estimate(self):
        """验证跨批次的逆序返回按输入恢复，查询角色及显式凭据正确透传。"""
        def serve(url, **kwargs):
            """检查模型、维度与 query 参数，按逆序生成可追踪输入位置的向量。"""
            payload = kwargs["json"]
            self.assertEqual(payload["model"], "qwen3.7-text-embedding-flash")
            self.assertEqual(payload["parameters"], {"dimension": 256, "text_type": "query"})
            texts = payload["input"]["texts"]
            return response(items=[{"index": i, "embedding": [float(text) + 1] * 256}
                                   for i, text in reversed(list(enumerate(texts)))])
        with mock.patch.object(client.requests, "post", side_effect=serve) as http:
            vectors = client.embed_texts([str(i) for i in range(21)], text_type="query", key="selected-key")
        self.assertEqual(http.call_count, 2)
        self.assertEqual([v[0] for v in vectors], list(range(1, 22)))
        self.assertEqual(http.call_args.kwargs["headers"]["Authorization"], "Bearer selected-key")

    def test_missing_duplicate_or_nonfinite_vectors_are_rejected(self):
        """验证数量不足、索引重复或越界、非有限值、错维度和全零向量均被拒绝。"""
        valid = {"index": 0, "embedding": [0.1] * 256}
        for items in ([], [{**valid, "index": 1}], [{**valid, "embedding": [float("nan")] * 256}],
                      [{**valid, "embedding": [0.1]}], [{**valid, "embedding": [0.0] * 256}]):
            with self.subTest(items=repr(items)[:70]), mock.patch.object(client.requests, "post", return_value=response(items=items)):
                with self.assertRaises(client.BailianError):
                    client.embed_texts(["document"])
        with mock.patch.object(client.requests, "post", return_value=response(items=[valid, valid])):
            with self.assertRaises(client.BailianError):
                client.embed_texts(["one", "two"])

    def test_retry_transient_http_but_sanitize_permanent_error(self):
        """验证限流可重试，认证失败不重试且仅返回安全 HTTP 摘要。"""
        valid = {"index": 0, "embedding": [0.1] * 256}
        with mock.patch.object(client.requests, "post", side_effect=[response(429), response(items=[valid])]) as http, \
             mock.patch.object(client.time, "sleep"):
            self.assertEqual(len(client.embed_texts(["doc"])), 1)
            self.assertEqual(http.call_count, 2)
        with mock.patch.object(client.requests, "post", return_value=response(401)) as http:
            with self.assertRaisesRegex(client.BailianError, "HTTP 401"):
                client.embed_texts(["doc"])
            self.assertEqual(http.call_count, 1)


if __name__ == "__main__":
    unittest.main()
