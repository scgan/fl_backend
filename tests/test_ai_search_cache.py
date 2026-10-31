"""离线验证 AI 检索的 SQL/结果快照缓存复用、用户隔离、过期和 Redis 故障行为。"""

import json
import unittest
from types import SimpleNamespace
from unittest import mock

from flask import Flask, g
from redis.exceptions import ConnectionError as RedisConnectionError

from app.api import AIsearch


class AISearchCacheTests(unittest.TestCase):
    """使用真实 Flask 检索路由，替换模型、数据库和 Redis 以禁止外部请求。"""

    def setUp(self):
        """绑定测试账号，并让 Redis 替身保存真实键值以验证跨请求复用。"""
        self.app = Flask(__name__)
        self.app.register_blueprint(AIsearch.bp, url_prefix='/ai_search')
        self.actor = 7  # 当前请求的用户 ID；改为 None 用于未登录测试。
        self.cache = {}  # 模拟 Redis：同时保存 SQL 与结果快照 JSON。
        self.sql = "SELECT * FROM corpus WHERE title_cn LIKE '%双减%'"
        self.rows = [{'id': i, 'title_cn': f'文献{i}'} for i in range(1, 47)]

        @self.app.before_request
        def authenticate():
            """提供登录上下文，测试时无需访问真实用户表或令牌服务。"""
            g.current_user = SimpleNamespace(id=self.actor) if self.actor is not None else None

        self.redis = self.patch(AIsearch, 'redis_client')
        self.redis.get.side_effect = self.cache.get
        self.redis.setex.side_effect = self.cache_set
        self.generate = self.patch(AIsearch.nl2sql_manager, 'generate_query', return_value=self.sql)
        self.execute = self.patch(AIsearch.nl2sql_manager, 'execute_query', return_value=self.rows)
        self.client = self.app.test_client()

    def patch(self, target, name, **kwargs):
        """注册兼容 Python 3.9 的替身清理，测试结束后恢复真实对象。"""
        patcher = mock.patch.object(target, name, **kwargs)
        self.addCleanup(patcher.stop)
        return patcher.start()

    def cache_set(self, key, ttl, value):
        """模拟 SETEX：SQL 与结果快照都按原文保存，有效期由调用断言检查。"""
        self.cache[key] = value
        return True

    def search(self, **payload):
        """向真实路由提交一次自然语言搜索或分页请求。"""
        return self.client.post('/ai_search/nl-search', json=payload)

    def test_pagination_serves_from_result_snapshot(self):
        """首次检索写 SQL 与结果快照，之后翻页/改页容量都不再查库也不再调模型。"""
        first = self.search(question='双减', page_num=1, page_size=10)
        self.assertEqual(first.status_code, 200)
        search_id = first.json['data']['search_id']
        self.assertRegex(search_id, r'^[0-9a-f]{32}$')
        self.assertEqual(first.json['data']['pagination']['total'], 46)
        self.assertEqual(self.generate.call_count, 1)
        self.assertEqual(self.execute.call_count, 1)

        # 首次检索同时缓存 SQL 与结果快照，两个键都带 15 分钟有效期。
        self.assertEqual(
            self.redis.setex.call_args_list,
            [
                mock.call(
                    f'ai_search:result:7:{search_id}',
                    AIsearch.AI_SEARCH_RESULT_TTL,
                    json.dumps(self.rows, ensure_ascii=False),
                ),
                mock.call(f'ai_search:sql:7:{search_id}', AIsearch.AI_SEARCH_SQL_TTL, self.sql),
            ],
        )

        second = self.search(search_id=search_id, page_num=2, page_size=10)
        self.assertEqual([row['id'] for row in second.json['data']['result']], list(range(11, 21)))
        self.assertEqual(second.json['data']['search_id'], search_id)
        resized = self.search(search_id=search_id, page_num=1, page_size=20)
        self.assertEqual(len(resized.json['data']['result']), 20)
        last_page = self.search(search_id=search_id, page_num=5, page_size=10)
        self.assertEqual([row['id'] for row in last_page.json['data']['result']], list(range(41, 47)))

        # 翻页与改页容量全部命中快照：模型与数据库都只被调用一次。
        self.assertEqual(self.generate.call_count, 1)
        self.assertEqual(self.execute.call_count, 1)
        self.redis.setex.assert_has_calls(
            [mock.call(f'ai_search:sql:7:{search_id}', AIsearch.AI_SEARCH_SQL_TTL, self.sql)],
        )

    def test_new_search_gets_new_id_even_for_same_question(self):
        """显式重新搜索会重新生成 SQL，且不会覆盖前一次检索的缓存。"""
        first = self.search(question='双减').json['data']['search_id']
        second = self.search(question='双减').json['data']['search_id']
        self.assertNotEqual(first, second)
        self.assertEqual(self.generate.call_count, 2)
        self.assertEqual(
            set(self.cache),
            {
                f'ai_search:sql:7:{first}',
                f'ai_search:result:7:{first}',
                f'ai_search:sql:7:{second}',
                f'ai_search:result:7:{second}',
            },
        )

    def test_missing_result_snapshot_falls_back_to_sql_and_recaches(self):
        """结果快照缺失但 SQL 仍在时，回源查询一次并重新写入快照。"""
        search_id = self.search(question='双减').json['data']['search_id']
        self.execute.reset_mock()
        self.redis.setex.reset_mock()
        del self.cache[f'ai_search:result:7:{search_id}']

        page_two = self.search(search_id=search_id, page_num=2, page_size=10)
        self.assertEqual([row['id'] for row in page_two.json['data']['result']], list(range(11, 21)))
        self.assertEqual(self.generate.call_count, 1)
        self.assertEqual(self.execute.call_count, 1)
        self.redis.setex.assert_called_once_with(
            f'ai_search:result:7:{search_id}',
            AIsearch.AI_SEARCH_RESULT_TTL,
            json.dumps(self.rows, ensure_ascii=False),
        )

    def test_oversized_result_is_not_snapshotted(self):
        """结果超过上限时只缓存 SQL，翻页回源查询但不再调用模型。"""
        self.execute.return_value = [
            {'id': i, 'title_cn': f'文献{i}'}
            for i in range(1, AIsearch.AI_SEARCH_RESULT_LIMIT + 2)
        ]
        search_id = self.search(question='双减').json['data']['search_id']
        self.assertNotIn(f'ai_search:result:7:{search_id}', self.cache)
        self.assertIn(f'ai_search:sql:7:{search_id}', self.cache)

        self.execute.reset_mock()
        page_two = self.search(search_id=search_id, page_num=2, page_size=10)
        self.assertEqual(len(page_two.json['data']['result']), 10)
        self.assertEqual(self.execute.call_count, 1)
        self.assertEqual(self.generate.call_count, 1)

    def test_missing_expired_and_other_users_cache_do_not_regenerate(self):
        """其他用户、过期或不存在的标识均不触发模型或数据库查询。"""
        search_id = self.search(question='双减').json['data']['search_id']
        self.generate.reset_mock()
        self.execute.reset_mock()
        self.actor = 8
        self.assertEqual(self.search(search_id=search_id).status_code, 410)
        self.actor = 7
        self.cache.clear()
        self.assertEqual(self.search(search_id=search_id, question='双减').status_code, 410)
        self.assertEqual(self.search(search_id='0' * 32).status_code, 410)
        self.generate.assert_not_called()
        self.execute.assert_not_called()

    def test_redis_failure_returns_retryable_error_without_regenerating(self):
        """缓存读取或写入失败时返回 503，不把翻页降级成重新调用模型。"""
        self.redis.get.side_effect = RedisConnectionError('offline')
        self.assertEqual(self.search(search_id='a' * 32).status_code, 503)
        self.generate.assert_not_called()
        self.execute.assert_not_called()
        self.redis.setex.side_effect = RedisConnectionError('offline')
        self.assertEqual(self.search(question='双减').status_code, 503)

    def test_invalid_parameters_and_unauthenticated_requests_are_rejected(self):
        """非法参数和无登录账号在访问模型、数据库及 Redis 前被拒绝。"""
        for payload in ({}, {'question': 12}, {'question': ' '}, {'search_id': '../other'},
                        {'search_id': ''}, {'question': '双减', 'page_num': 0},
                        {'question': '双减', 'page_size': None}):
            with self.subTest(payload=payload):
                self.assertEqual(self.search(**payload).status_code, 400)
        self.actor = None
        self.assertEqual(self.search(question='双减').status_code, 401)
        self.generate.assert_not_called()
        self.execute.assert_not_called()
        self.redis.get.assert_not_called()
        self.redis.setex.assert_not_called()

    def test_failed_query_is_not_cached_but_empty_query_is_cached(self):
        """执行失败不发布缓存；成功的空结果同样写快照，翻页无需再查库。"""
        self.execute.side_effect = RuntimeError('query failed')
        self.assertEqual(self.search(question='双减').status_code, 500)
        self.redis.setex.assert_not_called()
        self.execute.side_effect = None
        self.execute.return_value = []
        first = self.search(question='双减')
        self.assertEqual(first.json['data']['pagination']['total'], 0)
        self.execute.reset_mock()
        second = self.search(search_id=first.json['data']['search_id'])
        self.assertEqual(second.json['data']['result'], [])
        self.assertEqual(second.json['data']['pagination']['total'], 0)
        self.assertEqual(self.generate.call_count, 2)
        self.assertEqual(self.execute.call_count, 0)


if __name__ == '__main__':
    unittest.main()
