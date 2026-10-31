"""推荐检索权限、版本、去重、降级及 graph 并行汇合回归。"""

import asyncio
import hashlib
import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sqlalchemy import create_engine, insert, update
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool
from test_literature import FakeGateway, FakeMaterials

from app.agents.literature_graph import LiteratureGraph
from app.foreign_integration.recommendations import LiteratureRecommender, search_queries
from app.foreign_integration.schema import (
    foreign_corpus,
    foreign_corpus_index_job,
    foreign_user,
    metadata,
)


class RecommendationTests(unittest.TestCase):
    """用隔离文献与构建快照验证推荐的可见性、去重、版本过滤和分路降级。"""
    def setUp(self):
        """创建有效、未审核及删除文献和双路任务，固定附件指纹与工具配置。"""
        self.engine = create_engine(
            "sqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False}
        )
        metadata.create_all(self.engine)
        self.session = Session(self.engine)
        self.session.execute(insert(foreign_user).values(id=7, status=1, is_deleted=0))
        self.attachment = json.dumps([{"file_path": "paper.pdf"}])
        self.digest = hashlib.sha256(self.attachment.encode()).hexdigest()
        for key in range(1, 17):
            self.session.execute(
                insert(foreign_corpus).values(
                    id=key,
                    title_cn=f"双减教师负担研究{key}",
                    author="作者",
                    source="期刊",
                    publish_time=2024,
                    keyword_cn='["双减", "教师负担"]',
                    attachment=self.attachment,
                    is_deleted=key == 16,
                    status=1 if key == 15 else 2,
                    is_vectorized=True,
                )
            )
            self.session.execute(
                insert(foreign_corpus_index_job).values(
                    corpus_id=key,
                    build_id=f"build-{key}",
                    attachment_digest=self.digest,
                    status="SUCCESS",
                    vector_status="SUCCESS",
                    lexical_status="SUCCESS",
                    lexical_index="chunks",
                    is_deleted=False,
                )
            )
        self.session.commit()
        self.settings = SimpleNamespace(
            embedding_api_key="key",
            qdrant_url="http://qdrant",
            qdrant_collection="chunks",
            opensearch_url="http://lexical",
            opensearch_index="chunks",
            min_vector_score=0.35,
        )
        self.tool = LiteratureRecommender(7, self.settings)
        self.readings = [
            {
                "corpus_id": 1,
                "title": "“双减”教师负担",
                "keyword_cn": '["双减"]',
                "analysis": {"summary": [{"text": "教师工作压力研究"}]},
            }
        ]
        self.factory = patch(
            "app.foreign_integration.recommendations.session_factory",
            return_value=lambda: Session(self.engine),
        )
        self.factory.start()

    def tearDown(self):
        """撤销会话工厂替换，关闭数据库会话和连接池。"""
        self.factory.stop()
        self.session.close()
        self.engine.dispose()

    def point(self, key, index=0):
        """构造与数据库构建版本、附件指纹和正文哈希可核对的向量命中。"""
        text = f"双减教师负担研究{key}描述教师工作压力"
        return SimpleNamespace(
            id=key * 1_000_000 + index,
            score=0.8,
            payload={
                "corpus_id": key,
                "text": text,
                "file_path": "paper.pdf",
                "page_start": 1,
                "page_end": 2,
                "build_id": f"build-{key}",
                "attachment_digest": self.digest,
                "content_hash": hashlib.sha256(text.encode()).hexdigest(),
                "embedding_model": "qwen3.7-text-embedding-flash",
                "embedding_dimension": 1024,
            },
        )

    def run_search(self, points, *, vector_error=False, lexical_error=False):
        """隔离模型与双路索引，可分别注入向量或词法错误以验证降级。"""
        words = [{"_id": str(p.id), "_score": 20, "_source": dict(p.payload)} for p in points]
        with (
            patch("app.foreign_integration.recommendations.embed_texts", return_value=[[0.1]]),
            patch("app.foreign_integration.recommendations.QdrantClient") as constructor,
            patch.object(
                self.tool,
                "_vector",
                side_effect=RuntimeError("vector") if vector_error else None,
                return_value=points,
            ),
            patch.object(
                self.tool,
                "_lexical",
                side_effect=RuntimeError("lexical") if lexical_error else None,
                return_value=words,
            ),
        ):
            constructor.return_value.retrieve.return_value = points
            return self.tool.recommend(self.readings)

    def test_public_results_exclude_selection_and_deduplicate_limit_ten(self):
        """验证推荐排除已选文献，按文献去重且最多返回十篇公开结果。"""
        points = [
            self.point(1),
            self.point(2),
            self.point(2, 1),
            *[self.point(key) for key in range(3, 17)],
        ]
        result = self.run_search(points)
        ids = [row["corpus_id"] for row in result["recommended_documents"]]
        self.assertEqual(ids, list(range(2, 12)))
        self.assertEqual(result["recommendation_status"], "SUCCESS")
        self.assertNotIn("reason", result["recommended_documents"][0])
        self.assertEqual(result["recommended_documents"][0]["attachment_paths"], ["paper.pdf"])

    def test_stale_and_forged_fragments_do_not_enter_fulltext_ranking(self):
        """验证过期构建或伪造哈希的切块不参与全文推荐排序。"""
        points = [self.point(2), self.point(3), self.point(4)]
        points[0].payload["build_id"] = "old"
        points[1].payload["content_hash"] = "fake"
        with patch.object(self.tool, "_metadata", return_value=[]):
            result = self.run_search(points)
        self.assertEqual([r["corpus_id"] for r in result["recommended_documents"]], [4])

    def test_bm25_survives_vector_failure_and_metadata_covers_unindexed(self):
        """验证向量分路失败时 BM25 可继续，未索引文献由题录路径补充。"""
        self.session.execute(
            update(foreign_corpus).where(foreign_corpus.c.id == 5).values(is_vectorized=False)
        )
        self.session.commit()
        result = self.run_search([self.point(2)], vector_error=True)
        self.assertEqual(result["recommendation_status"], "PARTIAL")
        self.assertIn(5, [r["corpus_id"] for r in result["recommended_documents"]])
        self.assertEqual(result["recommended_documents"][0]["corpus_id"], 2)

    def test_all_services_failed_is_distinct_from_no_matches(self):
        """验证全部服务失败与正常零命中具有不同诊断状态。"""
        with patch.object(self.tool, "_metadata", return_value=[]):
            self.assertEqual(
                self.run_search([], vector_error=True, lexical_error=True)["recommendation_status"],
                "FAILED",
            )
            self.assertEqual(self.run_search([])["recommendation_status"], "SUCCESS")

    def test_publication_rechecks_deleted_candidates(self):
        """验证召回后删除的候选在发布前复核时被排除。"""
        original = self.tool._rows
        calls = 0

        def rows(ids):
            """在第二次数据库复核前删除目标文献，模拟召回期间可见性变化。"""
            nonlocal calls
            calls += 1
            if calls == 2:
                self.session.execute(
                    update(foreign_corpus).where(foreign_corpus.c.id == 2).values(is_deleted=True)
                )
                self.session.commit()
            return original(ids)

        with patch.object(self.tool, "_rows", side_effect=rows):
            result = self.run_search([self.point(2)])
        self.assertNotIn(2, [r["corpus_id"] for r in result["recommended_documents"]])

    def test_query_budget_and_empty_inputs(self):
        """验证推荐查询输入预算和空材料的边界处理。"""
        self.assertEqual(search_queries([]), [])
        self.assertLessEqual(
            len(
                search_queries(
                    [{"title": f"Topic number {i} {chr(65 + i) * 5}"} for i in range(20)]
                )
            ),
            5,
        )

    def test_recall_filters_exclude_selected_documents_and_validate_partial_search(self):
        """验证分路过滤排除已选文献，部分检索失败仍执行来源核验。"""
        client = SimpleNamespace(query_points=lambda **kwargs: None)
        with patch.object(client, "query_points") as query:
            query.return_value.points = []
            self.tool._vector(client, [0.1], {1, 2})
            filters = query.call_args.kwargs["query_filter"]
            self.assertEqual(filters.must_not[0].match.any, [1, 2])
            self.assertEqual(filters.must[0].match.value, "qwen3.7-text-embedding-flash")
        with patch("app.foreign_integration.recommendations.OpenSearch") as constructor:
            constructor.return_value.search.return_value = {"hits": {"hits": []}}
            self.tool._lexical("双减教师负担", {1, 2})
            body = constructor.return_value.search.call_args.kwargs["body"]
            self.assertEqual(body["query"]["bool"]["must_not"], [{"terms": {"corpus_id": [1, 2]}}])
            constructor.return_value.search.return_value = {"timed_out": True}
            with self.assertRaises(RuntimeError):
                self.tool._lexical("双减", {1})

    def test_graph_runs_recommendations_in_parallel_and_failure_is_nonfatal(self):
        """验证推荐与综合并行，推荐失败不阻止主阅读回答发布。"""
        async def check(fail):
            """用互等事件验证两条分支并行，并分别断言推荐成功和失败终态。"""
            entered = asyncio.Event()
            synth_started = asyncio.Event()

            class Materials(FakeMaterials):
                """让推荐等待综合开始，可按测试参数注入独立失败。"""
                async def recommend(inner, readings):
                    """记录推荐已启动，等待综合阶段后返回候选或抛出测试错误。"""
                    entered.set()
                    await synth_started.wait()
                    if fail:
                        raise RuntimeError("推荐检索失败")
                    return {
                        "recommended_documents": [{"corpus_id": 99, "title": "相关文献"}],
                        "recommendation_status": "SUCCESS",
                    }

            class Gateway(FakeGateway):
                """在概览调用处与推荐建立事件屏障，检验确实并行。"""
                async def structured(inner, instruction, schema, **kwargs):
                    """标记综合启动并等待推荐进入，再复用固定模型结果。"""
                    if schema.__name__ == "ReadingOverview":
                        synth_started.set()
                        await entered.wait()
                    return await super().structured(instruction, schema, **kwargs)

            steps = []

            async def step(name, state, operation):
                """执行并记录完成节点，供最终发布顺序断言。"""
                result = await operation()
                steps.append(name)
                return result

            result = await asyncio.wait_for(
                LiteratureGraph(Materials(), Gateway(), "测试", step).run({"task": "ANALYZE"}), 3
            )
            self.assertTrue(result["final_answer"])
            self.assertEqual(steps[-1], "FinalizeReading")
            payload = result["literature_result"]
            self.assertEqual(payload["recommendation_status"], "FAILED" if fail else "SUCCESS")
            self.assertEqual(len(payload["recommended_documents"]), 0 if fail else 1)

        asyncio.run(check(False))
        asyncio.run(check(True))
