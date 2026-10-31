"""验证启明统一分类、取消传播跳转及唯一活动 Prompt 的幂等迁移。"""

import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from pydantic import ValidationError
from sqlalchemy import create_engine, insert, select

from app.agents.catalog import QIMING_SYSTEM_PROMPT
from app.agents.qiming_graph import QimingGraph, build_intent_instruction
from app.agents.qiming_state import QimingState
from app.agents.schemas import QimingIntentClassification
from app.foreign_integration import migrate
from app.foreign_integration.schema import metadata, research_prompt


async def run_step(_name, _state, operation):
    """执行实际节点逻辑，不连接数据库。

    Args:
        _name: 节点名。
        _state: 当前状态。
        operation: 节点操作。

    Returns:
        节点结果。
    """
    return await operation()


class QimingRoutingTests(unittest.TestCase):
    """验证实际分类契约和拓扑均不包含自动或手动传播跳转。"""

    def make_graph(self, intent):
        """构造可控分类网关。

        Args:
            intent: 网关模拟标签。

        Returns:
            实际图与参数记录。
        """
        captured = {}

        async def structured(**kwargs):
            """记录分类参数并返回标签。

            Args:
                **kwargs: 网关参数。

            Returns:
                待本地校验的意图。
            """
            captured.update(kwargs)
            return SimpleNamespace(intent=intent)

        return QimingGraph(object(), SimpleNamespace(structured=structured),
                           QIMING_SYSTEM_PROMPT, run_step), captured

    def test_policy_and_writing_stay_in_qiming(self):
        """政策定义与介绍写作都能通过普通问答计划，不存在传播跳转节点。"""
        for query in ("什么是双减政策", "写给英国教师的双减政策介绍"):
            graph, captured = self.make_graph("SIMPLE_QA")
            state = QimingState(query=query)
            state.update(asyncio.run(graph._intent_classifier(state)))
            state.update(asyncio.run(graph._task_planner(state)))
            state.update(asyncio.run(graph._routing_decision(state)))
            self.assertIs(captured["schema"], QimingIntentClassification)
            self.assertEqual(state["task_plan"][0]["agent"], "QIMING")
            self.assertEqual(state["agent_route"], ["QIMING", "GENERAL_RAG"])
            self.assertEqual(graph._route_after_decision(state), "simple")
            self.assertNotIn("RongtongGraph", graph.graph.get_graph().nodes)

    def test_rejects_all_retired_labels(self):
        """网关不能通过旧标签恢复传播路由或跳转提示。"""
        for intent in (
            "POLICY_EXPLANATION", "CROSS_CULTURAL_COMMUNICATION", "MANUAL_COMMUNICATION",
        ):
            graph, _ = self.make_graph(intent)
            with self.subTest(intent=intent), self.assertRaises(ValidationError):
                asyncio.run(graph._intent_classifier(QimingState(query="什么是双减")))

    def test_compare_and_review_routes_are_preserved(self):
        """比较、综述和概念溯源仍走原专业分支。"""
        for intent, branch in (("COMPARE", "compare"), ("LITERATURE_REVIEW", "review"),
                               ("CONCEPT_TRACE", "review")):
            self.assertEqual(QimingGraph._route_after_decision(QimingState(intent=intent)), branch)

    def test_prompt_has_no_rongtong_guidance(self):
        """唯一系统提示和分类指令都不包含传播路由或融通提示。"""
        instruction = build_intent_instruction("什么是双减")
        self.assertIn("'什么是双减'→SIMPLE_QA", instruction)
        for retired in ("融通", "RONGTONG", "MANUAL_COMMUNICATION", "POLICY_EXPLANATION"):
            self.assertNotIn(retired, instruction + QIMING_SYSTEM_PROMPT)

    def test_migration_keeps_one_active_prompt_and_archives_history(self):
        """重复启动只更新当前配置，历史配置留存但不再可选。"""
        engine = create_engine("sqlite+pysqlite:///:memory:")
        metadata.create_all(engine)
        settings = SimpleNamespace(model_name="", model_base_url="")
        with (patch.object(migrate, "get_engine", return_value=engine),
              patch.object(migrate, "get_foreign_settings", return_value=settings)):
            migrate.migrate()
            with engine.begin() as connection:
                connection.execute(insert(research_prompt).values(
                    id=migrate.stable_id("prompt", "QIMING:3"),
                    agent_id=migrate.stable_id("agent", "QIMING"), version=3,
                    system_prompt="historical", status="ACTIVE", is_deleted=False))
            migrate.migrate()
            migrate.migrate()
        with engine.connect() as connection:
            rows = connection.execute(select(research_prompt.c.version,
                research_prompt.c.system_prompt, research_prompt.c.status).where(
                research_prompt.c.agent_id == migrate.stable_id("agent", "QIMING"))
                .order_by(research_prompt.c.version)).all()
        self.assertEqual(rows, [(0, QIMING_SYSTEM_PROMPT, "ACTIVE"), (3, "historical", "ARCHIVED")])
        engine.dispose()
