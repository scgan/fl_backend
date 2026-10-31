"""声明四个智能体共用的检索契约，权限与召回由 Foreign 适配层实现。"""

from __future__ import annotations

import uuid
from typing import Any, Protocol

from app.agents.ai_assistant_schemas import AssistantSearchPlan
from app.retrieval.schemas import RetrievalFilters, RetrievalResponse


class RetrievalTool(Protocol):
    """图仅依赖受控检索接口，不再依赖旧用户 ORM 和独立检索服务。"""

    async def search(
        self,
        *,
        knowledge_base_id: uuid.UUID,
        query: str,
        mode: str,
        filters: RetrievalFilters | None = None,
        evidence_limit: int | None = None,
    ) -> RetrievalResponse:
        """在运行绑定的身份和知识库范围内召回可核验证据。

        Args:
            knowledge_base_id: 运行固定的知识库标识。
            query: 原始问题或受控扩展查询。
            mode: FAST 或 DEEP。
            filters: 只收窄查询范围的筛选条件。
            evidence_limit: 可选证据数量上限。

        Returns:
            包含真实证据、引用及知识库版本的检索响应。
        """
        ...


class AssistantRetrievalTool(RetrievalTool, Protocol):
    """AI 入口额外支持题录发现，其他专业 Agent 继续只依赖证据检索协议。"""

    async def discover(self, plan: AssistantSearchPlan) -> dict[str, Any]:
        """在固定授权范围内按结构化条件返回首批题录及总数。"""
        ...
