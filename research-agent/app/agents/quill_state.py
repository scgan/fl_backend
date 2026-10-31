"""声明Quill论文写作图使用的 LangGraph State。

该状态继承所有 Agent 共用的运行、Evidence 和发布契约，并补充工作阶段、
研究主题、输出规格、查询计划、规范文献、理论分析草稿及Quill结果字段。
"""

from __future__ import annotations

from typing import Any

from app.agents.quill_schemas import (
    QuillResult,
    TheoryAnalysisDraft,
    TheoryConcept,
    TheoryQueryPlan,
)
from app.agents.state import AgentState
from app.retrieval.schemas import CitationRead, EvidenceRead


class QuillState(AgentState, total=False):
    """保存Quill从研究主题输入到 THEORY 结果发布的完整图状态。"""

    # 用户提交并冻结的工作阶段；Phase 1 仅接受 THEORY。
    stage: str
    # 深度思考开关：图内全部模型调用统一读取；未勾选时按关闭值调用。
    thinking: bool
    # 用户提交并冻结的研究主题、可选年份范围与输出规格。
    research_topic: str
    # 用户明确确认的篇幅、层级和侧重，仅作创作约束，不能替代事实证据。
    writing_requirements: str
    year_from: int | None
    year_to: int | None
    citation_style: str
    literature_limit: int
    # 模型解析的规范概念与有限检索表达计划。
    concept_plan: TheoryConcept
    query_plan: TheoryQueryPlan
    # 去重前的原始证据和引用，供 EvidenceDedup 确定性合并。
    raw_evidence: list[EvidenceRead]
    raw_citations: list[CitationRead]
    # 从 Evidence 规范化得到的文献级元数据。
    documents: list[dict[str, Any]]
    # 理论综合节点生成并等待引用校验的结构化草稿。
    analysis_draft: TheoryAnalysisDraft
    # 理论生成业务轮次及当前校验决策；失败原因只包含确定性规则文案。
    analysis_attempts: int
    analysis_validation_status: str
    analysis_validation_error: str
    # Quill专业工作台读取的最终结构化结果。
    references: list[str]
    quill_result: QuillResult
