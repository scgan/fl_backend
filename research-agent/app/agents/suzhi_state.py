"""声明溯知文献溯源与综述图使用的 LangGraph State。

该状态继承所有 Agent 共用的运行、Evidence 和发布契约，并补充研究范围、概念与
查询计划、规范文献、分析草稿、溯源图及综述结果字段。
"""

from __future__ import annotations

from typing import Any

from app.agents.state import AgentState
from app.agents.suzhi_schemas import (
    LiteratureAnalysisDraft,
    ProvenanceGraph,
    ReviewConcept,
    ReviewDraft,
    ReviewQueryPlan,
    SuzhiResult,
)
from app.retrieval.schemas import CitationRead, EvidenceRead


class SuzhiState(AgentState, total=False):
    """保存溯知从研究范围输入到可追溯综述发布的完整图状态。"""

    # 用户提交并冻结的研究主题、可选概念、年份范围与输出规格。
    research_topic: str
    requested_concept: str | None
    year_from: int | None
    year_to: int | None
    review_length: str
    citation_style: str
    # 模型解析的规范概念与有限检索表达计划。
    concept_plan: ReviewConcept
    query_plan: ReviewQueryPlan
    # 去重前的原始证据和引用，供 EvidenceDedup 确定性合并。
    raw_evidence: list[EvidenceRead]
    raw_citations: list[CitationRead]
    # 从 Evidence 规范化得到的文献级元数据。
    documents: list[dict[str, Any]]
    # 同次模型调用生成的分析草稿和溯源图。
    analysis_draft: LiteratureAnalysisDraft
    provenance_graph: ProvenanceGraph
    # 分析生成业务轮次及当前校验决策；失败原因只包含确定性规则文案。
    analysis_attempts: int
    analysis_validation_status: str
    analysis_validation_error: str
    # 等待 Reviewer 和 Citation Validator 审核的综述与参考文献。
    review_draft: ReviewDraft
    # 正文模型生成业务轮次及 Reviewer 的可重试失败原因。
    review_attempts: int
    reviewer_error: str
    # 模型连续未生成合法 Citation 时，是否改用已审核结构化事实确定性组织正文。
    review_fallback_used: bool
    references: list[str]
    # 溯知专业工作台读取的最终结构化结果。
    suzhi_result: SuzhiResult
