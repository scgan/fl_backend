"""声明独立文献阅读与问答图使用的 LangGraph State。

继承研究 Agent 共用的运行和发布契约，补充所选文献、阅读材料、分析及多轮对话状态。
"""

from __future__ import annotations

from typing import Any

from app.agents.state import AgentState
from app.retrieval.schemas import CitationRead, EvidenceRead


class LiteratureState(AgentState, total=False):
    """保存文献阅读从材料加载到分析或追问结果发布的完整图状态。"""

    # ANALYZE 生成首次阅读分析，ASK 基于同一文献集合继续提问。
    task: str
    focus_corpus_id: int | None
    # 已成功的对话轮次与首次分析摘要，用于理解追问指代。
    history: list[dict[str, Any]]
    previous_analysis: dict[str, Any] | None
    # 固定选择范围中的文献元数据及完整正文片段。
    documents: list[dict[str, Any]]
    materials: list[EvidenceRead]
    material_citations: list[CitationRead]
    # 逐篇固定模板分析与前端读取的最终结构化结果。
    readings: list[dict[str, Any]]
    literature_result: dict[str, Any]
    recommended_documents: list[dict[str, Any]]
    recommendation_status: str
    library_recommendation_status: str
    web_recommendation_status: str
    web_recommendation_details: dict[str, Any]
