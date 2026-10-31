"""声明Quill写作方案（PLAN）图的 LangGraph State。

复用Quill公共检索管线状态；额外承载前序 RQ 产物内容与 PLAN 阶段契约。
"""

from __future__ import annotations

from typing import Any

from app.agents.quill_schemas import PlanDraft, QuillPlanResult
from app.agents.quill_state import QuillState


class QuillPlanState(QuillState, total=False):
    """保存Quill从研究主题输入到 PLAN 结果发布的完整图状态。"""

    stage: str
    # 可选的前序研究问题产物内容；来自同会话 RQ 阶段的 research_writing_artifact。
    rq_artifact: dict[str, Any] | None
    # 引用的前序产物 ID，回填到 PLAN 结果便于审计与前端链接。
    based_on_artifact_id: str | None
    analysis_draft: PlanDraft
    plan_result: QuillPlanResult
