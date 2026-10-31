"""声明雕龙逐节起草（DRAFT）图的 LangGraph State。

复用雕龙公共检索管线状态；额外承载前序写作方案产物与目标章节，
分析草稿与结果字段的类型替换为 DRAFT 阶段契约。
"""

from __future__ import annotations

from typing import Any

from app.agents.quill_schemas import QuillDraftResult, SectionDraft
from app.agents.quill_state import QuillState


class QuillDraftState(QuillState, total=False):
    """保存雕龙从大纲章节输入到本节草稿发布的完整图状态。"""

    stage: str
    # 前序写作方案产物内容（同会话 PLAN 阶段的 research_writing_artifact）。
    plan_artifact: dict[str, Any] | None
    # 引用的写作方案产物 ID，回填到结果便于审计与前端链接。
    based_on_artifact_id: str | None
    # 本次起草的目标章节（大纲 section_id 与章节内容）。
    section_ref: str
    section: dict[str, Any] | None
    analysis_draft: SectionDraft
    draft_result: QuillDraftResult
