"""声明Quill研究问题生成（RQ）图的 LangGraph State。

复用Quill公共检索管线状态；分析草稿与结果字段的类型替换为 RQ 阶段契约。
"""

from __future__ import annotations

from app.agents.quill_schemas import QuillRQResult, RQDraft
from app.agents.quill_state import QuillState


class QuillRQState(QuillState, total=False):
    """保存Quill从研究主题输入到 RQ 结果发布的完整图状态。"""

    stage: str
    analysis_draft: RQDraft
    rq_result: QuillRQResult
