"""声明雕龙修改完善（REVISE）图的 LangGraph State。

复用雕龙公共检索管线状态；原始待改文本经 Run 的 query 字段冻结，
分析草稿与结果字段的类型替换为 REVISE 阶段契约。
"""

from __future__ import annotations

from app.agents.quill_schemas import QuillReviseResult, RevisedText
from app.agents.quill_state import QuillState


class QuillReviseState(QuillState, total=False):
    """保存雕龙从待改文本输入到修改结果发布的完整图状态。"""

    stage: str
    # 用户给出的修改要求，如扩写论证、补充引用、调整语气。
    revise_instruction: str
    analysis_draft: RevisedText
    revise_result: QuillReviseResult
