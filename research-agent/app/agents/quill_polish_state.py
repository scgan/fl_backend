"""声明Quill划词润改（POLISH）图的 LangGraph State。

划词润改属于纯创作操作：不检索知识库、不产生 Citation，状态只承载选区
文本、上下文、操作编码和润改结果。
"""

from __future__ import annotations

from app.agents.quill_schemas import PolishAction, QuillPolishResult
from app.agents.state import AgentState


class QuillPolishState(AgentState, total=False):
    """保存Quill从划词选区输入到润改结果发布的完整图状态。"""

    # 用户在编辑器中选中的文本；Run 的 query 字段即选区原文。
    selected_text: str
    # 选区前后的编辑器上下文，用于续写衔接；可为空。
    context_before: str
    context_after: str
    # 本次执行的划词操作编码。
    action: PolishAction
    # Quill划词润改工作台读取的最终结构化结果。
    polish_result: QuillPolishResult
