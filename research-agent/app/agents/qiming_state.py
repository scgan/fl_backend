"""声明启明 Supervisor 与通用问答图使用的 LangGraph State。

该状态在公共 Agent 运行契约上补充安全检查、复杂度和回答草稿字段，
并只接收专业子图的最终结果，不暴露智鉴、溯知工作流的内部处理中间态。
"""

from __future__ import annotations

from app.agents.state import AgentState
from app.agents.suzhi_schemas import SuzhiResult
from app.agents.zhijian_schemas import ZhijianResult


class QimingState(AgentState, total=False):
    """保存启明理解、规划、路由、通用问答及专业委派结果。"""

    # ComplexityClassifier 产出的 SIMPLE/COMPLEX，用于展示和分支编排。
    complexity: str
    # 深度思考开关：随运行冻结，GenerateAnswer 等模型调用统一读取；未开启时按关闭值调用。
    thinking: bool
    # 检索层在证据不足时给出的确定性说明，生成节点直接发布而不调用模型。
    retrieval_message: str | None
    # Model Gateway 生成且尚未通过双重发布闸门的普通问答草稿。
    draft_answer: str
    # SecurityCheck 是否命中提示词窃取或跨用户数据请求。
    security_blocked: bool
    # 智鉴子图通过发布闸门后的最终结果；启明不持有其内部比较过程字段。
    zhijian_result: ZhijianResult
    # 溯知子图通过发布闸门后的最终结果；启明不持有其文献分析内部过程字段。
    suzhi_result: SuzhiResult
