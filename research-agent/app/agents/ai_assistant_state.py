"""声明 AI 统一入口的 LangGraph State，继承公共运行、证据及发布契约。"""

from __future__ import annotations

from typing import Any, Literal

from app.agents.state import AgentState


class AssistantState(AgentState, total=False):
    """保存意图、题录搜索和启明问答各节点的状态增量。"""

    # 成功会话历史只用于理解指代，不能替代本轮检索证据。
    history: list[dict[str, Any]]
    # 独立检索问题与用户原问题分开，避免补全指代时丢失输出语言要求。
    retrieval_query: str
    # 服务根据请求参数给出的 GLOBAL/SELECTED/DATABASE 展示标记，实际授权范围由检索工具绑定。
    scope_type: Literal["GLOBAL", "SELECTED", "DATABASE", "KNOWLEDGE_BASE"]
    # 随 Run 冻结的模型思考开关。
    thinking: bool
    # 可发布的结构化结果，服务层将其放入已有 result_payload。
    assistant_result: dict[str, Any]
    # 模型提取的题录条件及 SQL 分页结果，独立于 QA 正文证据预算。
    search_plan: dict[str, Any]
    discovery: dict[str, Any]

    # 启明子图已通过自身发布闸门的回答，沿用原 AI conversation。
    qiming_answer: str
    # 启明内部的真实意图和专业结果，与入口 SEARCH/QA 标签分开，避免路由信息丢失。
    research_intent: str
    research_result: dict[str, Any]
    # 用户确认的写作参数与模型识别阶段分开，知识库范围在创建 Run 前冻结。
    writing_options: dict[str, Any]
    writing_stage: str | None
    # 纯主题与输出要求分开；阶段缺失先澄清，不自动选择提纲。
    research_topic: str
    writing_requirements: str
    needs_clarification: bool
    clarification_question: str
    section_ref: str
    writing_answer: str
    writing_result: dict[str, Any]
    # 保存实际使用的本轮引用编号，供阶段产物的导出与审计使用。
    used_citation_indices: list[int]
    # 独立于聊天字节预算的真实文献集合，只由服务从同会话成功Run恢复。
    reading_sources: list[dict[str, Any]]
    # 推荐问题按钮明确指定的追问操作，绕过自然语言任务分类。
    reading_options: dict[str, Any]
    reading_task: str
    reading_source_run_id: str | None
    reading_source_kind: str | None
    # 模型仅解释指代语义；服务按此决定勾选优先或明确篇目覆盖，不由模型生成ID。
    reading_reference: str
    reading_positions: list[int]
    reading_all_results: bool
    # 阅读分支输出及受控范围快照，继续保存在原AI会话中。
    reading_answer: str
    reading_result: dict[str, Any]
