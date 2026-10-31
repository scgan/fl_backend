"""声明所有 Agent 共用并可由专业工作流扩展的 LangGraph State 契约。

公共字段覆盖固定身份与资源上下文、路由审计、检索 Evidence/Citation、审核状态
及最终发布结果。启明、智鉴、溯知等业务字段由各自 State 继承后声明，避免公共层
反向依赖具体专业 Agent，也防止无关节点误用其他工作流的内部状态。
"""

from __future__ import annotations

from typing import TypedDict

from app.retrieval.schemas import CitationRead, EvidenceRead


class AgentState(TypedDict, total=False):
    """保存所有 Agent 图共享的上下文、证据、审核结果和发布结果。

    ``total=False`` 允许节点逐步补充字段，但并不表示所有字段都可随意缺失：
    AuthenticateContext 会检查身份与资源字段，后续节点只读取其上游已经产出的字段。
    State 只在受控图内流转；可持久化内容会由 Service 提取脱敏摘要，而不是整份落库。
    """

    # 客户端生成的幂等键，贯穿 API、AgentRun 与图执行，用于识别同一次请求。
    request_id: str
    # 创建 Run 时固定的用户主键字符串，禁止节点根据前端输入重新切换身份。
    user_id: str
    # 当前 Run 所属会话主键；最终 Assistant Message 必须写回该会话。
    conversation_id: str
    # 用户原始问题，是检索和生成的唯一问题来源，不从历史消息反向推断。
    # 说明：追问（如「继续刚刚的任务」「把它改成800字」）会先由 Service 层
    # 结合 history 改写成本轮的原始问题，再写入该字段，因此这里仍然是自包含问题。
    query: str
    # 同一会话内此前的问答对（按时间正序，仅 user/assistant 两角色），
    # 只用于指代解析与追问补全，不能替代本轮重新检索到的 Evidence。
    history: list[dict[str, str]]
    # 检索查询处理器检测出的 zh/en 等语言编码，用于语言解析和审计。
    original_language: str
    # 模型从用户自然语言解析出的 ZH/EN/BILINGUAL 输出语言；解析前为 AUTO。
    output_language: str
    # 创建 Run 时已授权并固定的知识库主键字符串。
    knowledge_base_id: str
    # Run 创建时冻结的知识库版本；检索返回版本不一致时必须安全失败。
    knowledge_base_version: int
    # IntentClassifier 产出的 SIMPLE_QA、COMPARE 等规范意图编码。
    intent: str
    # TaskPlanner 生成的有序任务描述；仅保存结构信息，不保存完整证据正文。
    task_plan: list[dict[str, object]]
    # 当前实际处理节点所属 Agent 编码，例如 QIMING 或 GENERAL_RAG。
    current_agent: str
    # 从总助到执行 Agent 的可审计路由链，按发生顺序记录。
    agent_route: list[str]
    # 传给检索 Tool 的结构化过滤条件；未指定筛选条件时使用空字典。
    retrieval_filters: dict[str, object]
    # 查询处理器生成的中英文扩展查询，用于运行审计而非直接展示全文。
    expanded_queries: dict[str, str]
    # 混合检索进入重排前的候选数量，用于解释召回规模。
    retrieved_candidates: int
    # 通过权限、READY 白名单与质量门槛的真实证据，顺序即 Citation 编号基础。
    evidence: list[EvidenceRead]
    # 多 Agent 节点共享的结构化中间结果容器，不允许存放密钥。
    intermediate_results: dict[str, object]
    # 与 Evidence 一一映射的真实文档、Chunk、页码和鉴权文件定位。
    citations: list[CitationRead]
    # 最终回答文本实际使用的 Citation 编号；未引用的检索证据仍保留在 Run Evidence 中。
    used_citation_indices: list[int]
    # CitationValidator 核验通过且允许写入回答消息的引用子集。
    validated_citations: list[CitationRead]
    # Reviewer 和 Citation Validator 均通过后可由 Service 原子发布的回答。
    final_answer: str
    # HIGH/MEDIUM/LOW/INSUFFICIENT 的非概率证据等级。
    evidence_level: str
    # 文档数、语言覆盖、年份范围和来源多样性等可解释统计。
    evidence_summary: dict[str, object]
    # ResultReviewer 的 PASS/FAIL 结果；非 PASS 时禁止进入最终发布。
    reviewer_status: str
    # CitationValidator 的 PASS/FAIL 结果；确保引用可回到真实来源。
    citation_validation_status: str
    # FAST/DEEP 运行模式，控制证据上限，不绕过任何权限或审核节点。
    mode: str
    # 当前图级重试次数；所有重试策略都必须有明确上限。
    retry_count: int
    # 供下游节点汇总的安全错误文本，不应包含密钥或基础设施响应体。
    errors: list[str]
