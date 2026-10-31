"""定义启明运行请求、详情、步骤、Evidence 与快照 API 契约。

请求模型约束幂等键、知识库和运行模式；响应模型只暴露前端恢复、
SSE 重放和 Evidence Panel 所需的安全字段。完整系统 Prompt、模型密钥和未审核
草稿不属于任何对外响应。
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from app.retrieval.schemas import CitationRead, EvidenceRead

# FAST/DEEP 只改变证据上限，二者都必须经过同一安全和引用校验流程。
RunMode = Literal["FAST", "DEEP"]


class QimingIntentClassification(BaseModel):
    """启明唯一的分类契约：普通问答、比较、综述、溯源和能力边界。"""

    intent: Literal[
        "SIMPLE_QA", "COMPARE", "LITERATURE_REVIEW", "CONCEPT_TRACE", "UNSUPPORTED",
    ] = Field(description="政策问答和一般介绍选 SIMPLE_QA，明确比较或综述才进入专业流程")


class ParsedLanguagePreference(BaseModel):
    """保存启明从用户问题中解析出的输出语言。"""

    output_language: Literal["ZH", "EN", "BILINGUAL"] = Field(
        description=(
            "最终回答语言；未明确指定时根据问题主要语言选择 ZH 或 EN，"
            "只有用户明确要求中英双语时才选择 BILINGUAL"
        )
    )


class QimingRunCreate(BaseModel):
    """校验一次启明基础问答的幂等输入。"""

    # 客户端生成的幂等键；网络重试必须复用它，避免双击产生重复 Run。
    request_id: str = Field(min_length=8, max_length=128)
    # 当前问答严格限定在一个已授权知识库中。
    knowledge_base_id: uuid.UUID
    # 继续已有对话时提供；为空则根据首个问题创建新对话。
    conversation_id: uuid.UUID | None = None
    # 用户原始问题，完整内容只保存到自己的 ConversationMessage。
    query: str = Field(min_length=2, max_length=4000)
    # FAST 使用较少 Evidence，DEEP 在同一安全流程中扩大证据上限。
    mode: RunMode = "FAST"

    @field_validator("query")
    @classmethod
    def normalize_query(cls, value: str) -> str:
        """清理问题两端空白但保留用户内部格式。

        Args:
            value: 用户提交的问题。

        Returns:
            规范化后的非空问题。
        """
        return value.strip()


class ZhijianRunCreate(BaseModel):
    """校验一句话智鉴任务的最小业务输入和幂等上下文。"""

    # 客户端幂等键属于运行治理字段，不是研究参数。
    request_id: str = Field(min_length=8, max_length=128)
    # 当前页面或会话已经选定的知识库上下文。
    knowledge_base_id: uuid.UUID
    # 继续已有同知识库会话时提供。
    conversation_id: uuid.UUID | None = None
    # 用户唯一需要输入的自然语言比较请求。
    query: str = Field(min_length=2, max_length=4000)

    @field_validator("query")
    @classmethod
    def normalize_query(cls, value: str) -> str:
        """清理问题两端空白但不改写内部语义。

        Args:
            value: 用户提交的比较请求。

        Returns:
            规范化后的非空请求。
        """
        return value.strip()


class AgentRunAccepted(BaseModel):
    """返回已创建或通过幂等键复用、可连接 SSE 的 Agent Run。"""

    # 本次请求是否真正创建了 Run；只有为真时 API 才允许调度后台执行。
    created: bool
    # 后端生成的运行主键，是读取状态、步骤、证据和 SSE 的统一标识。
    run_id: uuid.UUID
    # 原样返回客户端幂等键，便于调用方确认重试复用了同一运行。
    request_id: str
    # 新建或续接的会话主键，前端据此切换到稳定会话路由。
    conversation_id: uuid.UUID
    # 本次运行被限制到的知识库主键。
    knowledge_base_id: uuid.UUID
    # 创建 Run 时冻结的知识库版本号。
    knowledge_base_version: int
    # 接受时通常为 PENDING；幂等复用时也可能是已有终态。
    status: str
    # 当前 Run 的相对 SSE 地址，仍需携带用户认证信息访问。
    stream_url: str


class AgentStepRead(BaseModel):
    """向运行详情和 SSE 投影一个经过脱敏的 LangGraph 节点记录。"""

    # 持久化步骤主键。
    id: uuid.UUID
    # 与图定义一致的稳定节点名，例如 GeneralRAG、ResultReviewer。
    node_name: str
    # 同一 Run 内从 1 开始严格递增的执行顺序。
    sequence: int
    # RUNNING/SUCCESS/FAILED/SKIPPED 节点状态。
    status: str
    # 节点开始时间，使用带时区 UTC 时间戳。
    started_at: datetime
    # 节点结束时间；仍在运行时为空。
    finished_at: datetime | None
    # 仅含问题长度、证据数量等安全输入摘要，不包含完整问题或证据。
    input_summary: dict[str, object]
    # 仅含路由、数量和审核结果等安全输出摘要。
    output_summary: dict[str, object]
    # 节点实际重试次数；任何节点都不得执行无限重试。
    retry_count: int
    # 可展示的脱敏错误说明；成功节点为空。
    error_message: str | None


class ProfessionalResultRead(BaseModel):
    """描述可版本演进且只保存一份的专业 Agent 结果。"""

    # ZHIJIAN_COMPARISON 等稳定结果类型。
    result_type: str
    # 对应 payload 的正整数 Schema 版本；历史裸数据使用 0。
    schema_version: int = Field(ge=0)
    # 专业页面恢复所需的结构化快照。
    payload: dict[str, object]


class AgentRunRead(BaseModel):
    """返回一次固定 KB Version、模型与 Prompt 版本的可审计启明运行。"""

    # Agent Run 主键。
    id: uuid.UUID
    # 客户端幂等键。
    request_id: str
    # 所属会话主键。
    conversation_id: uuid.UUID
    # 执行 Agent 稳定编码与面向用户的名称。
    agent_code: str
    # Agent 的中文显示名称。
    agent_name: str
    # 用户进入运行时选择的入口模块；保留 agent_code 兼容旧客户端。
    entry_agent_code: str
    # 实际生成专业结果的模块，用于历史筛选和详情跳转。
    executor_agent_code: str
    # Run 固定的知识库标识、名称和版本。
    knowledge_base_id: uuid.UUID
    # 创建 Run 时的知识库显示名称。
    knowledge_base_name: str
    # 创建 Run 时冻结的知识库版本。
    knowledge_base_version: int
    # PENDING/RUNNING/SUCCESS/FAILED/CANCELLED 运行状态。
    status: str
    # 规范意图；分类节点尚未完成时为空。
    intent: str | None
    # 从 QIMING 到实际处理 Agent 的有序路由。
    route: list[str]
    # FAST/DEEP 模式与输出语言偏好。
    mode: str
    # AUTO/ZH/EN/BILINGUAL 输出语言偏好。
    output_language: str
    # 查询处理器检测到的原始语言；检索前为空。
    original_language: str | None
    # 仅在双重发布闸门通过后存在的最终回答。
    final_answer: str | None
    # 非概率证据等级及其可解释覆盖统计。
    evidence_level: str | None
    # 文档、语言、年份与来源覆盖统计。
    evidence_summary: dict[str, object]
    # 两道发布校验的状态；节点未执行时为空。
    reviewer_status: str | None
    # Citation 与真实来源映射的校验状态。
    citation_validation_status: str | None
    # 运行开始、结束和创建时间；排队中可能尚无开始/结束时间。
    started_at: datetime | None
    # 运行进入终态的时间；排队或运行中为空。
    finished_at: datetime | None
    # 稳定错误码和脱敏错误消息；成功运行均为空。
    error_code: str | None
    # 可安全展示的错误说明，不包含供应商响应体或密钥。
    error_message: str | None
    # Run 创建时间，用于会话历史排序。
    created_at: datetime
    # 统一专业结果外壳；payload 由 result_type 对应的版本化 Schema 解释。
    professional_result: ProfessionalResultRead | None = None
    # 智鉴兼容投影；从专业结果外壳读取，旧快照不再被当前严格 Schema 拒绝。
    zhijian_result: dict[str, object] | None = None
    # 溯知兼容投影；新版客户端优先读取统一 professional_result 外壳。
    suzhi_result: dict[str, object] | None = None
    # 融通版本化专业结果的兼容投影，实际存储仅 professional_result 一份。
    rongtong_result: dict[str, object] | None = None


class AgentRunHistoryRead(BaseModel):
    """返回当前用户拥有的专业模块历史列表轻量摘要。"""

    # 详情恢复使用的 Agent Run 主键。
    id: uuid.UUID
    # 所属统一会话主键和列表标题。
    conversation_id: uuid.UUID
    conversation_title: str
    # 本次运行的原始问题；摘要在知识库权限失效后置空，避免泄露回答和证据内容。
    query: str
    summary: str
    # 入口和实际执行模块；二者在启明委派时不同。
    entry_agent_code: str
    executor_agent_code: str
    # 专业结果类型与 Schema 版本；失败或普通结果为空。
    result_type: str | None
    schema_version: int | None
    # 当前运行终态或执行状态。
    status: str
    # 运行冻结的知识库标识、显示名称、当前可访问状态与版本。
    knowledge_base_id: uuid.UUID
    knowledge_base_name: str | None
    knowledge_base_accessible: bool
    knowledge_base_version: int
    # 运行创建时间，用于历史倒序排列。
    created_at: datetime


class AgentRunHistoryList(BaseModel):
    """返回分页的专业模块运行历史。"""

    # 本页轻量历史记录。
    items: list[AgentRunHistoryRead]
    # 筛选模块下当前用户的记录总数。
    total: int
    # 从 1 开始的当前页和单页上限。
    page: int
    page_size: int


class AgentRunDeleteResult(BaseModel):
    """返回一次专业模块 Run 真删除的实际影响范围。"""

    # 已删除的运行主键与其原会话主键。
    run_id: uuid.UUID
    conversation_id: uuid.UUID
    # 实际执行模块编码，用于前端校验历史列表来源。
    executor_agent_code: str
    # 随 Run 精确删除的已关联 USER/ASSISTANT 消息数。
    deleted_message_count: int
    # 删除最后一个 Run 时是否同步清理了空会话和旧版游离消息。
    conversation_deleted: bool


class AgentEvidenceRead(BaseModel):
    """同时返回 Agent Run 证据和对应真实 Citation。"""

    # 包含来源元数据与完整 Chunk 内容的检索证据。
    evidence: EvidenceRead
    # 与证据序号一致、可打开鉴权 PDF 页码的引用。
    citation: CitationRead
    # true 表示来源已在当前文献集合中软删除，历史摘录保留但原文入口不可用。
    source_deleted: bool = False
    # 智鉴运行时 A/B/SHARED 归属；普通问答为空。
    assignment: str | None = None
    supports_a: bool | None = None
    supports_b: bool | None = None
    relevance: str | None = None
    retrieval_branches: list[str] = Field(default_factory=list)
    # 模型确认的对象×维度支持关系；普通问答为空。
    cell_supports: list[dict[str, object]] = Field(default_factory=list)
    # Candidate 的对象×维度召回轨迹，不能替代语义归属。
    retrieval_contexts: list[dict[str, object]] = Field(default_factory=list)


class AgentRunEvidenceList(BaseModel):
    """返回固定知识库版本下的有序 Evidence/Citation 集合。"""

    # 所属 Run 与其冻结的知识库版本，用于前端防止跨运行混用证据。
    run_id: uuid.UUID
    # 该证据集合对应的冻结知识库版本。
    knowledge_base_version: int
    # 按 Citation 序号升序排列的证据与引用对。
    items: list[AgentEvidenceRead]
