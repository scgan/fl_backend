"""定义融通图私有运行字段；公共 Evidence/Citation/发布上下文继承 AgentState。"""

from app.agents.rongtong_schemas import (
    CommunicationDraft,
    ConceptFacts,
    CulturalRisk,
    PolicyFacts,
    PolicyRequestPlan,
    RongtongResult,
)
from app.agents.state import AgentState


class RongtongState(AgentState, total=False):
    """政策事实到跨文化传播的独立状态，不向启明公共层扩散业务字段。"""

    # 用户原始自然语言在本专业图内解析为传播规格。
    policy_or_concept: str
    # 必选语言与实际采用规格。
    target_language: str
    policy_plan: PolicyRequestPlan
    target_audience: str
    style: str
    desired_length: int
    # 结构提取节点先抽取来源事实，再独立转换为目标语言；后续传播节点不重复翻译。
    insufficient_evidence: bool
    localized_topic: str
    policy_facts: PolicyFacts | ConceptFacts
    # 结构提取业务轮次及引用范围校验结果；失败原因只保留稳定规则文案。
    structure_attempts: int
    structure_validation_status: str
    structure_validation_error: str
    cultural_risks: list[CulturalRisk]
    # 确定性受众表达规则及只包含传播正文的待审核草稿。
    adaptation: str
    communication_draft: CommunicationDraft
    communication_text: str
    # 传播正文业务轮次与 Reviewer 决策；失败原因只来自确定性规则。
    communication_attempts: int
    reviewer_error: str
    # 通过全部发布门槛的唯一专业结果。
    rongtong_result: RongtongResult
