"""定义融通自然语言请求、证据事实、传播段落和审核结果契约。"""

import uuid
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

OutputLanguage = Literal["ZH", "EN", "BILINGUAL"]
Audience = Literal[
    "INTERNATIONAL_RESEARCHER",
    "FOREIGN_TEACHER",
    "INTERNATIONAL_STUDENT",
    "POLICY_MAKER",
    "MEDIA",
    "GENERAL_PUBLIC",
]
Style = Literal["ACADEMIC", "POPULAR", "MEDIA", "BRIEFING"]


class RongtongRunCreate(BaseModel):
    """提交自然语言传播请求，受众、风格和篇幅统一从原话识别。"""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    request_id: str = Field(min_length=8, max_length=128, description="客户端幂等键")
    knowledge_base_id: uuid.UUID = Field(description="当前授权知识库")
    conversation_id: uuid.UUID | None = Field(default=None, description="可续接的同库会话")
    policy_or_concept: str = Field(min_length=2, max_length=4000, description="自然语言传播请求")
    target_language: OutputLanguage = Field(description="必选输出语言，不由模型覆盖")


class PolicyRequestPlan(BaseModel):
    """识别政策/概念类型与传播规格，不在检索前生成研究事实。"""

    subject_type: Literal["POLICY", "CONCEPT"] = Field(
        description="LLM 根据问题核心判断政策或概念，不由前端猜测"
    )
    topic: str = Field(min_length=2, max_length=200, description="规范政策或概念主题")
    target_audience: Audience = Field(description="原话指定受众，否则默认国际研究者")
    style: Style | None = Field(default=None, description="原话指定风格，否则使用受众默认")
    desired_length: int | None = Field(
        default=None, description="原话目标长度，图内限制为 100–5000"
    )


class PolicyFact(BaseModel):
    """一条不可因受众变化而改写含义的有来源事实。"""

    text: str = Field(
        min_length=2,
        max_length=2000,
        description="按提问语言抽取的 Evidence 事实，推断必须明确标识",
    )
    citation_ids: list[int] = Field(min_length=1, max_length=10, description="本次运行真实引用编号")


class PolicyFacts(BaseModel):
    """政策结构与影响的共同事实底座；无证据的分区保留为空。"""

    background: list[PolicyFact] = Field(default_factory=list, max_length=4, description="政策背景")
    objectives: list[PolicyFact] = Field(default_factory=list, max_length=4, description="政策目标")
    mechanisms: list[PolicyFact] = Field(
        default_factory=list, max_length=4, description="治理机制及工具"
    )
    stakeholders: list[PolicyFact] = Field(
        default_factory=list, max_length=4, description="利益相关者"
    )
    impacts: list[PolicyFact] = Field(
        default_factory=list, max_length=4, description="实施影响及推断边界"
    )
    international_comparison: list[PolicyFact] = Field(
        default_factory=list, max_length=4, description="有证据的国际比较，无资料时不得补全"
    )


class ConceptFacts(BaseModel):
    """概念解读的共同事实底座，不套用政策目标、利益群体或实施效果。"""

    definition: list[PolicyFact] = Field(default_factory=list, max_length=4, description="概念定义")
    boundaries: list[PolicyFact] = Field(default_factory=list, max_length=4, description="概念边界")
    characteristics: list[PolicyFact] = Field(
        default_factory=list, max_length=4, description="核心特征及原理"
    )
    applications: list[PolicyFact] = Field(
        default_factory=list, max_length=4, description="适用语境"
    )
    distinctions: list[PolicyFact] = Field(
        default_factory=list, max_length=4, description="与相关概念的区别，不补造比较结论"
    )


class PolicyInterpretationDraft(BaseModel):
    """政策结构提取节点生成的目标语言主题与政策解读。"""

    localized_topic: str = Field(
        min_length=2, max_length=400, description="指定输出语言的规范政策主题"
    )
    facts: PolicyFacts = Field(description="指定输出语言的政策事实分区")


class ConceptInterpretationDraft(BaseModel):
    """概念结构提取节点生成的目标语言主题与概念解读。"""

    localized_topic: str = Field(
        min_length=2, max_length=400, description="指定输出语言的规范概念主题"
    )
    facts: ConceptFacts = Field(description="指定输出语言的概念事实分区")


class CulturalRisk(BaseModel):
    """有证据背景的潜在误解及解释建议，不把传播建议包装成实证结果。"""

    misunderstanding: str = Field(
        min_length=2, max_length=1000, description="潜在误读，明确为可能性"
    )
    explanation: str = Field(min_length=2, max_length=2000, description="基于来源的语境澄清")
    citation_ids: list[int] = Field(min_length=1, max_length=10, description="支持语境的真实引用")


class CulturalRiskDraft(BaseModel):
    """跨文化检查节点的受控输出。"""

    risks: list[CulturalRisk] = Field(
        default_factory=list, max_length=6, description="误解风险列表"
    )


class CommunicationUnit(BaseModel):
    """传播生成节点返回的带真实句级 Citation 的中英文段落。"""

    text_zh: str = Field(
        default="", max_length=8000, description="带半角 [n] 句级引用的中文表达，ZH/BILINGUAL 必填"
    )
    text_en: str = Field(
        default="", max_length=8000, description="带半角 [n] 句级引用的英文表达，EN/BILINGUAL 必填"
    )


class CommunicationDraft(BaseModel):
    """传播生成节点产生的受众适配正文，不重复生成解读与文化解释。"""

    units: list[CommunicationUnit] = Field(
        min_length=1, max_length=30, description="带句级引用的目标语言传播段落"
    )


class RongtongResult(BaseModel):
    """历史可恢复的政策或概念传播快照，包含目标语言解读、文化解释及正文。"""

    status: Literal["SUCCESS", "INSUFFICIENT_EVIDENCE"] = Field(description="业务结果状态")
    subject_type: Literal["POLICY", "CONCEPT"] = Field(
        default="POLICY", description="输入理解节点识别类型；默认值仅兼容旧政策快照"
    )
    policy_or_concept: str = Field(description="指定输出语言的规范政策或概念主题")
    target_language: OutputLanguage = Field(description="用户指定输出语言")
    target_audience: Audience = Field(description="实际采用受众")
    style: Style = Field(description="实际采用风格")
    desired_length: int = Field(description="目标长度，不承诺精确计数")
    facts: PolicyFacts | ConceptFacts = Field(description="按识别类型组织的目标语言共同事实底座")
    cultural_risks: list[CulturalRisk] = Field(description="潜在误解及跨文化解释")
    communication_text: str = Field(description="通过语义及引用审核的目标语言正文")
    limitations: list[str] = Field(description="来源、推断及篇幅限制")
    evidence_level: str = Field(description="可解释证据等级")
    evidence_summary: dict[str, object] = Field(description="证据覆盖统计")
