"""定义智鉴自然语言解析、证据归属、比较分析和 API 输出结构。

这些模型是 ModelGateway 结构化输出与 LangGraph 状态之间的校验边界。所有引用
编号都必须在后续 Citation Validator 中映射到真实 AgentEvidence，不能由模型自报。
"""

from __future__ import annotations

import uuid
from typing import Annotated, Literal

from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_core import PydanticCustomError

Assignment = Literal["A", "B", "SHARED"]
Relevance = Literal["HIGH", "MEDIUM", "LOW", "IRRELEVANT"]
CoverageLevel = Literal["SUFFICIENT", "PARTIAL", "INSUFFICIENT"]
# 模型规划的比较维度必须是紧凑标题；长度约束同时进入 Qwen JSON Schema。
DimensionName = Annotated[
    str,
    Field(
        min_length=4,
        max_length=30,
        description="用于 A/B 比较的简短维度标题，不含序号、说明或首尾标点",
    ),
]
# 单次比较最多采用的业务维度；调整此常量会同步改变模型规划和运行时截断边界。
MAX_COMPARE_DIMENSIONS = 3


def normalize_dimension_title(value: str) -> str:
    """移除维度标题首尾的空白、标点和符号，保留标题内部连接符。

    Args:
        value: 模型或用户返回的原始维度标题。

    Returns:
        以中文、英文或数字开头和结尾的规范标题；没有有效文字时为空串。
    """
    start = 0
    end = len(value)
    while start < end and not value[start].isalnum():
        start += 1
    while end > start and not value[end - 1].isalnum():
        end -= 1
    return value[start:end]


class ParsedCompareRequest(BaseModel):
    """表示从一句自然语言请求中提取的比较任务。"""

    # 比较主题；缺失时留空并由验证节点要求自然语言澄清。
    topic: str = Field(
        default="",
        description="两个对象共同讨论的主题或问题；不能代替或抹去任一对象自身的识别限定",
    )
    # 保留用户真正对照的完整语义单元，对象可以是任意类型和粒度。
    object_a: str = Field(
        default="",
        description=(
            "用户要直接对照的第一项完整对象；保留识别该对象所必需的名称与限定，"
            "不强制抽成国家、人名、机构或某种固定类型；无法确定时留空"
        ),
    )
    object_b: str = Field(
        default="",
        description=(
            "用户要直接对照的第二项完整对象；保持其自身含义和粒度，"
            "不得套用第一项的名称或限定，不强行把两项改成同一类型；无法确定时留空"
        ),
    )
    # 可选起止年份；没有明确时间约束时均为空。
    year_from: int | None = Field(
        default=None,
        ge=1000,
        le=2100,
        description="用户明确要求的起始年份；没有明确时间约束时返回 null",
    )
    year_to: int | None = Field(
        default=None,
        ge=1000,
        le=2100,
        description="用户明确要求的结束年份；没有明确时间约束时返回 null",
    )
    # 兼容旧请求；ALL 不限来源语言，其他值为单一语言代码。
    language_filter: str = Field(
        default="ALL",
        description="兼容单一来源语言筛选；ALL 不限语言，否则填写语言代码，如 ZH、EN、JA",
    )
    # 兼容旧解析结果中的单一来源语种，新请求优先使用 source_language_codes。
    source_language_requirement: str | None = Field(
        default=None,
        description="兼容单一原始来源语言要求，例如 JA；未要求时返回 null",
    )
    # 多个代码表示允许这些语种的文献，不代表每种语言都已有可用证据。
    source_language_codes: list[str] = Field(
        default_factory=list,
        max_length=8,
        description=(
            "用户明确要求的原始文献语言代码列表，如 ['ZH','EN','JA']；"
            "不限制语言时为空，不得自行添加。仅筛选来源范围，不保证每种语言均有证据"
        ),
    )
    # 根据问题主要语言或用户的明确要求选择中文、英文或中英双语。
    output_language: Literal["ZH", "EN", "BILINGUAL"] = Field(
        description=(
            "最终比较结果语言；根据问题主要语言或用户明确要求选择，"
            "只有明确要求中英双语时才选择 BILINGUAL"
        )
    )
    # 用户在自然语言中明确指定的维度；未指定时为空并由后续节点规划。
    requested_dimensions: list[str] = Field(
        default_factory=list,
        max_length=12,
        description="用户在请求中明确指定的比较维度原文；未指定时返回空数组",
    )


class CompareObject(BaseModel):
    """保存可扩展比较对象的稳定名称、类型、标识和检索别名。"""

    # 用户原始写法。
    raw: str
    # 面向结果展示的规范名称。
    canonical_name: str
    # COUNTRY/POLICY/THEORY/METHOD/EDUCATION_MODEL/OTHER。
    type: str
    # 国家使用 ISO 3166-1 alpha-2，其他对象使用稳定规范名。
    identifier: str
    # 中英文检索别名，不包含空值和重复项。
    aliases: list[str]


class CompareDimensionPlan(BaseModel):
    """表示模型根据任务语义规划的候选比较维度。"""

    # 与规划 Prompt 保持一致，避免 Schema 接受少于业务要求的维度数量。
    dimensions: list[DimensionName] = Field(
        min_length=MAX_COMPARE_DIMENSIONS,
        max_length=MAX_COMPARE_DIMENSIONS,
        description="根据比较主题规划的互不重复维度，数量必须等于系统要求",
    )

    @field_validator("dimensions")
    @classmethod
    def validate_dimensions(cls, dimensions: list[str]) -> list[str]:
        """规范化标题并拒绝纯标点、空白和重复维度。

        Args:
            dimensions: 已通过基础长度约束的模型维度列表。

        Returns:
            去除首尾空白且保持原顺序的合法维度。

        Raises:
            ValueError: 任一维度不含文字或数字、清理后长度越界或存在重复项。
        """
        normalized = [normalize_dimension_title(item) for item in dimensions]
        if any(not 4 <= len(item) <= 30 for item in normalized):
            raise ValueError("每个比较维度必须是 4-30 个字符的短标题")
        if any(not any(character.isalnum() for character in item) for item in normalized):
            raise ValueError("比较维度必须包含中文、英文或数字，不能只有标点")
        if len(set(normalized)) != len(normalized):
            raise ValueError("比较维度不能重复")
        return normalized


class CompareQuery(BaseModel):
    """保存一个对象与一个比较维度对应的可审计检索计划。"""

    # 稳定查询标识由对象侧和维度序号组成，用于合并候选时保留召回轨迹。
    query_id: str = Field(min_length=3, max_length=32)
    # 当前查询只面向对象 A 或对象 B。
    object: Literal["A", "B"]
    # 当前查询只携带一个比较维度，避免多个语义目标稀释向量表示。
    dimension: DimensionName
    # 提交给统一 Hybrid Retrieval 的完整查询文本。
    text: str = Field(min_length=2, max_length=1200)


class CompareRetrievalContext(BaseModel):
    """记录一个 Candidate Chunk 被哪条对象×维度查询召回。"""

    query_id: str
    object: Literal["A", "B"]
    dimension: DimensionName
    # 当前查询内的确定性重排分数，只用于排序和审计，不表示概率。
    relevance_score: float = Field(ge=0, le=1)


class EvidenceCellSupport(BaseModel):
    """表示一个 Chunk 对某个对象×维度单元格的直接语义支持。"""

    object: Literal["A", "B"] = Field(
        description="该 Evidence 直接支持的比较对象侧，只能是 A 或 B",
    )
    dimension: DimensionName = Field(
        description="该 Evidence 直接支持的一个给定比较维度，必须逐字使用输入标题",
    )
    relevance: Literal["HIGH", "MEDIUM", "LOW"] = Field(
        description="该 Chunk 对指定对象与维度单元格的语义相关程度",
    )


class EvidenceAssignmentDecision(BaseModel):
    """保存一个 Candidate Chunk 对当前任务的语义归属判断。"""

    # 必须与输入候选真实 UUID 完全一致。
    chunk_id: uuid.UUID = Field(
        description="必须与输入 Candidate Chunk 完全一致的真实 UUID",
    )
    # 每项同时绑定对象和维度，禁止把对象判断与维度判断做笛卡尔积。
    supports: list[EvidenceCellSupport] = Field(
        default_factory=list,
        description=(
            "该 Chunk 直接支持的对象×维度单元格；没有直接支持项时返回空数组，"
            "即使内容与主题相关"
        ),
    )
    # 对当前比较任务的总体相关程度；IRRELEVANT 会被程序剔除。
    relevance: Relevance = Field(
        description=(
            "该 Chunk 对整个比较任务的总体相关度；supports 为空时必须为 IRRELEVANT，"
            "有支持项时不得为 IRRELEVANT"
        ),
    )
    # 简短可审计依据，不保存隐藏推理链。
    assignment_reason: str = Field(
        min_length=1,
        max_length=500,
        description="可审计的简短归属依据，只描述证据内容，不输出隐藏推理过程",
    )

    @model_validator(mode="after")
    def validate_supports(self) -> EvidenceAssignmentDecision:
        """保证有效候选至少支持一个单元格，并拒绝重复对象×维度。

        Returns:
            已验证的逐 Chunk 归属决定。

        Raises:
            PydanticCustomError: 相关度与支持单元格矛盾，或同一单元格重复出现。
        """
        if self.relevance == "IRRELEVANT" and self.supports:
            raise PydanticCustomError(
                "assignment_irrelevant_with_supports",
                "IRRELEVANT Candidate 不能声明支持单元格",
            )
        if self.relevance != "IRRELEVANT" and not self.supports:
            raise PydanticCustomError(
                "assignment_relevant_without_supports",
                "有效 Candidate 必须至少支持一个对象与维度单元格",
            )
        keys = [(item.object, item.dimension) for item in self.supports]
        if len(set(keys)) != len(keys):
            raise PydanticCustomError(
                "assignment_duplicate_cell",
                "同一对象与维度单元格不能重复",
            )
        return self


class EvidenceAssignmentBatch(BaseModel):
    """承载一批逐 Chunk EvidenceAssignment 结果。"""

    decisions: list[EvidenceAssignmentDecision] = Field(
        description="与输入 Candidate 一一对应的逐 Chunk 语义归属决定",
    )


class DimensionEvidenceCoverage(BaseModel):
    """保存一个比较维度下 A/B 两侧的真实 Evidence 覆盖。"""

    dimension: DimensionName
    object_a_evidence_count: int = Field(ge=0)
    object_b_evidence_count: int = Field(ge=0)
    status: CoverageLevel


class EvidenceBalance(BaseModel):
    """保存确定性 Evidence Balance Check 的统计和业务状态。"""

    # SUFFICIENT/INSUFFICIENT_EVIDENCE。
    status: str
    object_a_evidence_count: int
    object_b_evidence_count: int
    shared_evidence_count: int
    object_a_document_count: int
    object_b_document_count: int
    # 有真实出版年份时返回闭区间，否则为空列表。
    year_coverage: list[int]
    # 不重复来源文档数。
    source_diversity: int
    # 按语言统计不重复来源文档数。
    source_language_coverage: dict[str, int]
    # 按 HIGH/MEDIUM/LOW 统计通过归属判断的 Evidence 数。
    relevance_distribution: dict[str, int]
    # 按对象×维度计算的覆盖，阻止全局数量掩盖单元格错位。
    dimension_coverage: list[DimensionEvidenceCoverage] = Field(default_factory=list)


class CitedFinding(BaseModel):
    """表示一条必须由真实引用编号支撑的比较结论。"""

    text: str = Field(
        min_length=1,
        max_length=1600,
        description="只根据 Evidence 形成的一条共同点或差异点结论，不在文本中写 Citation",
    )
    citation_ids: list[int] = Field(
        min_length=1,
        description="直接支撑该结论的真实 Evidence citation_id",
    )


class ComparisonMatrixDraftRow(BaseModel):
    """表示模型为一个运行时维度生成的双侧文字和分侧引用。"""

    # 必须逐字使用本次运行时维度，后端会据此排序并拒绝模型新增维度。
    dimension: DimensionName = Field(
        description="当前矩阵行对应的给定比较维度，必须逐字使用输入标题",
    )
    object_a: str = Field(
        min_length=1,
        max_length=1200,
        description=(
            "该维度下对象 A 的证据结论；每个事实句在对应位置使用半角 [n] Citation；"
            "无专属证据时写“证据不足”"
        ),
    )
    object_b: str = Field(
        min_length=1,
        max_length=1200,
        description=(
            "该维度下对象 B 的证据结论；每个事实句在对应位置使用半角 [n] Citation；"
            "无专属证据时写“证据不足”"
        ),
    )
    similarity: str = Field(
        min_length=1,
        max_length=1200,
        description=(
            "该维度下由双侧或共享 Evidence 支持的共同点；每个事实句在对应位置使用"
            "半角 [n] Citation；允许综合双方各自的材料，不要求单条证据同时覆盖双方；"
            "仅归纳双方证据均支持的共同特征，未发现明确共同点时如实说明；"
            "缺少一侧依据时写“证据不足”"
        ),
    )
    difference: str = Field(
        min_length=1,
        max_length=1200,
        description=(
            "该维度下由双侧 Evidence 支持的差异；每个事实句在对应位置使用半角 [n] "
            "Citation；证据不足时明确说明"
        ),
    )
    # 三组编号只允许使用给定维度的真实 Evidence；共同点按引用集合校验双侧覆盖。
    object_a_citation_ids: list[int] = Field(
        description="直接支持对象 A 单元格结论的 citation_id；无证据时返回空数组",
    )
    object_b_citation_ids: list[int] = Field(
        description="直接支持对象 B 单元格结论的 citation_id；无证据时返回空数组",
    )
    shared_citation_ids: list[int] = Field(
        description=(
            "直接支撑该维度共同点的 citation_id 集合；可合并分别支持 A、B 的材料，"
            "或使用同时支持双方的材料，集合必须覆盖双方；缺少一侧依据时返回空数组"
        ),
    )


class ComparisonMatrixRow(ComparisonMatrixDraftRow):
    """表示经后端证据归属校验的最终矩阵行。"""

    # 默认值兼容升级前已持久化、只有统一 citation_ids 的历史运行结果。
    object_a_citation_ids: list[int] = Field(default_factory=list)
    object_b_citation_ids: list[int] = Field(default_factory=list)
    # 保留历史字段名；共同点引用既可来自共享材料，也可由双方各自的材料组成。
    shared_citation_ids: list[int] = Field(default_factory=list)
    # 三组分侧引用的确定性去重并集，供统一 Citation Validator 使用。
    citation_ids: list[int] = Field(default_factory=list)


class TimelineItem(BaseModel):
    """表示由证据支持的对象时间节点。"""

    year: int = Field(
        ge=1000,
        le=2100,
        description="Evidence 正文、标题或出版年份中明确出现的事件年份",
    )
    object: Literal["A", "B", "SHARED"] = Field(
        description="该时间节点归属的比较对象；模型生成阶段只使用 A 或 B",
    )
    event: str = Field(
        description="只依据所引 Evidence 描述的代表事件，不补充外部事实",
    )
    citation_ids: list[int] = Field(
        min_length=1,
        description="直接支持该年份、对象归属和事件描述的真实 citation_id",
    )


class TopicItem(BaseModel):
    """表示证据中可观察到的研究主题，不伪造统计比例。"""

    topic: str = Field(
        min_length=1,
        max_length=80,
        description="当前 Evidence 中反复出现或具有代表性的简短研究主题",
    )
    object: Literal["A", "B", "SHARED"] = Field(
        description="主题归属对象；SHARED 只能用于同时直接支持双方的 Evidence",
    )
    summary: str = Field(
        min_length=1,
        max_length=800,
        description="对当前 Evidence 中该主题的谨慎概括，不声称总体占比或全球趋势",
    )
    citation_ids: list[int] = Field(
        min_length=1,
        description="直接支持主题归属和概括内容的真实 citation_id",
    )


class ReferenceDocumentItem(BaseModel):
    """表示最终比较结论实际引用的一篇真实文献。"""

    citation_id: int
    document_id: uuid.UUID
    title: str


class ZhijianCoreAnalysisDraft(BaseModel):
    """承载第一版智鉴需要生成的核心文字比较结果。"""

    # 总览和双侧摘要保持必填，避免模型只返回零散要点而没有可读结论。
    summary: str = Field(
        min_length=1,
        max_length=2400,
        description=(
            "总体比较结论；每个事实句在对应位置使用半角 [n] Citation；如果有多个"
            " citation_id，不得写 [n,m]，应写 [n][m]"
        ),
    )
    object_a_summary: str = Field(
        min_length=1,
        max_length=1600,
        description="对象 A 的证据概览；每个事实句在对应位置使用半角 [n] Citation",
    )
    object_b_summary: str = Field(
        min_length=1,
        max_length=1600,
        description="对象 B 的证据概览；每个事实句在对应位置使用半角 [n] Citation",
    )
    # 第一版只生成文字要点和矩阵；每组限制数量，控制兼容模型输出长度和延迟。
    similarities: list[CitedFinding] = Field(
        default_factory=list,
        max_length=6,
        description=(
            "由双方 Evidence 直接支持的主要共同点；每个事实句在对应位置使用半角 [n] "
            "Citation"
        ),
    )
    differences: list[CitedFinding] = Field(
        default_factory=list,
        max_length=8,
        description=(
            "由对象专属 Evidence 直接支持的主要差异点；每个事实句在对应位置使用半角 "
            "[n] Citation"
        ),
    )
    comparison_matrix: list[ComparisonMatrixDraftRow] = Field(
        max_length=8,
        description="严格按输入维度顺序返回的 A/B 比较矩阵，不得增删或改名",
    )
    # 时间线只允许抽取 Evidence 中明确出现的年份，后端还会核对对象归属与年份来源。
    timeline: list[TimelineItem] = Field(
        max_length=20,
        description="仅包含 Evidence 明确支持年份和对象归属的代表性时间节点",
    )
    # “热点”仅表示本次 Evidence 中可观察到的代表主题，不表达总体频率或全球趋势。
    topic_distribution: list[TopicItem] = Field(
        max_length=6,
        description=(
            "当前 Evidence 中可观察到的代表主题；证据不足时返回空数组；每个事实句在"
            "对应位置使用半角 [n] Citation"
        ),
    )
    limitations: list[str] = Field(
        default_factory=list,
        max_length=12,
        description=(
            "由证据覆盖、语言、年份或对象不平衡导致的结果限制；每个事实句在对应位置"
            "使用半角 [n] Citation"
        ),
    )


class ZhijianAnalysisDraft(BaseModel):
    """承载 CrossAnalysis 生成的完整、待复核比较草稿。"""

    summary: str
    object_a_summary: str
    object_b_summary: str
    similarities: list[CitedFinding]
    differences: list[CitedFinding]
    comparison_matrix: list[ComparisonMatrixRow]
    timeline: list[TimelineItem]
    topic_distribution: list[TopicItem]
    limitations: list[str] = Field(default_factory=list)


class ReviewDecision(BaseModel):
    """表示 ResearchReviewer 对比较草稿的结构化审核结论。"""

    status: Literal["PASS", "REVISE", "FAIL"]
    issues: list[str] = Field(default_factory=list, max_length=12)


class ZhijianEvidenceRead(BaseModel):
    """向前端公开当前 Run 的 EvidenceAssignment 事实。"""

    chunk_id: uuid.UUID
    document_id: uuid.UUID
    citation_id: int
    assignment: Assignment
    supports_a: bool
    supports_b: bool
    relevance: Literal["HIGH", "MEDIUM", "LOW"]
    retrieval_branches: list[str]
    # 经过模型逐 Chunk 验证的对象×维度支持关系。
    cell_supports: list[EvidenceCellSupport] = Field(default_factory=list)
    # Candidate 的完整对象×维度召回轨迹，不能替代语义归属。
    retrieval_contexts: list[CompareRetrievalContext] = Field(default_factory=list)


class ZhijianResult(BaseModel):
    """定义智鉴专业工作台消费的完整运行结果。"""

    # SUCCESS/NEEDS_CLARIFICATION/UNSUPPORTED_SOURCE_LANGUAGE/INSUFFICIENT_EVIDENCE。
    status: str
    clarification_question: str | None = None
    topic: str | None = None
    object_a: CompareObject | None = None
    object_b: CompareObject | None = None
    time_range: list[int] = Field(default_factory=list)
    compare_dimensions: list[str] = Field(default_factory=list)
    summary: str = ""
    object_a_summary: str = ""
    object_b_summary: str = ""
    similarities: list[CitedFinding] = Field(default_factory=list)
    differences: list[CitedFinding] = Field(default_factory=list)
    comparison_matrix: list[ComparisonMatrixRow] = Field(default_factory=list)
    timeline: list[TimelineItem] = Field(default_factory=list)
    topic_distribution: list[TopicItem] = Field(default_factory=list)
    references: list[ReferenceDocumentItem] = Field(default_factory=list)
    evidence: list[ZhijianEvidenceRead] = Field(default_factory=list)
    evidence_balance: EvidenceBalance | None = None
    limitations: list[str] = Field(default_factory=list)
    evidence_level: str = "INSUFFICIENT"
