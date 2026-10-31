"""定义溯知文献综述输入、分析草稿与最终结构化结果。

模型草稿与最终结果分离：模型只能引用本次检索得到的编号，确定性节点负责
引用格式和最终 Citation 完整性校验。
"""

from __future__ import annotations

import uuid
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

ReviewLanguage = Literal["ZH", "EN", "BILINGUAL"]
ReviewLength = Literal["SHORT", "MEDIUM", "LONG"]
CitationStyle = Literal["GBT7714", "APA", "MLA"]
ProvenanceRelation = Literal[
    "EVOLVES_TO",
    "BRANCHES_TO",
    "ENABLES",
    "EXTENDS",
    "MERGES_INTO",
    "REPLACES",
]


class SuzhiRunCreate(BaseModel):
    """校验一次溯知综述任务的研究范围与格式要求。"""

    request_id: str = Field(min_length=8, max_length=128)
    knowledge_base_id: uuid.UUID
    conversation_id: uuid.UUID | None = None
    research_topic: str = Field(min_length=2, max_length=500)
    concept: str | None = Field(default=None, max_length=300)
    year_from: int | None = Field(default=None, ge=1000, le=2100)
    year_to: int | None = Field(default=None, ge=1000, le=2100)
    review_length: ReviewLength = "MEDIUM"
    citation_style: CitationStyle = "GBT7714"

    @field_validator("research_topic", "concept")
    @classmethod
    def normalize_text(cls, value: str | None) -> str | None:
        """清理用户输入两端空白，并把空概念转换为空值。

        Args:
            value: 待校验的研究主题或可选概念。

        Returns:
            去除首尾空白后的文本，或表示未填写的空值。
        """
        if value is None:
            return None
        return value.strip() or None

    @model_validator(mode="after")
    def validate_year_range(self) -> SuzhiRunCreate:
        """拒绝起止年份颠倒的综述范围。

        Returns:
            年份范围合法的当前请求模型。

        Raises:
            ValueError: 起始年份晚于截止年份。
        """
        if self.year_from and self.year_to and self.year_from > self.year_to:
            raise ValueError("year_from 不能晚于 year_to")
        return self


class ReviewConcept(BaseModel):
    """保存模型从用户任务中提取的概念及检索同义表达。"""

    canonical_name: str = Field(
        min_length=2,
        max_length=200,
        description="研究主题对应的规范核心概念名称，不添加事实解释",
    )
    search_terms: list[str] = Field(
        min_length=1,
        max_length=8,
        description="用于知识库检索的中英文同义词或紧密相关表达，按相关性排序",
    )
    output_language: ReviewLanguage = Field(
        description=(
            "最终综述语言；根据研究主题主要语言或用户明确要求选择 ZH、EN 或 BILINGUAL，"
            "只有用户明确要求中英双语时才选择 BILINGUAL"
        )
    )


class ReviewQueryPlan(BaseModel):
    """限制溯知扩展查询数量，避免自由扩展偏离研究主题。"""

    queries: list[str] = Field(
        min_length=1,
        max_length=5,
        description="围绕规范概念、用户研究主题和时间范围生成的检索表达，不得扩展到无关主题",
    )


class CitedReviewFinding(BaseModel):
    """表示一条由本次 Evidence 编号支撑的综述结论。"""

    title: str = Field(
        min_length=1,
        max_length=200,
        description="该条研究发现的简短标题，不包含 Citation 标记",
    )
    summary: str = Field(
        min_length=1,
        max_length=2000,
        description="只依据 Evidence 概括的研究发现正文；每个事实句在对应位置使用半角 [n] Citation",
    )
    citation_ids: list[int] = Field(
        min_length=1,
        description="summary 正文实际使用的全部真实 Evidence citation_id，按首次出现顺序排列",
    )


class ResearchGap(BaseModel):
    """描述本次检索文献呈现的研究不足及其证据逻辑，沿用原字段兼容历史结果。"""

    gap: str = Field(
        min_length=1,
        max_length=1000,
        description=(
            "基于本次检索 Evidence 归纳的研究不足简短标题，不包含 Citation 标记；"
            "不得声称未检索到就等于全球不存在"
        ),
    )
    evidence_logic: str = Field(
        min_length=1,
        max_length=1500,
        description=(
            "说明从本次检索文献的研究范围、时间或方法分布归纳研究不足的证据逻辑；"
            "每个事实句在对应位置使用半角 [n] Citation"
        ),
    )
    citation_ids: list[int] = Field(
        min_length=1,
        description=(
            "evidence_logic 正文实际使用的全部真实 Evidence citation_id，按首次出现顺序排列"
        ),
    )


class ProvenanceGraphNode(BaseModel):
    """表示模型从完整 Evidence 中归纳的一个技术或主题演进里程碑。"""

    id: str = Field(description="图内唯一且稳定的短标识，例如 transformer 或 gpt-3")
    label: str = Field(description="技术、模型、理论范式或关键应用阶段的简短名称")
    subtitle: str = Field(description="节点框内展示的一行简短特征、机制或定位说明")
    summary: str = Field(description="该里程碑在主题演进中的作用及与前后节点的关系")
    year: int | None = Field(default=None, description="该里程碑发生或发表的年份，未知时为 null")
    stage: int = Field(description="从左到右的零基拓扑层级；同一演进阶段使用相同值")
    lane: int = Field(
        description="分支纵向轨道；主线使用 0，上方分支用负数，下方分支用正数"
    )
    citation_ids: list[int] = Field(
        default_factory=list,
        description="支撑该演进里程碑的 Evidence citation_id",
    )


class ProvenanceGraphEdge(BaseModel):
    """表示模型归纳的两个溯源节点之间的有向语义关系。"""

    source: str = Field(description="起点节点 id")
    target: str = Field(description="终点节点 id")
    relation: ProvenanceRelation = Field(description="起点到终点的技术或主题演进关系")
    label: str = Field(description="点击关系后展示的简短关系名称")
    summary: str = Field(description="该演进关系在现有 Evidence 中的简要依据")
    citation_ids: list[int] = Field(
        default_factory=list,
        description="模型认为能够支撑该关系的 Evidence citation_id",
    )


class ProvenanceGraph(BaseModel):
    """保存主题分析与完整 Evidence 同次生成的主题演进有向图。"""

    title: str = Field(default="", description="与本次研究主题一致的溯源图标题")
    nodes: list[ProvenanceGraphNode] = Field(
        default_factory=list, description="按技术或主题发展顺序组织的里程碑节点"
    )
    edges: list[ProvenanceGraphEdge] = Field(
        default_factory=list, description="节点之间的有向演进关系"
    )


class LiteratureAnalysisDraft(BaseModel):
    """承载主题、理论、方法、争议和研究不足的 Evidence First 草稿。"""

    concept_definition: str = Field(
        min_length=1,
        max_length=3000,
        description="只依据 Evidence 给出的核心概念定义；每个事实句在对应位置使用半角 [n] Citation",
    )
    concept_citation_ids: list[int] = Field(
        min_length=1,
        description=(
            "concept_definition 正文实际使用的全部真实 Evidence citation_id，按首次出现顺序排列"
        ),
    )
    topics: list[CitedReviewFinding] = Field(
        default_factory=list,
        max_length=8,
        description=(
            "当前 Evidence 中可归纳的主要研究主题，不表达未经证据支持的总体热度；"
            "每个事实句在对应位置使用半角 [n] Citation"
        ),
    )
    theoretical_paths: list[CitedReviewFinding] = Field(
        default_factory=list,
        max_length=8,
        description=(
            "当前 Evidence 明确支持的理论视角、框架或解释路径；"
            "每个事实句在对应位置使用半角 [n] Citation"
        ),
    )
    methods: list[CitedReviewFinding] = Field(
        default_factory=list,
        max_length=8,
        description=(
            "当前 Evidence 明确出现的研究方法及其演变；"
            "每个事实句在对应位置使用半角 [n] Citation"
        ),
    )
    controversies: list[CitedReviewFinding] = Field(
        default_factory=list,
        max_length=8,
        description=(
            "当前 Evidence 中存在相反观点、权衡或未决问题的主要争议；"
            "每个事实句在对应位置使用半角 [n] Citation"
        ),
    )
    research_gaps: list[ResearchGap] = Field(
        default_factory=list,
        max_length=8,
        description="根据本次检索 Evidence 覆盖范围谨慎归纳的研究不足",
    )
    future_directions: list[CitedReviewFinding] = Field(
        default_factory=list,
        max_length=8,
        description="由当前 Evidence 和已归纳研究不足支持的后续研究方向",
    )


class LiteratureAnalysisWithGraph(BaseModel):
    """承载一次模型调用共同生成的综述分析与主题演进图。"""

    analysis_draft: LiteratureAnalysisDraft = Field(
        description="由本次固定 Evidence 支撑的概念、主题、方法、争议与研究不足分析"
    )
    provenance_graph: ProvenanceGraph = Field(
        description="由同一批 Evidence 支撑的技术、理论或研究主题演进里程碑图"
    )


class ReviewDraft(BaseModel):
    """保存模型生成且尚未通过确定性引用校验的完整综述。"""

    review: str = Field(
        min_length=1,
        max_length=30000,
        description=(
            "完整 Markdown 文献综述；每个事实句在对应位置使用半角 [n] Citation，"
            "不得输出 citation_id 内部字段名或在段末重复汇总引用"
        ),
    )


class SuzhiResult(BaseModel):
    """定义溯知工作台可恢复的版本化结果快照。"""

    status: str
    research_topic: str
    concept: str
    language: ReviewLanguage
    review_length: ReviewLength
    citation_style: CitationStyle
    concept_definition: str = ""
    concept_citation_ids: list[int] = Field(default_factory=list)
    topics: list[CitedReviewFinding] = Field(default_factory=list)
    theoretical_paths: list[CitedReviewFinding] = Field(default_factory=list)
    methods: list[CitedReviewFinding] = Field(default_factory=list)
    controversies: list[CitedReviewFinding] = Field(default_factory=list)
    research_gaps: list[ResearchGap] = Field(default_factory=list)
    future_directions: list[CitedReviewFinding] = Field(default_factory=list)
    provenance_graph: ProvenanceGraph | None = None
    review: str = ""
    references: list[str] = Field(default_factory=list)
    evidence_summary: dict[str, object] = Field(default_factory=dict)
    limitations: list[str] = Field(default_factory=list)
    evidence_level: str = "INSUFFICIENT"
