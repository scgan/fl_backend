"""定义Quill论文写作辅助的阶段参数、理论依据分析与最终结构化结果。

Phase 1 仅实现 THEORY 阶段（理论依据及文献推荐）：模型草稿只引用本次检索
得到的编号，确定性节点负责引用校验和结果组装。RQ、PLAN、DRAFT 与 REVISE
按《学术研究辅助设计方案》后续接入，编码在此先行冻结。
"""

from __future__ import annotations

import uuid
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

# 设计方案 §3.2 的五个显式阶段，另加模拟知网划词工具条的 POLISH 阶段；
# Phase 1 开放 THEORY，本次新增开放 POLISH，其余阶段后续接入。
QuillStage = Literal["RQ", "PLAN", "DRAFT", "REVISE", "THEORY", "POLISH"]
TheoryLanguage = Literal["ZH", "EN", "BILINGUAL"]
CitationStyle = Literal["GBT7714", "APA", "MLA"]
# 划词工具条的纯创作操作；概念解释与文献推荐属于硬证据操作，走 THEORY 阶段。
PolishAction = Literal[
    "CONTINUE",
    "EXPAND",
    "SHORTEN",
    "REWRITE",
    "POLISH",
    "ZH2EN",
    "EN2ZH",
]


class QuillRunCreate(BaseModel):
    """校验一次Quill THEORY 运行的研究主题与输出格式要求。"""

    request_id: str = Field(min_length=8, max_length=128)
    knowledge_base_id: uuid.UUID
    conversation_id: uuid.UUID | None = None
    research_topic: str = Field(min_length=2, max_length=500)
    year_from: int | None = Field(default=None, ge=1000, le=2100)
    year_to: int | None = Field(default=None, ge=1000, le=2100)
    citation_style: CitationStyle = "GBT7714"
    literature_limit: int = Field(default=10, ge=5, le=20)

    @field_validator("research_topic")
    @classmethod
    def normalize_topic(cls, value: str) -> str:
        """清理研究主题两端空白。

        Args:
            value: 用户提交的研究主题或研究问题。

        Returns:
            去除首尾空白后的非空主题文本。
        """
        return value.strip()

    @model_validator(mode="after")
    def validate_year_range(self) -> QuillRunCreate:
        """拒绝起止年份颠倒的检索范围。

        Returns:
            年份范围合法的当前请求模型。

        Raises:
            ValueError: 起始年份晚于截止年份。
        """
        if self.year_from and self.year_to and self.year_from > self.year_to:
            raise ValueError("year_from 不能晚于 year_to")
        return self


class TheoryConcept(BaseModel):
    """保存模型从研究主题中提取的规范概念及检索同义表达。"""

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
    output_language: TheoryLanguage = Field(
        description=(
            "最终输出语言；根据研究主题主要语言或用户明确要求选择 ZH、EN 或 BILINGUAL，"
            "只有用户明确要求中英双语时才选择 BILINGUAL"
        )
    )


class TheoryQueryPlan(BaseModel):
    """限制Quill扩展查询数量，避免自由扩展偏离研究主题。"""

    queries: list[str] = Field(
        min_length=1,
        max_length=5,
        description="围绕规范概念和研究主题生成的理论、方法与文献检索表达，不得扩展到无关主题",
    )


class CitedTheory(BaseModel):
    """一条由本次 Evidence 编号支撑的理论依据条目。"""

    theory: str = Field(
        min_length=1,
        max_length=200,
        description="理论、框架或概念视角的简短名称，不包含 Citation 标记",
    )
    description: str = Field(
        min_length=1,
        max_length=2000,
        description=(
            "只依据 Evidence 概括的理论要点及其与研究主题的关系；"
            "每个事实句在对应位置使用半角 [n] Citation"
        ),
    )
    citation_ids: list[int] = Field(
        min_length=1,
        description="description 正文实际使用的全部真实 Evidence citation_id，按首次出现顺序排列",
    )


class LiteratureRecommendation(BaseModel):
    """一条知识库内真实文献的推荐及理由。"""

    document_title: str = Field(
        min_length=1,
        max_length=1000,
        description="被推荐文献的标题，必须原样使用 Evidence 对应文献的真实 document_title",
    )
    relevance_reason: str = Field(
        min_length=1,
        max_length=1500,
        description=(
            "说明该文献与理论依据或研究主题的关联；"
            "每个事实句在对应位置使用半角 [n] Citation"
        ),
    )
    citation_ids: list[int] = Field(
        min_length=1,
        description=(
            "relevance_reason 正文实际使用的全部真实 Evidence citation_id，"
            "按首次出现顺序排列"
        ),
    )


class TheoryGap(BaseModel):
    """描述由现有研究分布推导出的空白及其证据逻辑。"""

    gap: str = Field(
        min_length=1,
        max_length=1000,
        description=(
            "基于当前 Evidence 推导的研究空白简短标题，不包含 Citation 标记；"
            "不得声称未检索到就等于全球不存在"
        ),
    )
    evidence_logic: str = Field(
        min_length=1,
        max_length=1500,
        description=(
            "说明从现有研究范围、时间或方法分布推导空白的证据逻辑；"
            "每个事实句在对应位置使用半角 [n] Citation"
        ),
    )
    citation_ids: list[int] = Field(
        min_length=1,
        description=(
            "evidence_logic 正文实际使用的全部真实 Evidence citation_id，"
            "按首次出现顺序排列"
        ),
    )


class TheoryAnalysisDraft(BaseModel):
    """承载理论依据、文献推荐与研究空白的 Evidence First 草稿。"""

    theoretical_basis: list[CitedTheory] = Field(
        default_factory=list,
        max_length=8,
        description=(
            "当前 Evidence 明确支持的理论、框架或概念视角；"
            "每个事实句在对应位置使用半角 [n] Citation"
        ),
    )
    literature_recommendations: list[LiteratureRecommendation] = Field(
        default_factory=list,
        max_length=20,
        description="知识库内真实文献的推荐列表及证据支撑的推荐理由",
    )
    research_gaps: list[TheoryGap] = Field(
        default_factory=list,
        max_length=8,
        description="根据当前 Evidence 覆盖范围谨慎推导的研究空白",
    )


class PolishOutput(BaseModel):
    """保存划词润改模型生成的纯文本结果。"""

    output_text: str = Field(
        min_length=1,
        max_length=20000,
        description="按操作要求生成的改写结果正文；不包含 [n] Citation、解释或前后缀说明",
    )


class QuillPolishResult(BaseModel):
    """定义划词润改工作台可恢复的版本化结果快照。"""

    status: str = Field(description="运行结果状态，当前固定为 SUCCESS 或 FAILED")
    stage: Literal["POLISH"] = Field(description="固定为划词润改阶段编码")
    action: PolishAction = Field(description="本次执行的划词操作编码")
    output_text: str = Field(description="模型生成的改写结果正文")
    academic_notice: str = Field(
        default="本结果由 AI 辅助生成，请遵守所在机构学术规范。",
        description="必须随结果展示的学术诚信提示",
    )
    evidence_level: str = Field(
        default="INSUFFICIENT",
        description="划词润改属于纯创作操作，固定为 INSUFFICIENT 且不产生 Citation",
    )


class QuillResult(BaseModel):
    """定义Quill工作台可恢复的版本化结果快照。"""

    status: str = Field(description="运行结果状态，当前固定为 SUCCESS 或 FAILED")
    stage: QuillStage = Field(description="产出该结果的工作阶段编码")
    research_topic: str = Field(description="用户提交的研究主题或研究问题原文")
    language: TheoryLanguage = Field(description="结果输出语言")
    citation_style: CitationStyle = Field(description="参考文献格式编码")
    theoretical_basis: list[CitedTheory] = Field(
        default_factory=list, description="已通过引用校验的理论依据条目"
    )
    literature_recommendations: list[LiteratureRecommendation] = Field(
        default_factory=list, description="受 literature_limit 约束的文献推荐列表"
    )
    research_gaps: list[TheoryGap] = Field(
        default_factory=list, description="由证据分布推导的研究空白"
    )
    references: list[str] = Field(default_factory=list, description="按引用格式生成的参考文献")
    evidence_summary: dict[str, object] = Field(
        default_factory=dict, description="证据数量、文献数、年份范围和语言覆盖统计"
    )
    limitations: list[str] = Field(
        default_factory=list, description="结果裁剪或降级时的确定性说明"
    )
    academic_notice: str = Field(
        default="本结果由 AI 辅助生成，请遵守所在机构学术规范。",
        description="必须随结果展示的学术诚信提示",
    )
    evidence_level: str = Field(default="INSUFFICIENT", description="证据强度等级")


class ResearchQuestion(BaseModel):
    """一条由本次 Evidence 编号支撑的候选研究问题。"""

    proposed_title: str = Field(
        default="",
        max_length=300,
        description="候选论文题目，属于创作建议，不包含事实结论或 Citation",
    )
    research_object: str = Field(
        default="",
        max_length=300,
        description="建议研究的对象，用户未限定时明确为建议，不伪称已确认",
    )
    research_scope: str = Field(
        default="",
        max_length=500,
        description="建议的研究范围，不将未提供的地区、时间或样本当成既定条件",
    )
    question: str = Field(
        min_length=10,
        max_length=300,
        description="以问句形式表述的候选研究问题，不包含 Citation 标记",
    )
    rationale: str = Field(
        min_length=1,
        max_length=1500,
        description="说明提出该问题的证据逻辑；每个事实句在对应位置使用半角 [n] Citation",
    )
    theory_hints: list[str] = Field(
        default_factory=list,
        max_length=4,
        description="供作者进一步查证的理论视角提示，属于创作性线索而非已验证事实",
    )
    method_hints: list[str] = Field(
        default_factory=list,
        max_length=4,
        description="供作者参考的研究方法提示，属于创作性线索而非已验证事实",
    )
    citation_ids: list[int] = Field(
        min_length=1,
        description="rationale 正文实际使用的全部真实 Evidence citation_id，按首次出现顺序排列",
    )


class RQDraft(BaseModel):
    """承载研究问题生成阶段的 Evidence First 草稿。"""

    research_questions: list[ResearchQuestion] = Field(
        min_length=1,
        max_length=5,
        description="基于当前证据分布生成的候选研究问题列表，按推荐优先级排序",
    )


class QuillRQResult(BaseModel):
    """定义研究问题生成阶段可恢复的版本化结果快照。"""

    status: str = Field(description="运行结果状态，当前固定为 SUCCESS 或 FAILED")
    stage: Literal["RQ"] = Field(description="固定为研究问题生成阶段编码")
    research_topic: str = Field(description="用户提交的研究主题或研究问题原文")
    language: TheoryLanguage = Field(description="结果输出语言")
    citation_style: CitationStyle = Field(description="参考文献格式编码")
    research_questions: list[ResearchQuestion] = Field(
        default_factory=list, description="已通过引用校验的候选研究问题列表"
    )
    references: list[str] = Field(default_factory=list, description="按引用格式生成的参考文献")
    evidence_summary: dict[str, object] = Field(
        default_factory=dict, description="证据数量、文献数、年份范围和语言覆盖统计"
    )
    limitations: list[str] = Field(
        default_factory=list, description="结果裁剪或降级时的确定性说明"
    )
    academic_notice: str = Field(
        default="本结果由 AI 辅助生成，请遵守所在机构学术规范。",
        description="必须随结果展示的学术诚信提示",
    )
    evidence_level: str = Field(default="INSUFFICIENT", description="证据强度等级")


class OutlineSection(BaseModel):
    """论文大纲中的一个章节及其结构化写作建议。"""

    section_id: str = Field(
        min_length=1,
        max_length=16,
        description="章节编号，如 1、2.1；同一大纲内唯一且体现层级顺序",
    )
    title: str = Field(
        min_length=1,
        max_length=200,
        description="章节标题，不包含 Citation 标记",
    )
    writing_points: list[str] = Field(
        min_length=1,
        max_length=8,
        description=(
            "该节应覆盖的写作要点，属于结构性创作建议；"
            "不得包含 [n] Citation 或具体研究事实"
        ),
    )
    suggested_citations: list[int] = Field(
        default_factory=list,
        description="建议本节参考的真实 Evidence citation_id；没有把握时留空，不得编造",
    )


class PlanDraft(BaseModel):
    """承载写作方案阶段的论文大纲草稿。"""

    thesis_statement: str = Field(
        min_length=5,
        max_length=500,
        description="论文中心论点或研究主线的简练表述，属于结构性创作；不得包含 [n] Citation",
    )
    sections: list[OutlineSection] = Field(
        min_length=3,
        max_length=12,
        description="按写作顺序组织的章节大纲",
    )


class QuillPlanResult(BaseModel):
    """定义写作方案阶段可恢复的版本化结果快照。"""

    status: str = Field(description="运行结果状态，当前固定为 SUCCESS 或 FAILED")
    stage: Literal["PLAN"] = Field(description="固定为写作方案阶段编码")
    research_topic: str = Field(description="用户提交的研究主题或研究问题原文")
    language: TheoryLanguage = Field(description="结果输出语言")
    citation_style: CitationStyle = Field(description="参考文献格式编码")
    based_on_artifact_id: str | None = Field(
        default=None, description="引用的研究问题产物 ID；独立生成时为空"
    )
    thesis_statement: str = Field(default="", description="论文中心论点表述")
    sections: list[OutlineSection] = Field(
        default_factory=list, description="已通过校验的章节大纲"
    )
    references: list[str] = Field(default_factory=list, description="被建议章节引用的参考文献")
    evidence_summary: dict[str, object] = Field(
        default_factory=dict, description="证据数量、文献数、年份范围和语言覆盖统计"
    )
    limitations: list[str] = Field(
        default_factory=list, description="结果裁剪或降级时的确定性说明"
    )
    academic_notice: str = Field(
        default="本结果由 AI 辅助生成，请遵守所在机构学术规范。",
        description="必须随结果展示的学术诚信提示",
    )
    evidence_level: str = Field(default="INSUFFICIENT", description="证据强度等级")


class SectionDraft(BaseModel):
    """承载单节论文草稿的模型输出。"""

    section_text: str = Field(
        min_length=1,
        max_length=8000,
        description=(
            "本节论文正文；每个事实性论断在对应位置使用半角 [n] Citation；"
            "证据不足处用 [待补充文献：主题线索] 占位，不得编造引用"
        ),
    )


class QuillDraftResult(BaseModel):
    """定义逐节起草阶段可恢复的版本化结果快照。"""

    status: str = Field(description="运行结果状态，当前固定为 SUCCESS 或 FAILED")
    stage: Literal["DRAFT"] = Field(description="固定为逐节起草阶段编码")
    research_topic: str = Field(description="用户提交的研究主题或研究问题原文")
    language: TheoryLanguage = Field(description="结果输出语言")
    citation_style: CitationStyle = Field(description="参考文献格式编码")
    based_on_artifact_id: str | None = Field(
        default=None, description="引用的写作方案产物 ID；独立生成时为空"
    )
    section_ref: str = Field(description="起草的章节编号，对应大纲 section_id")
    section_title: str = Field(description="起草的章节标题")
    draft_text: str = Field(description="已通过引用校验的本节论文正文")
    references: list[str] = Field(default_factory=list, description="被本节引用的参考文献")
    evidence_summary: dict[str, object] = Field(
        default_factory=dict, description="证据数量、文献数、年份范围和语言覆盖统计"
    )
    limitations: list[str] = Field(
        default_factory=list, description="结果裁剪或降级时的确定性说明"
    )
    academic_notice: str = Field(
        default="本结果由 AI 辅助生成，请遵守所在机构学术规范。",
        description="必须随结果展示的学术诚信提示",
    )
    evidence_level: str = Field(default="INSUFFICIENT", description="证据强度等级")


class RevisedText(BaseModel):
    """承载学术修改后的论文文本。"""

    revised_text: str = Field(
        min_length=1,
        max_length=20000,
        description=(
            "按修改要求改写后的论文文本；"
            "新增或调整的事实性论断在对应位置使用半角 [n] Citation"
        ),
    )


class QuillReviseResult(BaseModel):
    """定义修改完善阶段可恢复的版本化结果快照。"""

    status: str = Field(description="运行结果状态，当前固定为 SUCCESS 或 FAILED")
    stage: Literal["REVISE"] = Field(description="固定为修改完善阶段编码")
    research_topic: str = Field(description="用户提交的研究主题或原文摘要")
    language: TheoryLanguage = Field(description="结果输出语言")
    citation_style: CitationStyle = Field(description="参考文献格式编码")
    revise_instruction: str = Field(description="用户给出的修改要求原文")
    revised_text: str = Field(description="已通过引用校验的修改后文本")
    references: list[str] = Field(default_factory=list, description="被修改后文本引用的参考文献")
    evidence_summary: dict[str, object] = Field(
        default_factory=dict, description="证据数量、文献数、年份范围和语言覆盖统计"
    )
    limitations: list[str] = Field(
        default_factory=list, description="结果裁剪或降级时的确定性说明"
    )
    academic_notice: str = Field(
        default="本结果由 AI 辅助生成，请遵守所在机构学术规范。",
        description="必须随结果展示的学术诚信提示",
    )
    evidence_level: str = Field(default="INSUFFICIENT", description="证据强度等级")
