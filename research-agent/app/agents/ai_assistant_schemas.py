"""定义 AI 入口的任务识别、候选文献和结构化结果契约，不包含存储访问。"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class AssistantSearchPlan(BaseModel):
    """限定模型可表达的题录检索条件，概念间 AND、同组近义词间 OR。"""

    concepts: list[list[str]] = Field(
        max_length=8, description="主题概念组，每组1至5个词；仅作者或年份等检索可为空列表"
    )
    author: str | None = Field(default=None, max_length=100, description="用户明确限定的作者")
    language: Literal["ZH", "EN"] | None = Field(
        default=None,
        description=(
            "原始文献来源语言：明确要中文文献为ZH，英文文献为EN；"
            "未限定、不限语种或仅要求某种回答语言时为null，不得因问题改写遗漏原始要求"
        ),
    )
    year_from: int | None = Field(default=None, ge=1000, le=9999, description="出版起始年")
    year_to: int | None = Field(default=None, ge=1000, le=9999, description="出版结束年")
    keyword_only: bool = Field(default=False, description="仅在用户明确限定关键词字段时为真")

    @model_validator(mode="after")
    def validate_conditions(self):
        """拒绝空概念、超长词和倒置年份，避免把无效条件退化成全库浏览。

        Returns:
            去除词项首尾空白后的当前条件。

        Raises:
            ValueError: 概念组、词项或年份范围不合法。
        """
        if any(not 1 <= len(group) <= 5 for group in self.concepts):
            raise ValueError("每个概念必须包含1至5个检索词")
        self.concepts = [[word.strip() for word in group] for group in self.concepts]
        if any(not word or len(word) > 100 for group in self.concepts for word in group):
            raise ValueError("检索词必须为1至100个字符")
        if self.author is not None:
            self.author = self.author.strip()
            if not self.author:
                raise ValueError("作者条件不能为空")
        if not (self.concepts or self.author or self.language or self.year_from or self.year_to):
            raise ValueError("至少需要一个明确的检索条件")
        if self.year_from and self.year_to and self.year_from > self.year_to:
            raise ValueError("起始年份不能晚于结束年份")
        return self


class AssistantIntent(BaseModel):
    """将当前问题与有限历史解析为单一任务和独立检索问题。"""

    intent: Literal["SEARCH", "QA", "WRITING", "READING"] = Field(
        description="找文献、研究问答、知识库写作或基于前文具体文献的阅读分析"
    )
    search_query: str = Field(min_length=1, max_length=4000, description="结合上下文补全的检索问题")
    writing_stage: Literal["THEORY", "RQ", "PLAN", "DRAFT", "REVISE"] | None = Field(
        default=None, description="明确要求的写作成果类型；需求模糊或非写作时为空，禁止默认提纲"
    )
    research_topic: str = Field(
        default="", max_length=3900, description="纯研究主题，不含生成提纲等任务词；未知时留空"
    )
    writing_requirements: str = Field(
        default="", max_length=1000, description="用户明确指定的篇幅、层级、侧重等要求，与主题分开"
    )
    needs_clarification: bool = Field(
        default=False, description="已识别写作需求但成果类型尚不明确时为真"
    )
    clarification_question: str = Field(
        default="", max_length=300, description="需要明确写作成果时的一句简短询问"
    )
    section_ref: str = Field(
        default="", max_length=16, description="明确指定的章节编号，未知时留空"
    )
    reading_task: Literal["ANALYZE", "ASK"] = Field(
        default="ANALYZE", description="整体阅读分析用ANALYZE，具体比较或追问用ASK"
    )
    reading_source_run_id: str | None = Field(
        default=None, pattern=r"^[0-9a-f]{32}$", description="只能引用提供的文献来源Run"
    )
    reading_source_kind: Literal["CANDIDATES", "CITATIONS", "READING", "SELECTED"] | None = Field(
        default=None, description="搜索候选、实际引用、已有阅读集合或页面勾选集合"
    )
    reading_reference: Literal["AUTO", "SELECTED", "LIST", "SPECIFIC"] = Field(
        default="AUTO",
        description=(
            "材料指代语义：AUTO为上述/这些等泛指，服务优先勾选；SELECTED为明确所选；"
            "LIST为明确整页、全部或某来源集合；SPECIFIC为篇名、序号或引用编号限定"
        ),
    )
    reading_positions: list[int] = Field(
        default_factory=list, max_length=20, description="来源列表内从1开始的篇目序号，空为整组"
    )
    reading_all_results: bool = Field(
        default=False, description="明确要求全部检索结果时为真，当前页或上述文献不等于全部结果"
    )

    @model_validator(mode="after")
    def clarify_missing_stage(self):
        """写作阶段缺失或模型要求澄清时保持未选择状态，不静默生成提纲。"""
        if self.intent == "WRITING" and (self.needs_clarification or self.writing_stage is None):
            self.needs_clarification = True
            self.writing_stage = None
        return self


class AssistantReadingContext(BaseModel):
    """校验页面可见文献的提示；服务仍从原运行验证篇目，不信任客户端授权。"""

    model_config = ConfigDict(extra="forbid")
    source_run_id: str = Field(pattern=r"^[0-9a-f]{32}$", description="正在展示文献的原Run")
    page: int = Field(default=1, ge=1, le=50, strict=True, description="搜索结果当前页")
    selected_ids: list[Annotated[int, Field(strict=True, gt=0, lt=1 << 63)]] = Field(
        default_factory=list, max_length=20, description="页面勾选篇目"
    )

    @model_validator(mode="after")
    def validate_selection(self):
        """拒绝重复或非正整数的篇目提示。"""
        if any(key <= 0 for key in self.selected_ids) or len(set(self.selected_ids)) != len(
            self.selected_ids
        ):
            raise ValueError("请选择最多20篇不重复的有效文献")
        return self


class AssistantReadingOptions(BaseModel):
    """显式继续阅读操作，只引用原阅读卡片，不接收模型或客户端指定的文献ID。"""

    model_config = ConfigDict(extra="forbid")
    source_run_id: str = Field(pattern=r"^[0-9a-f]{32}$", description="同会话成功阅读卡片")
    task: Literal["ASK"] = Field(default="ASK", description="点击推荐问题直接执行文献追问")


class AssistantWritingOptions(BaseModel):
    """校验用户确认的写作参数；资源 ID 由服务端复核，不交给模型决定权限。"""

    model_config = ConfigDict(extra="forbid")
    knowledge_base_id: int = Field(gt=0, strict=True, description="当前任务明确选择的本人知识库")
    stage: Literal["THEORY", "RQ", "PLAN", "DRAFT", "REVISE"]
    artifact_id: str | None = Field(
        default=None, pattern=r"^[0-9a-f]{32}$", description="同库同会话前序产物"
    )
    section_ref: str = Field(default="", max_length=16, description="起草目标章节")
    revise_instruction: str = Field(default="", max_length=500, description="学术修改要求")
    source_text: str = Field(default="", max_length=4000, description="用户明确提供的待修改原文")
    writing_requirements: str = Field(
        default="", max_length=1000, description="用户确认的写作规格，与研究主题和任务类型分别冻结"
    )
    task_corrected: bool = Field(
        default=False, strict=True, description="用户通过更正任务主动改变明确识别的成果类型"
    )
    source_run_id: str | None = Field(
        default=None, pattern=r"^[0-9a-f]{32}$", description="发起操作的原确认卡或产物卡 Run"
    )
    source_action: str | None = Field(
        default=None, min_length=1, max_length=80, description="原卡内的稳定操作标识"
    )

    @model_validator(mode="after")
    def validate_action_reference(self):
        """来源卡和操作标识必须成对出现，避免无法恢复或去重的半份来源。"""
        if bool(self.source_run_id) != bool(self.source_action):
            raise ValueError("写作卡片来源和操作标识必须同时提供")
        return self


class AssistantDocument(BaseModel):
    """从真实题录或受控 Evidence 投影的可勾选文献，不允许模型生成题录。"""

    corpus_id: int = Field(gt=0, description="真实文献主键")
    title: str = Field(description="文献标题")
    author: str = Field(description="作者名称")
    year: int | None = Field(description="出版年份，未知时为空")
    source: str = Field(description="文献来源或期刊")
    snippet: str = Field(description="本轮命中的原文片段摘要")
    readable: bool | None = Field(
        default=None, description="正文双路索引是否就绪，题录检索不以此过滤"
    )


class AssistantResult(BaseModel):
    """AI_ASSISTANT 结果的版本一载荷，保存候选、实际引用与任务上下文。"""

    intent: Literal["SEARCH", "QA", "WRITING", "READING"] = Field(description="本轮任务意图")
    documents: list[AssistantDocument] = Field(description="按召回顺序去重的候选文献")
    citations: list[dict[str, Any]] = Field(description="最终回答实际使用且通过校验的引用")
    search_query: str = Field(description="结合历史补全的独立检索问题")
    scope: Literal["GLOBAL", "SELECTED", "DATABASE", "KNOWLEDGE_BASE"] = Field(
        description="服务层冻结的检索范围类型"
    )
    query: str = Field(description="用户原始问题，保持语言及输出要求")
    search_plan: AssistantSearchPlan | None = Field(
        default=None, description="分页复用的固定检索条件"
    )
    total: int | None = Field(default=None, description="题录检索匹配总数，不是证据片段数")
    document_index: dict[str, Any] | None = Field(
        default=None, description="服务层首次检索生成的有序题录 ID 和总数快照，供翻页复用",
    )
    page: int = Field(default=1, description="当前题录页码")
    page_size: int = Field(default=10, description="题录每页固定十篇")
    research_intent: str | None = Field(default=None, description="启明分类后的真实研究意图")
    agent_route: list[str] = Field(default_factory=list, description="当前 AI Run 的实际执行路径")
    research_result: dict[str, Any] | None = Field(
        default=None, description="原 AI 会话内的专业结果快照"
    )
    writing_result: dict[str, Any] | None = Field(
        default=None, description="待确认任务或经过校验的阶段产物"
    )
    reading_result: dict[str, Any] | None = Field(
        default=None, description="固定文献范围的分析或追问结果，含来源及版本快照"
    )
