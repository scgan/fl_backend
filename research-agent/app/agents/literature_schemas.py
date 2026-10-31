"""文献阅读的固定展示模板与追问的结构化 chunk 引用。"""

import re
from typing import Annotated, Literal

from pydantic import BaseModel, Field, model_validator

# 展示分区与验证器共同使用的固定键集合；前端按这些键映射中文标题。
SECTION_TITLES = {
    "background": "背景与研究问题",
    "ideas": "核心观点与框架",
    "methods": "研究方法与材料",
    "findings": "主要发现与结论",
    "contributions": "贡献与价值",
    "limitations": "局限与适用范围",
}

# 分析与追问共用的问题长度约束，限制可直接回填输入框的推荐问题大小。
RecommendedQuestion = Annotated[str, Field(min_length=2, max_length=300)]


# 下列模型以源码注释补充职责：类 docstring 会自动进入 JSON Schema，影响既有模型提示词。
# 一个摘要、观点或创新方向的展示段落；真实性及引用定位由其他流程负责。
# 阅读分析的单项正文声明。
#
# Attributes:
#     text: 非空正文，最多一千字符；模型提示和引用规则由调用方另外提供。
class ReadingClaim(BaseModel):
    # 纯正文，不携带模型自报的原文路径或页码。
    text: str = Field(min_length=1, max_length=1000)


# 单篇分析的一个固定分区；无适用材料时也保留该分区及原因。
# 固定六分区中的一个分析分区，允许材料不足但必须解释原因。
#
# Attributes:
#     key: 固定分区键，由 SECTION_TITLES 定义。
#     claims: 本区最多五条正文声明。
#     missing_reason: 无声明时必须提供的不适用或材料不足说明。
class ReadingSection(BaseModel):
    # 分区稳定标识，不允许模型自行增删或改名。
    key: Literal["background", "ideas", "methods", "findings", "contributions", "limitations"]
    # 最多五条展示段落；可为空，但为空时必须由 missing_reason 解释。
    claims: list[ReadingClaim] = Field(max_length=5)
    # 方法不适用或原文未提供材料的说明，有 claims 时允许为空字符串。
    missing_reason: str = Field(max_length=300)


# 一篇文献或一个分析批次的六分区输出；长文献的多批结果复用该结构层级合并。
# 约束单篇文献阅读结果的类型、摘要、六分区及推荐追问。
#
# Attributes:
#     document_type: 文献类型描述。
#     summary: 一至三条摘要声明。
#     sections: 必须完整且不重复的六个固定分析分区。
#     recommended_questions: 两至六个后续阅读问题。
class DocumentReading(BaseModel):
    # 政策解读、理论研究、实证研究等文献类型的简要描述。
    document_type: str = Field(min_length=1, max_length=80)
    # 一至三段摘要；推荐检索目前只使用第一段的有限长度文本。
    summary: list[ReadingClaim] = Field(min_length=1, max_length=3)
    # 必须覆盖六个不同的固定分区，由后置验证器核对键集合。
    sections: list[ReadingSection] = Field(min_length=6, max_length=6)
    # 可围绕本篇继续追问的问题，不是推荐文献名单。
    recommended_questions: list[RecommendedQuestion] = Field(min_length=2, max_length=6)

    @model_validator(mode="after")
    def fixed_sections(self):
        """校验六分区完整性及空分区解释，避免模板缺项被当成完整分析。

        Returns:
            校验通过的当前模型实例，不重排或改写模型生成的正文。

        Raises:
            ValueError: 分区键缺失、重复导致集合不完整，或空分区没有原因说明。
        """
        if {item.key for item in self.sections} != set(SECTION_TITLES):
            raise ValueError("阅读分析必须包含六个固定分区")
        if any(not item.claims and not item.missing_reason.strip() for item in self.sections):
            raise ValueError("无证据分区必须说明不适用或材料未提供")
        return self


class ReadingSynthesisClaim(ReadingClaim):
    """带主题标题的充分展开的综合正文，兼容历史纯文本并忽略旧文献列表字段。"""

    # 综合正文独立放宽长度，支持段间空行；不改变逐篇分析及追问的长度限制。
    text: str = Field(min_length=1, max_length=2400)
    # 新结果使用具体主题标题，旧记录和模型降级结果允许为空。
    title: str = Field(default="", max_length=80)


class ReadingInnovationPoint(BaseModel):
    """创新方向内的一个展开角度，标签按具体内容生成而非固定栏目。"""

    # 简短角度名称，例如机制假设、研究设计或适用边界，不包含列表或标题标记。
    label: str = Field(min_length=1, max_length=20)
    # 完整解释该角度的段落，可含少量 Markdown 强调。
    text: str = Field(min_length=1, max_length=1200)


class ReadingInnovation(ReadingSynthesisClaim):
    """以自适应分项展开创新设想，同时兼容只有正文和验证路径的旧记录。"""

    # 新结果由 points 承载正文；保留 text 供旧记录恢复，避免重复生成两套正文。
    text: str = Field(default="", max_length=2400)
    # 创新方向的小标题与正文独立存储，前端直接完整展示。
    title: str = Field(
        min_length=1, max_length=60,
        description="8～20 个中文字的小标题，概括本项具体研究切入点，不把正文塞进标题。",
    )
    # 每个新方向包含三至五个展开角度，空列表仅用于读取旧正文格式。
    points: list[ReadingInnovationPoint] = Field(default_factory=list, max_length=5)
    # 仅兼容旧结果；新格式把检验方法融入合适的分项。
    validation_path: str = Field(default="", max_length=600)

    @model_validator(mode="after")
    def points_or_legacy_text(self):
        """校验新分项完整性或旧正文，防止发布空创新方向及重复空标签。

        Returns:
            校验通过的当前创新方向，保留分项正文及顺序，新格式清空兼容旧字段。

        Raises:
            ValueError: 分项不足三项、标签或正文空白、标签重复，或没有任何正文。
        """
        if self.points:
            labels = [point.label.strip() for point in self.points]
            if len(self.points) < 3:
                raise ValueError("创新方向需要三至五个展开角度")
            if any(not label for label in labels) or len(set(labels)) != len(labels):
                raise ValueError("创新方向的分项标签不能空白或重复")
            if any(not point.text.strip() for point in self.points):
                raise ValueError("创新方向的分项正文不能为空白")
            self.text = ""
            self.validation_path = ""
        elif not self.text.strip():
            raise ValueError("创新方向必须提供分项内容或旧版正文")
        return self


# 基于已完成的逐篇分析生成综合概览。
# 保存单篇或多篇阅读的综合概览，兼容旧记录没有创新方向字段的情况。
#
# Attributes:
#     overview: 一至六段综合正文，不在此截断模型段落。
#     common_themes: 最多五项共同主题。
#     differences: 最多五项差异。
#     potential_innovations: 最多五项待探索方向，旧数据缺失时使用空列表。
#     recommended_questions: 两至六个继续研究的问题。
class ReadingOverview(BaseModel):
    # 主题正文充分解释方法、发现及联系；不通过额外论文列表承载正文内容。
    overview: list[ReadingSynthesisClaim] = Field(
        min_length=1,
        max_length=6,
        description=(
            "整体概览按主题组织，每项 text 可有 1～2 个连贯段落，用空行分段。"
            "多篇通常 5～6 项、共约 1000～1600 中文字；单篇通常 3～4 项、"
            "共约 600～900 中文字。"
            "展开研究背景、关注对象、方法与主要发现、文献联系及阅读启发，避免只写简短摘要。"
        ),
    )
    # 多篇之间可具体说明的共同主题；单篇阅读按节点指令返回空列表。
    common_themes: list[ReadingSynthesisClaim] = Field(max_length=5)
    # 对象、方法或发现的差异，不能自动解释为研究结论彼此冲突。
    differences: list[ReadingSynthesisClaim] = Field(max_length=5)
    # 待探索方向而非已证实发现；默认空列表兼容没有该字段的历史结构。
    potential_innovations: list[ReadingInnovation] = Field(
        default_factory=list,
        max_length=5,
        description=(
            "结合所选文献提出 3～5 个可进一步研究的潜在创新方向。每项 title 为具体小标题，"
            "points 包含 3～5 个按该方向实际内容选择的展开角度，每项有简短 label 和完整 text。"
            "不同方向的角度和数量可不同，不套用固定栏目，不列文献清单。总正文约 350～550 "
            "中文字，重点解释具体推进思路与研究设计，同时交代研究空间、价值或边界。"
            "以待探索的方向表述，不当作已经得到的研究结论；外层 text、validation_path 留空。"
        ),
    )
    # 用于综合阅读后的快捷追问，与库内文献推荐独立。
    recommended_questions: list[RecommendedQuestion] = Field(min_length=2, max_length=6)


# 一段追问回答及其实际使用的局部引用编号；继承 ReadingClaim 的正文长度限制。
# 保存追问回答段落与其实际使用的结构化原文引用。
#
# Attributes:
#     citation_ids: 最多二十个严格正整数编号；正文不手写方括号引用，发布时统一编号。
class ReadingAnswerPart(ReadingClaim):
    # 在实际生成字段上说明正文与引用的分工，而不只依赖外层指令。
    text: str = Field(
        min_length=1,
        max_length=1000,
        description=(
            "本段可直接阅读的正文，来源编号仅填写在 citation_ids，由程序统一排版；"
            "概括原文时不复制论文参考文献序号或历史回答中的数字引用标记。"
        ),
    )
    # 编号对应本次提供给模型的 Evidence，最终由 cited_answer 复查并重新编号。
    citation_ids: list[Annotated[int, Field(strict=True, ge=1)]] = Field(
        max_length=20,
        description=(
            "直接支持本段具体论点的最少必要片段 citation_id 整数数组；普通论点通常1至3条，"
            "跨文献比较通常每篇1至2条。同篇重复证据择优保留，不枚举所有材料或每段复制整组。"
            "数组长度是容错上限，不是数量目标。参考文献表、作者简介不能支持研究结论。"
            "从输入片段的 citation_id 选择，不使用文献 ID、页码、原文参考文献序号或历史编号。"
            "标题、过渡语和研究建议不引用原文时填写空数组。"
        ),
    )

    @model_validator(mode="after")
    def structured_citations_only(self):
        """确保引用只出现在结构化字段，正文展示编号由后端统一追加。

        Returns:
            未被改写的当前回答段落实例。

        Raises:
            ValueError: 正文直接包含方括号数字引用，可能与最终编号不一致。
        """
        if re.search(r"\[\d+\]", self.text):
            raise ValueError("正文引用编号必须放入 citation_ids，不能手写 [n]")
        return self


# 一轮追问的完整结构化输出；没有可用原文时必须明确说明不足原因。
# 约束追问回答或材料不足原因，使无依据结果有明确解释。
#
# Attributes:
#     answer: 最多十六个回答段落，可在不足时为空。
#     insufficient_reason: 无回答时必填的不足说明。
#     recommended_questions: 最多四个推荐后续问题。
class ReadingAnswer(BaseModel):
    # 最多十六个带结构化引用的回答段落；证据不足时允许为空。
    answer: list[ReadingAnswerPart] = Field(max_length=16)
    # 无回答正文时必填的解释，避免向用户发布空白成功结果。
    insufficient_reason: str = Field(max_length=500)
    # 后续可问的问题，允许在无可读材料时返回空列表。
    recommended_questions: list[RecommendedQuestion] = Field(max_length=4)

    @model_validator(mode="after")
    def answer_or_reason(self):
        """要求回答正文或不足原因至少有一项有效内容。

        Returns:
            校验通过的当前回答，不把推荐问题当作有效回答正文。

        Raises:
            ValueError: 回答段落为空且没有非空白的不足原因。
        """
        if not self.answer and not self.insufficient_reason.strip():
            raise ValueError("回答缺少内容或证据不足说明")
        return self
