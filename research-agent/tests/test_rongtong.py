"""验收融通事实一致、语言转换、受众适配、真实引用及 API 权限边界。"""

from __future__ import annotations

import asyncio
import json
import re
import uuid
from pathlib import Path

import pytest
from test_suzhi import _citation, _evidence

from app.agents.rongtong_graph import RONGTONG_NODES, RongtongGraph
from app.agents.rongtong_schemas import (
    CommunicationDraft,
    CommunicationUnit,
    ConceptFacts,
    ConceptInterpretationDraft,
    CulturalRisk,
    CulturalRiskDraft,
    PolicyFact,
    PolicyFacts,
    PolicyInterpretationDraft,
    PolicyRequestPlan,
    RongtongResult,
    RongtongRunCreate,
)
from app.common.responses import AppError
from app.retrieval.schemas import RetrievalResponse

GOLD = json.loads((Path(__file__).parent / "data/rongtong_gold.json").read_text(encoding="utf-8"))


class PolicyRetrieval:
    """返回可控来源，用于全图和真实持久化调用链验收。"""

    def __init__(self, knowledge_base_id, *, empty=False, version=7):
        """准备一条政策证据。

        Args:
            knowledge_base_id: 当前测试库。
            empty: 证据不足情形。
            version: 冻结版本匹配或冲突情形。
        """
        # 真实映射格式由既有 Evidence/Citation 工厂构造。
        self.knowledge_base_id = knowledge_base_id
        self.items = (
            []
            if empty
            else [_evidence("双减研究", 2024, "双减政策旨在减轻学生作业和校外培训负担。")]
        )
        self.version = version
        self.calls = []

    async def search(self, **kwargs):
        """返回固定版本 Evidence。

        Args:
            kwargs: 图提交的范围、查询及模式。

        Returns:
            与真实检索契约一致的响应。
        """
        self.calls.append(kwargs)
        query = str(kwargs["query"])
        return RetrievalResponse(
            knowledge_base_id=self.knowledge_base_id,
            knowledge_base_version=self.version,
            original_query=query,
            detected_language="zh" if re.search(r"[\u4e00-\u9fff]", query) else "en",
            rewritten_query=query,
            expanded_queries={"zh": "双减", "en": "Double Reduction"},
            candidate_count=len(self.items),
            evidence=self.items,
            citations=[_citation(item, self.knowledge_base_id) for item in self.items],
            insufficient_evidence=not self.items,
            message=None,
        )


def _fixture_language(language: str, zh: str, en: str) -> str:
    """为受控模型提供指定语言的同义事实，不替代外部模型质量验收。

    Args:
        language: 测试输入的冻结输出语言。
        zh: 已有来源支持的中文表达。
        en: 同一事实的英文表达。

    Returns:
        单语或双语测试文本。
    """
    return en if language == "EN" else f"{zh}\n{en}" if language == "BILINGUAL" else zh


class PolicyGateway:
    """受控模型替身，验证编排和发布契约，不替代真实模型效果验收。"""

    def __init__(self, *, bad_citation=False, subject_type="POLICY"):
        """配置纠错和失败情形。

        Args:
            bad_citation: 提取事实引用越界。
            subject_type: 模型替身识别结果，验证图不按关键词猜测类型。
        """
        # 由模型提供的类型用于选择政策或概念内容 Schema。
        self.subject_type = subject_type
        self.calls = []
        self.bad_citation = bad_citation
        self.generation_count = 0

    async def structured(self, *, schema, instruction, **_kwargs):
        """按节点契约返回输出并核对生成受众指令。

        Args:
            schema: 业务契约。
            instruction: 图提交的受控指令。
            _kwargs: 固定 Prompt 与模型参数。

        Returns:
            固定且有来源的输出。
        """
        self.calls.append((schema, instruction))
        if schema is PolicyRequestPlan:
            audience = next(
                (item["audience"] for item in GOLD if item["audience_phrase"] in instruction),
                "INTERNATIONAL_RESEARCHER",
            )
            style = next(
                (
                    style_code
                    for style_phrase, style_code in {
                        "学术风格": "ACADEMIC",
                        "通俗风格": "POPULAR",
                        "媒体风格": "MEDIA",
                        "简报风格": "BRIEFING",
                    }.items()
                    if style_phrase in instruction
                ),
                None,
            )
            length_match = re.search(r"(\d+)\s*(?:字|词)", instruction)
            return PolicyRequestPlan(
                subject_type=self.subject_type,
                topic=(
                    "Double Reduction policy"
                    if "What is Double Reduction" in instruction
                    else "双减政策"
                ),
                target_audience=audience,
                style=style,
                desired_length=int(length_match.group(1)) if length_match else None,
            )
        if schema in {PolicyFacts, ConceptFacts}:
            assert "指定输出语言" not in instruction, "首次事实抽取不得接收目标输出语言"
            assert "用户原始请求（仅限定关注范围，不作为证据）" in instruction
            fact = PolicyFact(
                text="双减旨在减轻学生作业和校外培训负担。",
                citation_ids=[99 if self.bad_citation else 1],
            )
            return (
                ConceptFacts(definition=[fact])
                if schema is ConceptFacts
                else PolicyFacts(objectives=[fact])
            )
        if schema in {PolicyInterpretationDraft, ConceptInterpretationDraft, CulturalRiskDraft}:
            language_match = re.search(r"指定输出语言：(ZH|EN|BILINGUAL)", instruction)
            assert language_match, "解读与文化解释节点必须收到冻结输出语言"
            language = language_match.group(1)
            if schema in {PolicyInterpretationDraft, ConceptInterpretationDraft}:
                fact = PolicyFact(
                    text=_fixture_language(
                        language,
                        "双减旨在减轻学生作业和校外培训负担。",
                        "Double Reduction aims to reduce homework and off-campus training burdens.",
                    ),
                    citation_ids=[99 if self.bad_citation else 1],
                )
                return (
                    ConceptInterpretationDraft(
                        localized_topic=_fixture_language(
                            language, "双减政策", "Double Reduction policy"
                        ),
                        facts=ConceptFacts(definition=[fact]),
                    )
                    if schema is ConceptInterpretationDraft
                    else PolicyInterpretationDraft(
                        localized_topic=_fixture_language(
                            language, "双减政策", "Double Reduction policy"
                        ),
                        facts=PolicyFacts(objectives=[fact]),
                    )
                )
            return CulturalRiskDraft(
                risks=[
                    CulturalRisk(
                        misunderstanding=_fixture_language(
                            language,
                            "读者可能误以为这是取消所有作业。",
                            "Readers may mistake this for abolishing all homework.",
                        ),
                        explanation=_fixture_language(
                            language,
                            "减轻负担不等于取消作业。",
                            "Reducing burdens does not mean abolishing homework.",
                        ),
                        citation_ids=[1],
                    )
                ]
            )
        if schema is CommunicationDraft:
            assert "用户原始请求（仅内容范围与表达偏好" in instruction
            self.generation_count += 1
            language_match = re.search(r"语言：(ZH|EN|BILINGUAL)", instruction)
            assert language_match, "传播生成必须收到冻结输出语言"
            language = language_match.group(1)
            audience = next(
                a
                for a in (
                    "FOREIGN_TEACHER",
                    "GENERAL_PUBLIC",
                    "INTERNATIONAL_RESEARCHER",
                    "INTERNATIONAL_STUDENT",
                    "POLICY_MAKER",
                    "MEDIA",
                )
                if f"受众：{a}；" in instruction
            )
            focus = {
                "FOREIGN_TEACHER": "For teachers in schools",
                "GENERAL_PUBLIC": "What problem is addressed?",
                "INTERNATIONAL_RESEARCHER": "Policy objectives and research boundaries",
                "INTERNATIONAL_STUDENT": "For international students",
                "POLICY_MAKER": "Decision briefing",
                "MEDIA": "Policy news background",
            }[audience]
            assert "事实 ID" not in instruction
            return CommunicationDraft(
                units=[
                    CommunicationUnit(
                        text_zh="双减旨在减轻学生作业和校外培训负担。[1] 这不等于取消所有作业。[1]",
                        text_en=(
                            f"{focus}: Double Reduction aims to reduce homework and off-campus "
                            "training burdens.[1] This does not imply that all homework is "
                            "abolished.[1]"
                        ),
                    )
                ],
            )
        raise AssertionError(schema)


def graph_run(
    *,
    language="BILINGUAL",
    gateway=None,
    retrieval=None,
    query="解释双减政策",
):
    """运行全图并返回状态和真实节点顺序。

    Args:
        language: 必选语言。
        gateway: 可控模型替身。
        retrieval: 可控检索替身。
        query: 原始自然语言请求。

    Returns:
        最终状态、节点名、检索和模型替身。
    """
    kb = retrieval.knowledge_base_id if retrieval else uuid.uuid4()
    retrieval = retrieval or PolicyRetrieval(kb)
    gateway = gateway or PolicyGateway()
    nodes = []

    async def step(name, _state, operation):
        """记录真实调用阶段。

        Args:
            name: 当前图节点。
            _state: 当前专业状态。
            operation: 当前阶段操作。

        Returns:
            阶段输出。
        """
        nodes.append(name)
        return await operation()

    graph = RongtongGraph(retrieval, gateway, "安全测试 Prompt", step)
    final = asyncio.run(
        graph.run(
            dict(
                knowledge_base_id=str(kb),
                knowledge_base_version=7,
                query=query,
                policy_or_concept=query,
                target_language=language,
            )
        )
    )
    return final, nodes, retrieval, gateway


@pytest.mark.parametrize("case", GOLD, ids=lambda case: case["id"])
def test_rongtong_gold_audience_language_style(case):
    """逐条运行 Gold 参数组合，验证共同事实、受众规则及双语来源。

    Args:
        case: 已标注受众、语言和风格的用例。
    """
    style_phrase = {
        "ACADEMIC": "学术风格",
        "POPULAR": "通俗风格",
        "MEDIA": "媒体风格",
        "BRIEFING": "简报风格",
    }[case["style"]]
    final, nodes, retrieval, gateway = graph_run(
        query=f"{case['query']}采用{style_phrase}。", language=case["language"]
    )
    result = final["rongtong_result"]
    assert nodes == list(RONGTONG_NODES)
    assert result.target_audience == case["audience"] and result.style == case["style"]
    assert result.target_language == case["language"]
    assert result.subject_type == "POLICY"
    assert result.facts.objectives[0].text == _fixture_language(
        case["language"],
        "双减旨在减轻学生作业和校外培训负担。",
        "Double Reduction aims to reduce homework and off-campus training burdens.",
    )
    assert final["used_citation_indices"] == [1]
    assert retrieval.calls[0]["query"] == "双减政策"
    assert retrieval.calls[0]["evidence_limit"] == 20
    assert final["reviewer_status"] == final["citation_validation_status"] == "PASS"
    if case["language"] in {"EN", "BILINGUAL"}:
        assert "Double Reduction" in result.communication_text
        assert "Dual Reduction" not in result.communication_text
    assert gateway.calls[0][0] is PolicyRequestPlan


def test_rongtong_three_audiences_same_facts_different_expression():
    """M8 三类验收受众应共同事实一致而传播正文不同。"""
    results = [
        graph_run(query=f"面向{audience_phrase}解释双减政策", language="EN")[0]["rongtong_result"]
        for audience_phrase in ("国际研究者", "国外教师", "普通公众")
    ]
    assert len({result.facts.model_dump_json() for result in results}) == 1
    assert len({result.communication_text for result in results}) == 3


def test_rongtong_natural_language_specs_are_used_without_extra_generation():
    """自然语言规格被识别，传播生成保持单次模型调用。"""
    final, _, _, gateway = graph_run(
        query="面向国外教师以简报风格解释双减，控制在 300 字",
    )
    assert final["rongtong_result"].target_audience == "FOREIGN_TEACHER"
    assert final["rongtong_result"].style == "BRIEFING"
    assert final["rongtong_result"].desired_length == 300
    assert gateway.generation_count == 1
    assert all(schema.__name__ != "PolicyReview" for schema, _ in gateway.calls)


def test_rongtong_empty_evidence_and_version_conflict():
    """证据不足不调用事实模型，版本冲突不得继续生成。"""
    final, _, _, gateway = graph_run(retrieval=PolicyRetrieval(uuid.uuid4(), empty=True))
    assert final["rongtong_result"].status == "INSUFFICIENT_EVIDENCE"
    assert final["validated_citations"] == [] and len(gateway.calls) == 1
    with pytest.raises(ValueError, match="知识库版本"):
        graph_run(retrieval=PolicyRetrieval(uuid.uuid4(), version=8))
    bad_gateway = PolicyGateway(bad_citation=True)
    with pytest.raises(ValueError, match="越界 Citation"):
        graph_run(gateway=bad_gateway)
    assert sum(schema is PolicyFacts for schema, _ in bad_gateway.calls) == 2


@pytest.mark.parametrize("invalid_part", ["citation_ids", "inline_marker"])
def test_rongtong_out_of_range_fact_citation_retries_structure_once(invalid_part):
    """首次事实引用越界只重做结构提取，复用已固定 Evidence 后继续发布。

    Args:
        invalid_part: 首轮越界发生在结构化编号或事实正文原位引用。
    """

    class CorrectingCitationGateway(PolicyGateway):
        """仅首次事实抽取返回越界编号的受控模型替身。"""

        def __init__(self):
            """初始化事实抽取轮次。"""
            super().__init__()
            self.extraction_count = 0

        async def structured(self, *, schema, instruction, **kwargs):
            """首次生成越界事实，随后返回有来源的同一事实。

            Args:
                schema: 当前结构化输出契约。
                instruction: 当前阶段指令。
                **kwargs: 固定模型参数。

            Returns:
                当前轮次对应的结构化输出。
            """
            output = await super().structured(schema=schema, instruction=instruction, **kwargs)
            if schema is PolicyFacts:
                self.extraction_count += 1
                if self.extraction_count == 1:
                    if invalid_part == "citation_ids":
                        output.objectives[0].citation_ids = [99]
                    else:
                        output.objectives[0].text += "[99]"
            return output

    gateway = CorrectingCitationGateway()
    final, nodes, retrieval, _ = graph_run(gateway=gateway)
    assert final["rongtong_result"].status == "SUCCESS"
    assert final["structure_attempts"] == 2
    assert nodes.count("PolicyStructureExtraction") == 2
    assert nodes.count("PolicyAnalysisValidation") == 1
    assert len(retrieval.calls) == 1
    assert (
        "上轮校验失败"
        in [instruction for schema, instruction in gateway.calls if schema is PolicyFacts][1]
    )


def test_rongtong_translation_retries_once_after_validation_failure():
    """语言转换首次改变事实数量时纠正一次，持续错误仍拒绝发布。"""

    class DroppingTranslationGateway(PolicyGateway):
        """模拟目标语言转换在指定次数内遗漏整个事实分区。"""

        def __init__(self, *, always_drop: bool = False):
            """配置只破坏首次转换或持续破坏每次转换。

            Args:
                always_drop: 是否让纠正后的第二次结果继续遗漏事实。
            """
            super().__init__()
            self.always_drop = always_drop
            self.translation_count = 0

        async def structured(self, *, schema, instruction, **kwargs):
            """在保留标准模型替身行为后按配置删除翻译结果事实。

            Args:
                schema: 当前结构化输出契约。
                instruction: 当前节点指令。
                **kwargs: 固定 Prompt 与模型参数。

            Returns:
                首次抽取结果或被破坏的目标语言结果。
            """
            result = await super().structured(
                schema=schema,
                instruction=instruction,
                **kwargs,
            )
            if schema is PolicyInterpretationDraft:
                self.translation_count += 1
                if self.always_drop or self.translation_count == 1:
                    result.facts.objectives = []
            return result

    gateway = DroppingTranslationGateway()
    final, _, _, _ = graph_run(gateway=gateway)
    assert final["rongtong_result"].status == "SUCCESS"
    assert gateway.translation_count == 2
    translation_instructions = [
        instruction for schema, instruction in gateway.calls if schema is PolicyInterpretationDraft
    ]
    assert "上一次翻译未通过后端校验" in translation_instructions[1]
    assert "融通事实语言转换改变了分区事实数量" in translation_instructions[1]
    assert '"objectives": 1' in translation_instructions[1]

    persistent_gateway = DroppingTranslationGateway(always_drop=True)
    with pytest.raises(ValueError, match="改变了分区事实数量"):
        graph_run(gateway=persistent_gateway)
    assert persistent_gateway.translation_count == 2


def test_rongtong_requires_language_and_rejects_blank():
    """输出语言无默认，空白主题、未知规格和已移除参数必须被拒绝。"""
    data = dict(
        request_id="rongtong-schema-test", knowledge_base_id=uuid.uuid4(), policy_or_concept="双减"
    )
    with pytest.raises(ValueError):
        RongtongRunCreate(**data)
    with pytest.raises(ValueError):
        RongtongRunCreate(**{**data, "policy_or_concept": "   "}, target_language="EN")
    with pytest.raises(ValueError):
        RongtongRunCreate(**data, target_language="FR")
    with pytest.raises(ValueError):
        RongtongRunCreate(**data, target_language="EN", desired_length=300)


def test_rongtong_length_defaults_and_clamps_natural_language_value():
    """自然语言未描述篇幅时使用默认值，超出边界时收敛到安全范围。"""
    default = graph_run()[0]["rongtong_result"]
    too_short = graph_run(query="用 20 字解释双减政策")[0]["rongtong_result"]
    too_long = graph_run(query="用 10000 字解释双减政策")[0]["rongtong_result"]
    assert default.desired_length == 800
    assert too_short.desired_length == 100
    assert too_long.desired_length == 5000


@pytest.mark.parametrize(
    "field,value",
    [
        ("document_id", uuid.uuid4()),
        ("chunk_id", uuid.uuid4()),
        ("page_start", 99),
        ("quote", "不存在的原文引文"),
    ],
)
def test_rongtong_rejects_fabricated_source_mapping(field, value):
    """错误文献、Chunk、页码与引文不能进入消息 Citation。

    Args:
        field: 被污染的来源字段。
        value: 不对应 Evidence 的字段值。
    """
    final, _, retrieval, gateway = graph_run()
    final["citations"][0] = final["citations"][0].model_copy(update={field: value})
    graph = RongtongGraph(retrieval, gateway, "", None)
    with pytest.raises(ValueError, match="不合法"):
        graph._finalize(final)


def test_rongtong_rejects_missing_bilingual_text_and_invalid_citation():
    """传播正文必须包含双语字段，且每段引用只能指向本次真实 Evidence。"""
    from app.agents.rongtong_graph import _validate_communication

    final, _, _, _ = graph_run()
    with pytest.raises(ValueError, match="缺失目标语言"):
        _validate_communication(
            CommunicationDraft(units=[CommunicationUnit(text_zh="双减政策。[1]")]),
            final,
        )
    with pytest.raises(ValueError, match="不属于本次 Evidence"):
        _validate_communication(
            CommunicationDraft(
                units=[
                    CommunicationUnit(
                        text_zh="双减减轻相关负担。[2]",
                        text_en="Double Reduction reduces related burdens.[2]",
                    )
                ],
            ),
            final,
        )


def test_rongtong_communication_may_leave_risks_in_their_dedicated_tab():
    """传播正文可以不重复跨文化页签内容，也不需要维护风险 ID。"""
    from app.agents.rongtong_graph import _validate_communication

    final, _, _, _ = graph_run(language="EN")
    _validate_communication(
        CommunicationDraft(
            units=[
                CommunicationUnit(
                    text_en="Double Reduction reduces homework and training burdens.[1]",
                )
            ],
        ),
        final,
    )


def test_rongtong_keeps_sentence_level_citations_without_paragraph_suffix():
    """传播正文保留模型句级引用，不再把引用统一追加到段尾。"""
    final, _, _, gateway = graph_run(language="EN")
    text = final["rongtong_result"].communication_text
    assert "burdens.[1] This does not imply" in text
    assert text.endswith("abolished.[1]")
    assert "系统会添加真实 Citation" not in next(
        instruction for schema, instruction in gateway.calls if schema is CommunicationDraft
    )


def test_rongtong_rejects_missing_or_mismatched_sentence_citations():
    """传播正文缺少引用或使用非本次 Evidence 编号时必须拒绝发布。"""
    from app.agents.rongtong_graph import _validate_communication

    final, _, _, _ = graph_run(language="EN")
    for text in ("Double Reduction reduces burdens.", "Double Reduction reduces burdens.[2]"):
        with pytest.raises(ValueError, match="Citation 缺失或不属于本次 Evidence"):
            _validate_communication(
                CommunicationDraft(
                    units=[
                        CommunicationUnit(
                            text_en=text,
                        )
                    ]
                ),
                final,
            )


def test_rongtong_media_style_reaches_generation_instruction():
    """媒体风格采用明确指令，融通运行不依赖术语词条。"""
    from app.agents.rongtong_graph import STYLE_PROFILES

    final, _, _, gateway = graph_run(query="面向媒体读者以媒体风格解释双减政策", language="EN")
    assert final["rongtong_result"].style == "MEDIA"
    assert any(STYLE_PROFILES["MEDIA"] in instruction for _, instruction in gateway.calls)


@pytest.mark.parametrize("always_invalid", [False, True])
def test_rongtong_reviewer_retries_communication_once(always_invalid):
    """正文规则审核失败只重写正文一次，连续失败仍拒绝发布。

    Args:
        always_invalid: 是否让第二轮正文继续使用越界引用。
    """

    class Gateway(PolicyGateway):
        """按测试情形在传播正文中放入越界引用。"""

        async def structured(self, *, schema, **kwargs):
            """按测试情形注入无效编号。

            Args:
                schema: 当前输出契约。
                kwargs: 受控模型参数。

            Returns:
                按测试情形生成的结构化结果。
            """
            value = await super().structured(schema=schema, **kwargs)
            if schema is CommunicationDraft and (always_invalid or self.generation_count == 1):
                value.units[0].text_zh = "双减政策。[2]"
                value.units[0].text_en = "Double Reduction policy.[2]"
            return value

    gateway = Gateway()
    if always_invalid:
        with pytest.raises(AppError, match="Citation 缺失或不属于本次 Evidence"):
            graph_run(gateway=gateway)
    else:
        final, nodes, retrieval, _ = graph_run(gateway=gateway)
        assert final["rongtong_result"].status == "SUCCESS"
        assert final["communication_attempts"] == 2
        assert nodes.count("TranslationOrBilingualGeneration") == 2
        assert nodes.count("ResearchReviewer") == 2
        assert len(retrieval.calls) == 1
        assert (
            "上轮审核失败"
            in [
                instruction for schema, instruction in gateway.calls if schema is CommunicationDraft
            ][1]
        )
    assert gateway.generation_count == 2


@pytest.mark.parametrize("subject_type", ["POLICY", "CONCEPT"])
@pytest.mark.parametrize("language", ["ZH", "EN", "BILINGUAL"])
def test_rongtong_llm_type_selects_schema_and_all_tabs_follow_language(subject_type, language):
    """模型类型决定 Schema；三个页签内容遵循同一必选语言。

    Args:
        subject_type: 模型识别结果，即使原话包含政策字样也可为概念。
        language: 用户必选输出语言。
    """
    gateway = PolicyGateway(subject_type=subject_type)
    final, nodes, _, _ = graph_run(language=language, gateway=gateway, query="解释双减政策中的概念")
    result = final["rongtong_result"]
    assert result.subject_type == subject_type and nodes == list(RONGTONG_NODES)
    assert isinstance(result.facts, ConceptFacts if subject_type == "CONCEPT" else PolicyFacts)
    restored = RongtongResult.model_validate_json(result.model_dump_json())
    assert isinstance(restored.facts, ConceptFacts if subject_type == "CONCEPT" else PolicyFacts)
    facts = result.facts.definition if subject_type == "CONCEPT" else result.facts.objectives
    texts = [
        facts[0].text,
        result.cultural_risks[0].misunderstanding,
        result.cultural_risks[0].explanation,
        result.communication_text,
        *result.limitations,
    ]
    if language == "EN":
        assert all(not re.search(r"[\u4e00-\u9fff]", text) for text in texts)
    else:
        assert all(re.search(r"[\u4e00-\u9fff]", text) for text in texts)
    if language == "BILINGUAL":
        assert all(re.search(r"[A-Za-z]{2,}", text) for text in texts)
    assert final["used_citation_indices"] == [1]
    interpretation_schema = (
        ConceptInterpretationDraft if subject_type == "CONCEPT" else PolicyInterpretationDraft
    )
    extraction_schema = ConceptFacts if subject_type == "CONCEPT" else PolicyFacts
    extraction_instruction = next(
        instruction for schema, instruction in gateway.calls if schema is extraction_schema
    )
    translation_instruction = next(
        instruction for schema, instruction in gateway.calls if schema is interpretation_schema
    )
    assert "指定输出语言" not in extraction_instruction
    assert "按语义判断主题相关性" in extraction_instruction
    assert "将下面 JSON 中的所有字符串值翻译" in translation_instruction
    assert "键、层级、数组长度、元素顺序和数字值保持不变" in translation_instruction
    assert '"localized_topic":' in translation_instruction
    assert '"facts":' in translation_instruction
    assert sum(schema is extraction_schema for schema, _ in gateway.calls) == 1
    assert sum(schema is interpretation_schema for schema, _ in gateway.calls) == 1


def test_rongtong_interpretation_nodes_generate_display_language_before_communication():
    """解读与文化节点直接生成英文展示字段，传播节点只生成正文。"""
    gateway = PolicyGateway()
    final, _, _, _ = graph_run(language="EN", gateway=gateway, query="解释双减政策")
    assert final["policy_facts"].objectives[0].text.startswith("Double Reduction")
    assert final["cultural_risks"][0].explanation.startswith("Reducing burdens")
    result = final["rongtong_result"]
    display_texts = [
        result.policy_or_concept,
        result.facts.objectives[0].text,
        result.cultural_risks[0].misunderstanding,
        result.cultural_risks[0].explanation,
        result.communication_text,
    ]
    assert all(not re.search(r"[\u4e00-\u9fff]", text) for text in display_texts)
    assert sum(schema is CommunicationDraft for schema, _ in gateway.calls) == 1


def test_rongtong_english_question_chinese_output_is_generated_by_interpretation_nodes():
    """英文提问时，解读节点仍按冻结输出语言直接生成中文展示内容。"""
    final, _, retrieval, _ = graph_run(language="ZH", query="What is Double Reduction policy?")

    assert retrieval.calls[0]["query"] == "Double Reduction policy"
    assert final["original_language"] == "en"
    assert final["policy_facts"].objectives[0].text.startswith("双减")
    result = final["rongtong_result"]
    display_texts = [
        result.policy_or_concept,
        result.facts.objectives[0].text,
        result.cultural_risks[0].misunderstanding,
        result.cultural_risks[0].explanation,
        result.communication_text,
    ]
    assert all(re.search(r"[\u4e00-\u9fff]", text) for text in display_texts)
