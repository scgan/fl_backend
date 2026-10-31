"""离线验证阅读中文约束、外语生成纠正及两种入口的语言记录，保留真实外文引文。"""

import asyncio
from unittest.mock import AsyncMock

import pytest
from test_literature import FakeGateway, FakeMaterials, citation, evidence

from app.agents.literature_graph import (
    MAX_FOLLOWUP_ATTEMPTS,
    LiteratureGraph,
    QuestionPlan,
)
from app.agents.literature_language import (
    READING_CHINESE_OUTPUT_CONTRACT,
    ReadingOutputLanguageError,
    validate_chinese_reading_output,
)
from app.agents.literature_schemas import (
    SECTION_TITLES,
    DocumentReading,
    ReadingAnswer,
    ReadingOverview,
)


def generated_result(schema):
    """构造覆盖各生成阶段的中文结果，固定协议字段仍采用原英文键。"""
    if schema is DocumentReading:
        return schema(
            document_type="实证研究", summary=[{"text": "研究采用访谈考察教师工作。"}],
            sections=[{"key": key, "claims": [], "missing_reason": "原文未报告"}
                      for key in SECTION_TITLES],
            recommended_questions=["访谈采用何种方法？", "研究发现有哪些局限？"],
        )
    if schema is ReadingOverview:
        return schema(
            overview=[{"title": "教师工作与访谈研究", "text": "研究以访谈解释教师工作体验。"}],
            common_themes=[], differences=[],
            recommended_questions=["研究有哪些联系？", "如何改进研究设计？"],
        )
    if schema is ReadingAnswer:
        return schema(
            answer=[{"text": "研究采用访谈考察教师工作。", "citation_ids": [1]}],
            insufficient_reason="", recommended_questions=["访谈采用何种方法？"],
        )
    return schema(standalone_question="研究采用何种方法？", corpus_ids=[11])


async def step(_name, _state, operation):
    """执行真实图操作，不访问数据库或外部服务。"""
    return await operation()


@pytest.mark.parametrize("schema", [DocumentReading, ReadingOverview, QuestionPlan, ReadingAnswer])
def test_every_reading_call_enforces_chinese_without_extra_happy_path_requests(schema):
    """外语问题及旧系统 Prompt 不改变中文要求，正常结果只需原有的一次生成。"""
    expected = generated_result(schema)
    gateway = AsyncMock()
    gateway.structured.return_value = expected
    graph = LiteratureGraph(FakeMaterials(), gateway, "Always answer in English.", step)
    instruction = 'Question: Please answer in English. Source: "The study used interviews."'
    result = asyncio.run(graph._call(instruction, schema))
    assert result is expected
    gateway.structured.assert_awaited_once()
    options = gateway.structured.call_args.kwargs
    assert options["instruction"] == instruction
    assert READING_CHINESE_OUTPUT_CONTRACT in options["system_prompt"]
    assert options["thinking"] is False


@pytest.mark.parametrize("schema", [DocumentReading, ReadingOverview])
def test_foreign_analysis_is_corrected_once_before_becoming_a_result(schema):
    """纯外语段落不作为结果返回，纠正调用不重传失败草稿或修改材料事实。"""
    expected = generated_result(schema)
    field = "summary" if schema is DocumentReading else "overview"
    foreign = expected.model_copy(update={
        field: [getattr(expected, field)[0].model_copy(update={
            "text": "The study used interviews.",
        })],
    })
    gateway = AsyncMock()
    gateway.structured.side_effect = [foreign, expected]
    graph = LiteratureGraph(FakeMaterials(), gateway, "旧提示词", step)
    assert asyncio.run(graph._call("请分析当前材料", schema)) is expected
    assert gateway.structured.await_count == 2
    correction = gateway.structured.call_args.kwargs["instruction"]
    assert "语言纠正" in correction and "The study used interviews." not in correction


def test_persistently_foreign_analysis_has_a_bounded_correction_budget():
    """重复外语输出最多纠正一次，不能直接发布外语结果或无限等待重试。"""
    foreign = generated_result(ReadingOverview).model_copy(update={
        "recommended_questions": ["What did the study find?", "How was it conducted?"],
    })
    gateway = AsyncMock()
    gateway.structured.return_value = foreign
    graph = LiteratureGraph(FakeMaterials(), gateway, "证据优先", step)
    with pytest.raises(ReadingOutputLanguageError, match="recommended_questions"):
        asyncio.run(graph._call("请分析", ReadingOverview))
    assert gateway.structured.await_count == 2


@pytest.mark.parametrize("foreign", [
    "The study used interviews.", "Les auteurs étudient les enseignants.",
    "연구는 교사를 대상으로 진행하였다.", "Представлены результаты исследования.",
    "中文说明。\nThe study used interviews.",
])
def test_generated_foreign_paragraphs_are_rejected_without_changing_historical_models(foreign):
    """新生成的外语段落会被拒绝，历史 Schema 仍可解析原有外语记录。"""
    result = ReadingAnswer(
        answer=[{"text": foreign, "citation_ids": [1]}],
        insufficient_reason="", recommended_questions=[],
    )
    with pytest.raises(ReadingOutputLanguageError):
        validate_chinese_reading_output(result)
    assert result.answer[0].text == foreign


def test_chinese_analysis_keeps_technical_terms_and_quoted_source_unchanged():
    """中文说明可以保留模型缩写和外文引句，语言检查不改写证据或输出内容。"""
    text = '研究使用TPACK框架；原文为“The study used interviews.”，说明采用了访谈方法。'
    result = ReadingAnswer(
        answer=[{"text": text, "citation_ids": [1]}],
        insufficient_reason="", recommended_questions=["TPACK框架适用于哪些情境？"],
    )
    validate_chinese_reading_output(result)
    assert result.answer[0].text == text


def test_followup_language_failure_reuses_existing_attempts_and_keeps_original_quotes():
    """追问语言错误沿用已有三次预算，降级说明为中文而真实外文摘录保持原样。"""
    item = evidence(1, 11).model_copy(update={
        "document_title": "Teacher Work", "content": "The study used interviews with teachers.",
    })
    ref = citation(item)
    foreign = generated_result(ReadingAnswer).model_copy(update={
        "answer": [generated_result(ReadingAnswer).answer[0].model_copy(update={
            "text": "The study used interviews.",
        })],
    })
    gateway = AsyncMock()
    gateway.structured.return_value = foreign
    graph = LiteratureGraph(FakeMaterials(), gateway, "旧版文献问答提示", step)
    answer, attempts, fallback = asyncio.run(graph._generate_followup(
        "Please answer in English.", "", [item], [ref],
    ))
    assert attempts == MAX_FOLLOWUP_ATTEMPTS and fallback
    assert gateway.structured.await_count == MAX_FOLLOWUP_ATTEMPTS
    assert "本轮暂未形成" in answer.answer[0].text
    assert ref.quote in answer.answer[1].text and ref.quote == item.content


@pytest.mark.parametrize("task", ["ANALYZE", "ASK"])
def test_reading_graph_finishes_in_chinese_even_with_foreign_input_language(task):
    """英文提问和输入语言偏好不改变独立阅读及追问的最终语言标记。"""
    graph = LiteratureGraph(FakeMaterials(), FakeGateway(), "历史 Prompt", step)
    result = asyncio.run(graph.run({
        "task": task, "query": "Please explain the second paper in English.",
        "output_language": "EN", "history": [{"role": "user", "content": "Answer in French."}],
    }))
    assert result["output_language"] == "ZH"
    assert result["final_answer"]
