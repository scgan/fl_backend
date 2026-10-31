"""验证写作需求澄清、明确阶段直达和独立写作规格，不请求在线模型或知识库。"""

import asyncio
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from docx import Document

from app.agents.ai_assistant_graph import AssistantGraph
from app.agents.ai_assistant_schemas import AssistantIntent
from app.agents.ai_assistant_state import AssistantState
from app.agents.quill_draft_graph import QuillDraftGraph
from app.agents.quill_graph import QuillGraph
from app.agents.quill_plan_graph import QuillPlanGraph
from app.agents.quill_revise_graph import QuillReviseGraph
from app.agents.quill_rq_graph import QuillRQGraph, _build_rq_answer
from app.agents.quill_schemas import QuillRQResult, ResearchQuestion
from app.foreign_integration.docx_export import _add_rq_section


async def tracked(name, state, operation):
    """执行实际节点操作，仅省略数据库步骤记录。"""
    return await operation()


def state(query, **extra):
    """创建授权范围已冻结的 AI 图输入。"""
    return AssistantState(
        request_id=uuid.uuid4().hex,
        user_id="7",
        conversation_id=uuid.uuid4().hex,
        query=query,
        knowledge_base_id=uuid.uuid4().hex,
        knowledge_base_version=1,
        scope_type="GLOBAL",
        history=[],
        mode="FAST",
        **extra,
    )


@pytest.mark.parametrize(
    "classification",
    [
        {"intent": "WRITING", "search_query": "双减政策"},
        {
            "intent": "WRITING",
            "search_query": "双减政策",
            "writing_stage": "PLAN",
            "needs_clarification": True,
        },
    ],
)
def test_unclear_writing_never_defaults_to_plan_or_starts_generation(classification):
    """缺失阶段和显式歧义均要求选择成果，澄清轮不检索、不启动写作。"""
    retrieval = SimpleNamespace(search=AsyncMock())
    runner = AsyncMock()
    gateway = SimpleNamespace(structured=AsyncMock(return_value=AssistantIntent(**classification)))
    graph = AssistantGraph(retrieval, gateway, "", tracked, writing_runner=runner)
    result = asyncio.run(graph.run(state("我想写双减政策论文，帮帮我")))
    card = result["assistant_result"]["writing_result"]
    assert card["stage"] is None and card["task_selection_mode"] == "CHOOSE"
    assert "stage" in card["missing_fields"] and "research_topic" in card["missing_fields"]
    retrieval.search.assert_not_awaited()
    runner.assert_not_awaited()


def test_explicit_plan_keeps_pure_topic_and_requirements_without_rq_prerequisite():
    """明确要提纲就固定 PLAN；主题和规格独立保存，不要求先生成研究问题。"""
    intent = AssistantIntent(
        intent="WRITING",
        search_query="双减政策",
        writing_stage="PLAN",
        research_topic="双减政策",
        writing_requirements="分三级标题",
    )
    gateway = SimpleNamespace(structured=AsyncMock(return_value=intent))
    graph = AssistantGraph(SimpleNamespace(), gateway, "", tracked)
    result = asyncio.run(graph.run(state("为双减政策设计论文提纲，分三级标题")))
    card = result["assistant_result"]["writing_result"]
    assert card["stage"] == "PLAN" and card["task_selection_mode"] == "FIXED"
    assert card["research_topic"] == "双减政策"
    assert card["writing_requirements"] == "分三级标题"
    assert card["missing_fields"] == ["knowledge_base_id"]
    assert result["assistant_result"]["query"] == "为双减政策设计论文提纲，分三级标题"


@pytest.mark.parametrize(
    ("query", "stage"),
    [
        ("围绕双减政策先推荐选题，再生成论文提纲", "RQ"),
        ("先选题，然后生成提纲", "RQ"),
        ("先提出三个研究问题，再设计论文大纲", "RQ"),
        ("不用先推荐选题，再生成提纲，直接给我提纲", None),
        ("先解释如何选题，再生成提纲", None),
    ],
)
def test_ordered_writing_uses_first_explicit_creation_stage_only(query, stage):
    """模型即使误判顺序需求为模糊，也只在肯定的生成指令下固定第一阶段。"""
    gateway = SimpleNamespace(
        structured=AsyncMock(
            return_value=AssistantIntent(
                intent="WRITING", search_query="双减政策", needs_clarification=True
            )
        )
    )
    graph = AssistantGraph(SimpleNamespace(), gateway, "", tracked)
    result = asyncio.run(graph._intent_classifier(state(query)))
    assert result["writing_stage"] == stage
    assert result["needs_clarification"] == (stage is None)


@pytest.mark.parametrize(
    ("query", "intent", "stage"),
    [
        ("双减政策论文提纲设计", "WRITING", "PLAN"),
        ("围绕双减政策推荐三个论文选题", "WRITING", "RQ"),
        ("基于文献提出三个双减政策研究问题", "WRITING", "RQ"),
        ("我想写双减政策论文，帮帮我", "WRITING", None),
        ("如何提出一个好的研究问题？", "QA", None),
        ("教师负担有哪些影响因素？", "QA", None),
        ("推荐几篇双减政策相关论文", "SEARCH", None),
    ],
)
def test_offline_classifier_separates_creation_consultation_and_search(query, intent, stage):
    """受限离线分类也区分生成问题、回答问题和查文献，且不猜测主题。"""
    gateway = SimpleNamespace(
        structured=AsyncMock(side_effect=RuntimeError("当前模型 Provider 不支持结构化生成"))
    )
    graph = AssistantGraph(SimpleNamespace(), gateway, "", tracked)
    result = asyncio.run(graph._intent_classifier(state(query)))
    assert result["intent"] == intent and result["writing_stage"] == stage
    assert result["research_topic"] == ""


@pytest.mark.parametrize(
    ("graph_type", "method"),
    [
        (QuillGraph, "_analysis_instruction"),
        (QuillRQGraph, "_synthesis_instruction"),
        (QuillPlanGraph, "_synthesis_instruction"),
        (QuillDraftGraph, "_synthesis_instruction"),
        (QuillReviseGraph, "_synthesis_instruction"),
    ],
)
def test_each_writing_stage_receives_requirements_separately(graph_type, method):
    """写作规格真正进入各子图模型输入，而不是只在卡片展示或混入检索主题。"""
    graph = graph_type(SimpleNamespace(), SimpleNamespace(), "", tracked)
    instruction = getattr(graph, method)(
        {
            "research_topic": "双减政策",
            "writing_requirements": "重点讨论教师支持机制",
            "output_language": "ZH",
            "evidence": [],
            "literature_limit": 10,
            "query": "待修改文本",
            "revise_instruction": "补充文献支持",
        }
    )
    assert "重点讨论教师支持机制" in instruction and "用户写作要求" in instruction


def test_rq_candidates_include_titles_objects_and_scope_without_breaking_old_artifacts():
    """新增选题信息可呈现，旧产物缺少这些字段仍能恢复。"""
    old = ResearchQuestion(
        question="教师支持机制如何改善教师工作负担？", rationale="已有研究[1]", citation_ids=[1]
    )
    question = old.model_copy(
        update={
            "proposed_title": "双减政策下教师支持机制研究",
            "research_object": "教师",
            "research_scope": "建议聚焦课后服务，地区待确定",
        }
    )
    result = QuillRQResult(
        status="SUCCESS",
        stage="RQ",
        research_topic="双减政策",
        language="ZH",
        citation_style="GBT7714",
        research_questions=[question],
    )
    answer = _build_rq_answer(result)
    assert "候选题目：双减政策下教师支持机制研究" in answer
    assert "研究对象（建议）：教师" in answer
    assert "地区待确定" in answer and old.proposed_title == ""
    document = Document()
    _add_rq_section(document, result.model_dump(mode="json"))
    text = "\n".join(paragraph.text for paragraph in document.paragraphs)
    assert "候选题目：双减政策下教师支持机制研究" in text
    assert "研究对象（建议）：教师" in text and "地区待确定" in text
