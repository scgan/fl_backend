"""验证追问的校验错误反馈、有界重生成和原文受限发布，不按引用数量误杀。"""

import asyncio
import uuid

import pytest
from test_literature import FakeGateway, FakeMaterials, evidence

from app.agents.gateway import StructuredSchemaValidationError
from app.agents.literature_graph import MAX_FOLLOWUP_ATTEMPTS, LiteratureGraph
from app.agents.literature_schemas import ReadingAnswer


def answer_with(ids, *, repeat=1):
    """生成可控的引用集合，重复正文模拟截图中的整组挂引。"""
    return ReadingAnswer(
        answer=[{"text": f"第{position + 1}段讨论教师工作与支持条件。", "citation_ids": ids}
                for position in range(repeat)],
        insufficient_reason="", recommended_questions=[],
    )


@pytest.mark.parametrize("scenario", [
    "unknown_twice", "schema_once", "always_invalid", "many_valid", "auth_error", "insufficient",
])
def test_followup_prompt_validation_retry_and_restricted_result(scenario):
    """错误反馈保持固定证据；耗尽后给原文，合法多引用不误杀。"""
    materials = FakeMaterials()
    materials.items = [evidence(index, 12) for index in range(1, 21)]
    if scenario == "always_invalid":
        materials.items[0] = materials.items[0].model_copy(update={
            "content": materials.items[0].content + "原文参考序号[99]。",
        })
    calls = []

    class Gateway(FakeGateway):
        """模拟引用错误、格式错误、合法多引用及正常回答。"""

        async def structured(self, instruction, schema, **kwargs):
            """仅替换追问回答，保留真实问题规划与阅读图执行。"""
            if schema is ReadingAnswer:
                calls.append(instruction)
                if scenario == "auth_error":
                    raise RuntimeError("模型服务返回 HTTP 401")
                if scenario == "schema_once" and len(calls) == 1:
                    raise StructuredSchemaValidationError("ReadingAnswer", [{
                        "error_count": 1, "issues": [{"path": "answer[0].citation_ids",
                                                     "type": "int_type"}],
                    }])
                if scenario == "always_invalid" or (
                    scenario == "unknown_twice" and len(calls) < 3
                ):
                    return answer_with([999])
                if scenario == "many_valid":
                    return answer_with(list(range(1, 21)), repeat=3)
                if scenario == "insufficient":
                    return ReadingAnswer(answer=[], insufficient_reason="当前原文无法确认研究方法",
                                         recommended_questions=[])
                return answer_with([7, 13])
            return await super().structured(instruction, schema, **kwargs)

    async def step(name, state, operation):
        """执行真实阅读节点，省略数据库步骤事务。"""
        return await operation()

    graph = LiteratureGraph(materials, Gateway(), "依据原文回答", step)
    state = {
        "task": "ASK", "query": "两篇文献的区别", "mode": "FAST",
        "history": [{"role": "assistant", "content": "旧回答[1][2][3]"},
                    {"role": "user", "content": "引用[2]指什么"}],
    }
    if scenario == "auth_error":
        with pytest.raises(RuntimeError, match="401"):
            asyncio.run(graph.run(state))
        assert len(calls) == 1
        return
    result = asyncio.run(graph.run(state))
    payload = result["literature_result"]
    assert "旧回答" not in calls[0] and "引用[2]指什么" in calls[0]
    if scenario == "always_invalid":
        assert len(calls) == MAX_FOLLOWUP_ATTEMPTS
        assert payload["fallback_used"] is True
        assert "仅列出所选文献的原文片段" in result["final_answer"]
        assert materials.items[0].content.replace("[99]", "［99］") in result["final_answer"]
        assert "[99]" in result["validated_citations"][0].quote
        assert result["evidence_level"] == "INSUFFICIENT"
    elif scenario == "many_valid":
        assert len(calls) == 1 and payload["fallback_used"] is False
        assert len(result["validated_citations"]) == 20
    elif scenario == "insufficient":
        assert len(calls) == 1 and payload["fallback_used"] is False
        assert result["evidence_level"] == "INSUFFICIENT"
        assert result["final_answer"] == "当前原文无法确认研究方法"
    else:
        assert len(calls) == (3 if scenario == "unknown_twice" else 2)
        assert result["final_answer"].endswith("[1][2]")
        assert [item.chunk_id for item in result["validated_citations"]] == [
            uuid.UUID(int=7), uuid.UUID(int=13),
        ]
        assert "上轮校验失败" in calls[1]
        assert "第1段讨论" not in calls[1]
        assert payload["fallback_used"] is False
    assert payload["followup_attempts"] == len(calls)


@pytest.mark.parametrize("persistent", [False, True])
def test_current_single_document_does_not_resolve_ordinal_from_old_history(persistent):
    """当前只有第二篇时，历史中的旧第一篇ID必须重试或澄清，不能替换当前目标。"""
    materials = FakeMaterials()
    materials.documents = [item for item in materials.documents if item["corpus_id"] == 12]
    materials.items = [evidence(1, 12)]
    plans = []

    class Gateway(FakeGateway):
        """模拟模型首次误用历史ID，纠正后选择当前唯一文献。"""

        async def structured(self, instruction, schema, **kwargs):
            """记录指代重试，并转发正常回答生成。"""
            if schema.__name__ == "QuestionPlan":
                plans.append(instruction)
                return schema(standalone_question="当前文章如何分析难点？", corpus_ids=[
                    11 if persistent or len(plans) == 1 else 12,
                ])
            return await super().structured(instruction, schema, **kwargs)

    async def step(name, state, operation):
        """执行节点，不写运行记录。"""
        return await operation()

    result = asyncio.run(LiteratureGraph(materials, Gateway(), "只基于所选材料", step).run({
        "task": "ASK", "query": "第一篇如何分析难点？", "mode": "FAST",
        "history": [{"role": "assistant", "content": "旧列表第一篇是文献11，第二篇是文献12"}],
    }))
    assert "只能选择当前列表ID：[12]" in plans[1]
    assert result["literature_result"]["target_corpus_ids"] == [12]
    if persistent:
        assert len(plans) == MAX_FOLLOWUP_ATTEMPTS
        assert result["literature_result"]["needs_clarification"] is True
        assert result["validated_citations"] == []
        assert result["evidence_level"] == "INSUFFICIENT"
    else:
        assert len(plans) == 2
        assert {item.document_id.int for item in result["validated_citations"]} == {12}
