"""覆盖Quill POLISH 划词润改的操作契约、API 白名单与纯创作发布边界。"""

from __future__ import annotations

import asyncio
import uuid

from fastapi import HTTPException

from app.agents.gateway import ModelGateway
from app.agents.quill_polish_graph import ACTION_BRIEFS, QuillPolishGraph
from app.agents.quill_schemas import PolishOutput


class FakePolishGateway(ModelGateway):
    """返回固定润改正文的结构化测试网关。"""

    def __init__(self) -> None:
        """记录收到的指令，供断言操作语义进入模型输入。"""
        self.instructions: list[str] = []

    async def generate_answer(self, **_kwargs) -> str:
        """满足网关抽象；润改图只使用结构化接口。"""
        return ""

    async def structured(self, *, instruction, schema, **_kwargs):
        """按 Schema 返回固定润改结果，并保留指令供审计断言。"""
        if schema is PolishOutput:
            self.instructions.append(instruction)
            return PolishOutput(output_text="这是润改后的正文。")
        raise AssertionError(f"未处理的 Schema：{schema}")


def _state(**overrides) -> dict[str, object]:
    """构造完整润改图回归所需的冻结输入。"""
    state = {
        "request_id": "quill-polish-test",
        "user_id": str(uuid.uuid4()),
        "conversation_id": str(uuid.uuid4()),
        "query": "是否应该对文献实施向量化？",
        "selected_text": "是否应该对文献实施向量化？",
        "context_before": "我有10万篇文献，想基于向量做语义检索，",
        "context_after": "",
        "action": "POLISH",
        "output_language": "AUTO",
        "knowledge_base_id": str(uuid.uuid4()),
        "knowledge_base_version": 7,
        "mode": "FAST",
        "agent_route": [],
    }
    state.update(overrides)
    return state


def test_quill_polish_graph_publishes_creation_result() -> None:
    """润改图发布纯创作结果：无 Citation、证据等级 INSUFFICIENT。"""
    tracked_nodes: list[str] = []

    async def track_step(name, _state, operation):
        """记录图节点顺序并执行节点闭包。"""
        tracked_nodes.append(name)
        return await operation()

    gateway = FakePolishGateway()
    graph = QuillPolishGraph(gateway, "固定Quill Prompt", track_step)
    final = asyncio.run(graph.run(_state()))

    result = final["polish_result"]
    assert result.status == "SUCCESS"
    assert result.stage == "POLISH"
    assert result.action == "POLISH"
    assert result.output_text == "这是润改后的正文。"
    assert final["final_answer"] == "这是润改后的正文。"
    # 纯创作边界：不产生任何 Citation，证据等级固定 INSUFFICIENT。
    assert final["validated_citations"] == []
    assert final["used_citation_indices"] == []
    assert final["evidence_level"] == "INSUFFICIENT"
    assert final["agent_route"] == ["QUILL"]
    assert final["intent"] == "POLISH"
    assert "AI 辅助生成" in result.academic_notice
    assert tracked_nodes == ["TextTransform"]


def test_quill_polish_instruction_carries_context_and_boundary() -> None:
    """续写指令必须携带选区前后文，并约束只输出正文与不虚构事实。"""

    async def run_step(_name, _state, operation):
        """执行节点闭包，不创建测试持久化步骤。"""
        return await operation()

    gateway = FakePolishGateway()
    graph = QuillPolishGraph(gateway, "固定Quill Prompt", run_step)
    asyncio.run(
        graph.run(
            _state(
                action="CONTINUE",
                context_after="后续章节会讨论检索效果评估。",
            )
        )
    )
    instruction = gateway.instructions[0]
    assert "续写" in instruction
    assert "选区前文" in instruction
    assert "选区后文" in instruction
    assert "不得虚构新的研究事实" in instruction
    assert "不要任何前后缀解释" in instruction


def test_quill_polish_covers_all_cnki_creation_actions() -> None:
    """知网划词工具条的七个纯创作操作都有确定的指令映射。"""
    assert set(ACTION_BRIEFS) == {
        "CONTINUE",
        "EXPAND",
        "SHORTEN",
        "REWRITE",
        "POLISH",
        "ZH2EN",
        "EN2ZH",
    }
    for _name, brief in ACTION_BRIEFS.values():
        assert brief.strip()


def test_quill_polish_options_validation() -> None:
    """POLISH options 白名单只接受合法操作编码与受限上下文。"""
    from app.foreign_integration.api import _validate_options

    assert _validate_options(
        "QUILL",
        {"stage": "POLISH", "action": "CONTINUE", "context_before": "前文"},
    ) == {"stage": "POLISH", "action": "CONTINUE", "context_before": "前文"}
    assert _validate_options(
        "QUILL", {"stage": "POLISH", "action": "ZH2EN"}
    ) == {"stage": "POLISH", "action": "ZH2EN"}
    invalid_payloads = (
        {"stage": "POLISH"},
        {"stage": "POLISH", "action": "TRANSLATE"},
        {"stage": "POLISH", "action": "POLISH", "context_before": "x" * 3001},
        {"stage": "POLISH", "action": "POLISH", "year_from": 2020},
    )
    for payload in invalid_payloads:
        try:
            _validate_options("QUILL", payload)
        except HTTPException as exc:
            assert exc.status_code == 422
        else:
            raise AssertionError(f"应拒绝非法Quill润改参数：{payload}")


def test_quill_theory_options_still_accepted() -> None:
    """THEORY 白名单在引入 POLISH 后保持原契约不变。"""
    from app.foreign_integration.api import _validate_options

    assert _validate_options("QUILL", {}) == {
        "stage": "THEORY",
        "citation_style": "GBT7714",
        "literature_limit": 10,
    }
