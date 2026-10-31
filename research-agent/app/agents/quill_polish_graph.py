"""实现Quill POLISH 阶段（划词润改）的 LangGraph。

续写、扩写、缩写、改写、润色与中英互译属于设计方案的纯创作操作：
不检索知识库、不产生 Citation，只对用户选区做表达层变换。固定单节点
工作流调用冻结模型网关，输出经结构化校验后确定性发布。
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from langgraph.graph import END, START, StateGraph

from app.agents.gateway import ModelGateway
from app.agents.quill_polish_state import QuillPolishState
from app.agents.quill_schemas import (
    PolishAction,
    PolishOutput,
    QuillPolishResult,
)

NodeOperation = Callable[[], Awaitable[dict[str, Any]]]
StepRunner = Callable[[str, QuillPolishState, NodeOperation], Awaitable[dict[str, Any]]]

# 操作编码到人类可读名称与指令的映射；指令明确只改表达、不新增事实。
ACTION_BRIEFS: dict[PolishAction, tuple[str, str]] = {
    "CONTINUE": (
        "续写",
        "从选区文字的结尾自然续写，风格与上下文保持一致，只输出续写的新内容。",
    ),
    "EXPAND": (
        "扩写",
        "把选区文字扩写得更充分具体，保留原意与全部事实，不得添加虚构细节。",
    ),
    "SHORTEN": (
        "缩写",
        "把选区文字压缩为更精炼的表述，保留核心含义，不得引入新事实。",
    ),
    "REWRITE": (
        "改写",
        "用不同表达重写选区文字，原意、事实与论断强度必须保持不变。",
    ),
    "POLISH": (
        "润色",
        "润色选区文字的措辞、语序与学术文体，只改表达，不改事实与含义。",
    ),
    "ZH2EN": (
        "中译英",
        "把选区中文翻译为学术英语，术语准确、语句符合学术论文规范。",
    ),
    "EN2ZH": (
        "英译中",
        "把选区英文翻译为学术中文，术语准确、语句通顺符合中文学术表达习惯。",
    ),
}


class QuillPolishGraph:
    """执行划词选区输入到润改结果的固定单节点工作流。"""

    def __init__(
        self,
        gateway: ModelGateway,
        system_prompt: str,
        step_runner: StepRunner,
    ) -> None:
        """注入固定模型、版本化 Prompt 和步骤记录器。

        Args:
            gateway: 只接收受控输入的结构化模型网关。
            system_prompt: Run 创建时冻结的Quill Prompt 正文。
            step_runner: 在节点前后持久化 AgentStep 的运行回调。
        """
        # 模型网关只负责受控的表达层变换，不接触数据源，也不产生引用。
        self.gateway = gateway
        # Prompt 从 Run 固定版本读取，运行期间不得切换到当前激活版本。
        self.system_prompt = system_prompt
        # 步骤回调负责节点可观察状态和脱敏失败信息。
        self.step_runner = step_runner
        # 图在构造期编译一次，单次运行的可变数据全部通过状态传递。
        self.graph = self._build()

    def _build(self):
        """编译固定单节点的划词润改图。

        Returns:
            可异步调用并以稳定节点名记录步骤的已编译 LangGraph。
        """
        builder = StateGraph(QuillPolishState)
        # 唯一节点：按操作编码对选区文本做表达层变换，不检索、不引用。
        builder.add_node("TextTransform", self._text_transform)
        builder.add_edge(START, "TextTransform")
        builder.add_edge("TextTransform", END)
        return builder.compile()

    async def run(self, state: QuillPolishState) -> QuillPolishState:
        """执行划词润改图并返回包含结构化结果的最终状态。

        Args:
            state: 已冻结用户、选区文本和操作编码的初始状态。

        Returns:
            包含 QuillPolishResult 的最终状态。
        """
        return await self.graph.ainvoke(state)

    async def _tracked(
        self, name: str, state: QuillPolishState, operation: NodeOperation
    ) -> dict[str, Any]:
        """通过 Service 回调持久化节点检查点。

        Args:
            name: 与前端阶段和 AgentStep 审计记录一致的稳定节点名。
            state: 当前节点收到的润改状态。
            operation: 不直接接触数据库的节点业务操作。

        Returns:
            当前节点产生并合并回 LangGraph 的状态增量。

        Raises:
            Exception: 节点操作失败；回调记录脱敏失败检查点后继续向上抛出。
        """
        return await self.step_runner(name, state, operation)

    def _instruction(self, state: QuillPolishState) -> str:
        """构造只依赖选区与上下文的纯创作指令。

        Args:
            state: 包含选区文本、上下文和操作编码的润改状态。

        Returns:
            限定输出契约与学术诚信边界的模型指令。
        """
        _name, brief = ACTION_BRIEFS[state["action"]]
        parts = [
            f"请对下面的选区文字执行「{brief}」",
            "这是学术写作辅助：只做表达层变换，不得虚构新的研究事实或数据。",
            "只输出变换后的正文本身，不要任何前后缀解释、引号或 [n] 引用标记。",
        ]
        before = state.get("context_before") or ""
        after = state.get("context_after") or ""
        if before:
            parts.append(f"\n选区前文（仅供理解语境，不要改写）：{before}")
        parts.append(f"\n选区文字：{state['selected_text']}")
        if after:
            parts.append(f"\n选区后文（仅供理解语境，不要改写）：{after}")
        return "\n".join(parts)

    async def _text_transform(self, state: QuillPolishState) -> dict[str, Any]:
        """调用冻结模型完成选区的表达层变换。

        Args:
            state: 包含选区文本和操作编码的润改状态。

        Returns:
            结构化润改结果与发布回答组成的状态增量。
        """

        async def operation() -> dict[str, Any]:
            """从受控输入生成润改结果。

            Returns:
                等待确定性发布的结构化润改结果。
            """
            output = await self.gateway.structured(
                instruction=self._instruction(state),
                schema=PolishOutput,
                system_prompt=self.system_prompt,
                thinking=bool(state.get("thinking")),
            )
            result = QuillPolishResult(
                status="SUCCESS",
                stage="POLISH",
                action=state["action"],
                output_text=output.output_text.strip(),
            )
            return {
                "intent": "POLISH",
                "agent_route": self._route(state),
                "polish_result": result,
                "final_answer": result.output_text,
                "evidence_level": result.evidence_level,
                "validated_citations": [],
                "used_citation_indices": [],
            }

        return await self._tracked("TextTransform", state, operation)

    def _route(self, state: QuillPolishState) -> list[str]:
        """在既有路由链末尾追加 QUILL，避免重复节点。

        Args:
            state: 包含运行路由链的润改状态。

        Returns:
            追加 QUILL 后的可审计路由链。
        """
        route = list(state.get("agent_route", []))
        if not route or route[-1] != "QUILL":
            route.append("QUILL")
        return route
