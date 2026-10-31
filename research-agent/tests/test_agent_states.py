"""验证公共 AgentState 与启明、智鉴、溯知专业 State 的字段边界。"""

import asyncio
import uuid
from collections.abc import Awaitable, Callable
from typing import Any

from app.agents.qiming_graph import QimingGraph
from app.agents.qiming_state import QimingState
from app.agents.schemas import QimingIntentClassification
from app.agents.state import AgentState
from app.agents.suzhi_graph import SuzhiGraph
from app.agents.suzhi_schemas import SuzhiResult
from app.agents.suzhi_state import SuzhiState
from app.agents.zhijian_graph import ZhijianGraph
from app.agents.zhijian_state import ZhijianState


async def _run_operation(
    _name: str,
    _state: AgentState,
    operation: Callable[[], Awaitable[dict[str, Any]]],
) -> dict[str, Any]:
    """执行测试图节点而不创建持久化步骤。

    Args:
        _name: 当前节点名称；边界测试不使用该值。
        _state: 当前公共状态；边界测试不持久化该值。
        operation: 实际节点操作。

    Returns:
        节点产生的状态增量。
    """
    return await operation()


def test_agent_state_contains_only_shared_contract_fields() -> None:
    """公共契约应覆盖统一运行字段，但不得反向包含专业工作流内部字段。"""
    shared_fields = {
        "request_id",
        "user_id",
        "conversation_id",
        "query",
        "knowledge_base_id",
        "knowledge_base_version",
        "agent_route",
        "evidence",
        "citations",
        "validated_citations",
        "final_answer",
    }
    forbidden_specialist_fields = {
        "security_blocked",
        "compare_request",
        "zhijian_result",
        "research_topic",
        "suzhi_result",
    }

    assert shared_fields <= AgentState.__annotations__.keys()
    assert forbidden_specialist_fields.isdisjoint(AgentState.__annotations__)


def test_specialized_states_extend_without_leaking_internal_fields() -> None:
    """三个专业 State 应继承公共字段，并只暴露自身流程所需的扩展字段。"""
    assert "evidence" in QimingState.__annotations__
    assert "security_blocked" in QimingState.__annotations__
    assert "zhijian_result" in QimingState.__annotations__
    assert "suzhi_result" in QimingState.__annotations__
    assert "compare_request" not in QimingState.__annotations__

    assert "evidence" in ZhijianState.__annotations__
    assert "compare_request" in ZhijianState.__annotations__
    assert "research_topic" not in ZhijianState.__annotations__

    assert "evidence" in SuzhiState.__annotations__
    assert "research_topic" in SuzhiState.__annotations__
    assert "compare_request" not in SuzhiState.__annotations__


def test_qiming_graph_exposes_only_professional_result_boundaries() -> None:
    """启明应接收专业最终结果，但不接管智鉴和溯知内部处理 Channel。"""
    dependency = object()
    zhijian = ZhijianGraph(dependency, dependency, "", _run_operation)  # type: ignore[arg-type]
    suzhi = SuzhiGraph(dependency, dependency, "", _run_operation)  # type: ignore[arg-type]
    qiming = QimingGraph(
        dependency,
        dependency,
        "",
        _run_operation,  # type: ignore[arg-type]
        zhijian_graph=zhijian,
        suzhi_graph=suzhi,
    )

    assert "zhijian_result" in qiming.graph.channels
    assert "suzhi_result" in qiming.graph.channels
    assert "compare_request" not in qiming.graph.channels
    assert "research_topic" not in qiming.graph.channels
    assert "compare_request" in zhijian.graph.channels
    assert "research_topic" in suzhi.graph.channels


class FakeDelegatedSuzhiGraph:
    """记录启明传入的溯知规格，并返回已通过发布闸门的最小专业结果。"""

    def __init__(self) -> None:
        """初始化尚未接收专业状态的测试替身。"""
        # 首次委派后保存输入，用于核对启明到溯知的字段适配边界。
        self.received_state: SuzhiState | None = None

    async def run(self, state: SuzhiState) -> SuzhiState:
        """模拟完整溯知图返回可发布结果。

        Args:
            state: 启明适配后的溯知专业状态。

        Returns:
            含公共发布字段与结构化综述结果的最终状态。
        """
        self.received_state = state
        result = SuzhiResult(
            status="SUCCESS",
            research_topic=state["research_topic"],
            concept="人工智能与教育变革",
            language="ZH",
            review_length="MEDIUM",
            citation_style="GBT7714",
            review="已生成溯知综述。",
            evidence_level="MEDIUM",
        )
        return SuzhiState(
            **{
                **state,
                "retrieved_candidates": 2,
                "evidence": [],
                "citations": [],
                "used_citation_indices": [],
                "validated_citations": [],
                "reviewer_status": "PASS",
                "citation_validation_status": "PASS",
                "evidence_summary": {"total": 2},
                "evidence_level": "MEDIUM",
                "final_answer": result.review,
                "suzhi_result": result,
            }
        )


class ReviewIntentGateway:
    """模拟启明结构化模型把自然语言请求识别为文献综述。"""

    async def structured(self, *, schema, **_kwargs):
        """返回符合启明枚举契约的文献综述意图。"""
        assert schema is QimingIntentClassification
        return QimingIntentClassification(intent="LITERATURE_REVIEW")


def test_qiming_routes_review_to_suzhi_with_default_professional_spec() -> None:
    """启明识别综述意图后应进入溯知，并只回传公共字段与最终结果。"""
    tracked_nodes: list[str] = []

    async def track_step(
        name: str,
        _state: QimingState,
        operation: Callable[[], Awaitable[dict[str, Any]]],
    ) -> dict[str, Any]:
        """记录启明 Supervisor 节点并执行节点逻辑。

        Args:
            name: 当前启明节点名。
            _state: 当前启明状态；测试只验证路由，不读取该值。
            operation: 实际节点逻辑。

        Returns:
            节点产生的状态增量。
        """
        tracked_nodes.append(name)
        return await operation()

    specialist = FakeDelegatedSuzhiGraph()
    dependency = object()
    graph = QimingGraph(
        dependency,
        ReviewIntentGateway(),
        "固定启明 Prompt",
        track_step,  # type: ignore[arg-type]
        suzhi_graph=specialist,  # type: ignore[arg-type]
    )
    final = asyncio.run(
        graph.run(
            QimingState(
                request_id="review-route-request",
                user_id=str(uuid.uuid4()),
                conversation_id=str(uuid.uuid4()),
                query="梳理人工智能与教育变革的研究演进",
                output_language="AUTO",
                knowledge_base_id=str(uuid.uuid4()),
                knowledge_base_version=12,
                mode="FAST",
                retry_count=0,
                retrieval_filters={},
                errors=[],
            )
        )
    )

    assert final["intent"] == "LITERATURE_REVIEW"
    assert final["current_agent"] == "SUZHI"
    assert final["agent_route"] == ["QIMING", "SUZHI"]
    assert final["final_answer"] == "已生成溯知综述。"
    assert final["suzhi_result"].status == "SUCCESS"
    assert final["task_plan"][0]["expected_output"] == (
        "EVIDENCE_GROUNDED_LITERATURE_REVIEW"
    )
    assert "UnsupportedIntent" not in tracked_nodes
    assert specialist.received_state is not None
    assert specialist.received_state["research_topic"] == final["query"]
    # 启明不再替专业图决定输出语言，溯知会在 ConceptExtraction 中自行解析。
    assert specialist.received_state["output_language"] == "AUTO"
    assert specialist.received_state["review_length"] == "MEDIUM"
    assert specialist.received_state["citation_style"] == "GBT7714"
    assert specialist.received_state["mode"] == "DEEP"
