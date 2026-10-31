"""实现智鉴 Evidence First 专业比较 LangGraph。

图显式控制自然语言解析、确定性对象规范化、对象×维度检索、逐 Chunk 单元格归属、
证据平衡、比较分析和引用校验。模型不决定跳过节点，也不能访问数据库。
"""

from __future__ import annotations

import asyncio
import json
import re
import uuid
from collections.abc import Awaitable, Callable
from typing import Any

from langgraph.graph import END, START, StateGraph

from app.agents.gateway import ModelGateway, StructuredSchemaValidationError
from app.agents.tools import RetrievalTool
from app.agents.zhijian_normalization import ambiguity_question, normalize_compare_object
from app.agents.zhijian_schemas import (
    MAX_COMPARE_DIMENSIONS,
    CompareDimensionPlan,
    CompareQuery,
    CompareRetrievalContext,
    ComparisonMatrixDraftRow,
    ComparisonMatrixRow,
    DimensionEvidenceCoverage,
    EvidenceAssignmentBatch,
    EvidenceAssignmentDecision,
    EvidenceBalance,
    ParsedCompareRequest,
    ReferenceDocumentItem,
    ZhijianAnalysisDraft,
    ZhijianCoreAnalysisDraft,
    ZhijianEvidenceRead,
    ZhijianResult,
    normalize_dimension_title,
)
from app.agents.zhijian_state import ZhijianState
from app.common.config import get_settings
from app.retrieval.languages import normalize_source_language
from app.retrieval.query_processing import BilingualQueryProcessor
from app.retrieval.schemas import CitationRead, EvidenceRead, RetrievalFilters

# EvidenceAssignment 分批限制单次 JSON 的数组长度，降低兼容模型遗漏字段或截断 UUID 的概率。
EVIDENCE_ASSIGNMENT_BATCH_SIZE = 4
# 最多并行两个归属批次，缩短等待时间同时避免对兼容模型服务形成突发压力。
EVIDENCE_ASSIGNMENT_MAX_CONCURRENCY = 2
# 智鉴每条对象×维度查询的返回预算；FAST 原始召回规模与其他 Agent 默认值不变。
MAX_QUERY_EVIDENCE = 10
# 去重后提交给 EvidenceAssignment 的全局候选上限；先保证每条查询至少保留首个候选。
MAX_MERGED_CANDIDATES = 40
# 归属过滤后供比较分析与持久化使用的最终证据上限，独立于候选预算。
MAX_FINAL_EVIDENCE = 30
# 核心分析最多生成两轮；模型网关内部的传输和 Schema 重试独立计数。
MAX_COMPARE_ANALYSIS_ATTEMPTS = 2
# 归属模型遗漏或虚构候选时最多重做一次，同一轮仍使用固定合并候选。
MAX_EVIDENCE_ASSIGNMENT_ATTEMPTS = 2
# 单批连续未通过本地 Schema 时，仅重送该批一次；网关内部仍各有一次格式纠正。
MAX_EVIDENCE_ASSIGNMENT_BATCH_ATTEMPTS = 2

NodeOperation = Callable[[], Awaitable[dict[str, Any]]]
StepRunner = Callable[[str, ZhijianState, NodeOperation], Awaitable[dict[str, Any]]]


class _AssignmentCoverageError(ValueError):
    """标识归属模型输出与固定候选集合不符，可在图中有界纠正。"""


def _requested_dimensions(state: ZhijianState) -> list[str]:
    """规范化用户明确给出的比较维度并保持原始顺序。

    Args:
        state: 包含自然语言解析结果的智鉴状态。

    Returns:
        去除空值、非法短标题和重复项后的维度列表。
    """
    requested = state["compare_request"].requested_dimensions
    return list(
        dict.fromkeys(
            normalized
            for item in requested
            if (normalized := normalize_dimension_title(item)) and 4 <= len(normalized) <= 30
        )
    )


def _select_final_candidates(
    candidates: list[dict[str, object]],
    decisions: dict[uuid.UUID, EvidenceAssignmentDecision],
    dimensions: list[str],
) -> list[dict[str, object]]:
    """过滤无关候选，并在最终预算内兼顾专属证据门槛和实际单元格覆盖。

    Args:
        candidates: 已按真实 Chunk 去重且保留召回轨迹的固定候选。
        decisions: 已完整校验候选与维度边界的模型归属判断。
        dimensions: 本次实际采用的比较维度原文。

    Returns:
        最多三十条候选，保留输入顺序以稳定后续 Evidence 与 Citation 编号。
        有效候选不足预算时全部保留，不以无关证据补齐。
    """
    eligible = [
        item for item in candidates
        if decisions[item["evidence"].chunk_id].relevance != "IRRELEVANT"
        and decisions[item["evidence"].chunk_id].supports
    ]
    if len(eligible) <= MAX_FINAL_EVIDENCE:
        return eligible

    # 1. 总体语义相关度优先，同等级按检索重排分稳定排序；召回分支不决定归属。
    relevance_order = {"HIGH": 2, "MEDIUM": 1, "LOW": 0}
    ranked = sorted(eligible, key=lambda item: (
        relevance_order[decisions[item["evidence"].chunk_id].relevance],
        item["evidence"].relevance_score,
    ), reverse=True)
    cells = [(side, dimension) for dimension in dimensions for side in ("A", "B")]
    coverage = dict.fromkeys(cells, 0)
    selected: set[uuid.UUID] = set()
    exclusive_counts = {"A": 0, "B": 0}

    def keep(item: dict[str, object]) -> None:
        """加入一个未选 Chunk，同时累计真实单元格覆盖与两侧专属证据数。"""
        chunk_id = item["evidence"].chunk_id
        if chunk_id in selected or len(selected) >= MAX_FINAL_EVIDENCE:
            return
        selected.add(chunk_id)
        supports = decisions[chunk_id].supports
        sides = {support.object for support in supports}
        if len(sides) == 1:
            exclusive_counts[next(iter(sides))] += 1
        for support in supports:
            coverage[(support.object, support.dimension)] += 1

    # 2. 先保留能满足现有平衡闸门的专属文献和证据，SHARED 不替代任一侧专属证据。
    settings = get_settings()
    for side in ("A", "B"):
        side_items = [item for item in ranked if {
            support.object for support in decisions[item["evidence"].chunk_id].supports
        } == {side}]
        documents: set[uuid.UUID] = set()
        for item in side_items:
            if (len(documents) >= settings.zhijian_min_side_documents
                    or len(selected) >= MAX_FINAL_EVIDENCE):
                break
            document_id = item["evidence"].document_id
            if document_id not in documents:
                keep(item)
                documents.add(document_id)
        for item in side_items:
            if (exclusive_counts[side] >= settings.zhijian_min_side_evidence
                    or len(selected) >= MAX_FINAL_EVIDENCE):
                break
            keep(item)

    # 3. 轮流补足当前覆盖较少的真实对象×维度单元格，每次选择该单元格最高相关候选。
    # 已耗尽的单元格直接跳过，不凭召回轨迹补造支持关系。
    while len(selected) < MAX_FINAL_EVIDENCE:
        remaining = [item for item in ranked if item["evidence"].chunk_id not in selected]
        if not remaining:
            break
        for cell in sorted(cells, key=coverage.__getitem__):
            winner = next((item for item in remaining if any(
                (support.object, support.dimension) == cell
                for support in decisions[item["evidence"].chunk_id].supports
            )), None)
            if winner is not None:
                keep(winner)
                break
        else:
            break
    return [item for item in eligible if item["evidence"].chunk_id in selected]


class ZhijianGraph:
    """执行一句话输入到可追溯专业比较结果的固定工作流。"""

    def __init__(
        self,
        retrieval: RetrievalTool,
        gateway: ModelGateway,
        system_prompt: str,
        step_runner: StepRunner,
    ) -> None:
        """注入受控检索、模型网关、冻结 Prompt 与步骤记录器。

        Args:
            retrieval: 已绑定当前用户的 M4 检索 Tool。
            gateway: 支持结构化调用的 Run 固定模型网关。
            system_prompt: 智鉴固定 Prompt 版本正文。
            step_runner: 每个节点前后持久化 AgentStep 的回调。
        """
        self.retrieval = retrieval
        self.gateway = gateway
        self.system_prompt = system_prompt
        self.step_runner = step_runner
        # 编译后的比对状态图，运行数据独立传入，不能把上一轮证据当作本轮输入。
        self.graph = self._build()

    def _build(self):
        """编译带双路检索汇合与核心分析有界回退的智鉴图。

        Returns:
            可异步调用的 LangGraph。
        """
        # 1. 注册从输入理解到引用校验的全部业务节点
        # 节点名称同时是 SSE 与审计契约；模型只能完成节点内部结构化任务，
        # 不得改变图拓扑或绕过双路检索、EvidenceAssignment 和引用校验。
        builder = StateGraph(ZhijianState)
        # 解析自然语言请求，提取比较对象、主题、时间、来源语言及用户指定维度。
        builder.add_node("ParseCompareRequest", self._parse_request)
        # 同一节点先规范化对象，再检查歧义与请求范围，形成唯一的检索前决策。
        builder.add_node("ValidateCompareRequest", self._validate_request)
        # 优先采用用户指定的至多三个有效维度，否则由模型恰好规划三个维度。
        builder.add_node("GenerateCompareDimensions", self._generate_dimensions)
        # 将对象、别名、主题、单个维度和时间组合为对象×维度检索查询。
        builder.add_node("BuildCompareQueries", self._build_queries)
        # 执行 A 侧混合检索，取得候选证据；召回分支不等于证据最终归属。
        builder.add_node("RetrieveObjectA", self._retrieve_a)
        # 执行 B 侧混合检索，与 A 侧并行，并将候选证据写入独立状态字段。
        builder.add_node("RetrieveObjectB", self._retrieve_b)
        # 按真实 chunk_id 去重，保留全部对象×维度召回轨迹及最高检索分数。
        builder.add_node("MergeCandidates", self._merge_candidates)
        # 由模型判断候选支持哪些对象×维度单元格，并派生 A/B/SHARED 展示归属。
        builder.add_node("EvidenceAssignment", self._assign_evidence)
        # 统计双侧证据覆盖并检查充分性门槛，证据不足时阻止后续比较分析。
        builder.add_node("EvidenceBalanceCheck", self._check_balance)
        # 基于已确认的证据和动态维度生成结构化比较草稿，供后续节点校验与整理。
        builder.add_node("CrossAnalysis", self._cross_analysis)
        # 一次核对比较草稿的共同点与差异点引用，并记录两类条目数量。
        builder.add_node("CompareAnalysisValidation", self._compare_analysis_validation)
        # 校验时间节点的对象归属与证据年份，剔除无效、重复节点并按年份排序。
        builder.add_node("TimelineBuilder", self._timeline_builder)
        # 按分侧引用过滤研究热点，拒绝错侧或缺少真实 Citation 的主题。
        builder.add_node("TopicAggregator", self._topic_aggregator)
        # 补齐矩阵维度，校验分侧引用及共同点引用集合的双侧覆盖。
        builder.add_node("MatrixGenerator", self._matrix_generator)
        # 校验专业结论引用与真实 Document、Chunk、页码的对应关系，构造最终智鉴结果。
        builder.add_node("CitationValidator", self._citation_validator)
        # 为需澄清或证据不足的请求生成明确的受限结果。
        builder.add_node("FinalizeRestricted", self._finalize_restricted)

        # 2. 解析后统一规范化和验证；只有合法请求才进入检索
        builder.add_edge(START, "ParseCompareRequest")
        builder.add_edge("ParseCompareRequest", "ValidateCompareRequest")
        builder.add_conditional_edges(
            "ValidateCompareRequest",
            lambda state: "continue" if not state.get("compare_block_status") else "restricted",
            {"continue": "GenerateCompareDimensions", "restricted": "FinalizeRestricted"},
        )
        builder.add_edge("GenerateCompareDimensions", "BuildCompareQueries")

        # 3. A/B 分支各自执行单维度查询，在两个结果都完成后汇合去重
        # 两个节点只写各自 namespaced 字段，避免并行状态更新发生覆盖冲突。
        builder.add_edge("BuildCompareQueries", "RetrieveObjectA")
        builder.add_edge("BuildCompareQueries", "RetrieveObjectB")
        builder.add_edge(["RetrieveObjectA", "RetrieveObjectB"], "MergeCandidates")
        builder.add_edge("MergeCandidates", "EvidenceAssignment")
        builder.add_conditional_edges(
            "EvidenceAssignment",
            self._route_evidence_assignment,
            {"PASS": "EvidenceBalanceCheck", "RETRY": "EvidenceAssignment"},
        )
        builder.add_conditional_edges(
            "EvidenceBalanceCheck",
            lambda state: (
                "continue" if state["evidence_balance"].status == "SUFFICIENT" else "restricted"
            ),
            {"continue": "CrossAnalysis", "restricted": "FinalizeRestricted"},
        )

        # 4. 核心结论首轮引用校验失败只重做一次 CrossAnalysis，固定 Evidence 不变。
        # 第二轮仍失败由校验节点终止，最终来源映射仍由 CitationValidator 把关。
        builder.add_edge("CrossAnalysis", "CompareAnalysisValidation")
        builder.add_conditional_edges(
            "CompareAnalysisValidation",
            self._route_compare_analysis_validation,
            {"PASS": "TimelineBuilder", "RETRY": "CrossAnalysis"},
        )
        builder.add_edge("TimelineBuilder", "TopicAggregator")
        builder.add_edge("TopicAggregator", "MatrixGenerator")
        builder.add_edge("MatrixGenerator", "CitationValidator")
        builder.add_edge("CitationValidator", END)
        builder.add_edge("FinalizeRestricted", END)
        return builder.compile()

    def _route_evidence_assignment(self, state: ZhijianState) -> str:
        """按归属输出的候选覆盖结果继续或重做一次归属。

        Args:
            state: 包含归属校验状态的智鉴图状态。

        Returns:
            PASS 或 RETRY 路由标识。
        """
        return state["evidence_assignment_status"]

    def _route_compare_analysis_validation(self, state: ZhijianState) -> str:
        """按本轮核心分析引用校验结果继续整理或重新生成。

        Args:
            state: 包含校验状态的智鉴图状态。

        Returns:
            PASS 或 RETRY 路由标识。
        """
        return state["compare_analysis_validation_status"]

    async def run(self, state: ZhijianState) -> ZhijianState:
        """执行智鉴图并返回完整状态。

        Args:
            state: 固定身份、知识库版本和自然语言请求的初始状态。

        Returns:
            包含结构化智鉴结果、真实 Evidence 和校验状态的最终状态。
        """
        return await self.graph.ainvoke(state)

    async def _tracked(
        self, name: str, state: ZhijianState, operation: NodeOperation
    ) -> dict[str, Any]:
        """通过 Service 回调记录节点检查点。

        Args:
            name: 稳定节点名。
            state: 当前图状态。
            operation: 节点业务逻辑。

        Returns:
            节点状态增量。
        """
        return await self.step_runner(name, state, operation)

    async def _parse_request(self, state: ZhijianState) -> dict[str, Any]:
        """使用 ModelGateway.structured 解析一句话比较请求。

        Args:
            state: 当前比较 Run 的图状态，包含该节点所需输入。

        Returns:
            解析后的比较对象、主题及用户明确限制。
        """

        async def operation() -> dict[str, Any]:
            """生成只负责理解请求的结构化字段。

            Returns:
                解析后的比较请求字段。
            """
            parsed     = await self.gateway.structured(
                instruction=(
                    "解析以下自然语言比较请求。不要检索、分析或补造用户未表达的限制。"
                    "object_a/object_b 是用户希望直接对照的两项完整内容，不是固定类型的实体抽取。"
                    "对象可以是具体事物、概念、观点、做法、集合、阶段、案例、方案或其他内容；"
                    "这些只是示例，不构成对象类型白名单。按用户表达保持对象的含义和粒度，"
                    "保留识别它所必需的所属、名称、场景、时期、版本等限定；"
                    "不要将具体对象缩成其所属国家、机构或人物，也不要给抽象对象擅加具体限定。"
                    "两个对象分别识别，不把一侧的专属名称或限定移到另一侧，不强行改成同一类型。"
                    "共同讨论的问题放入 topic，独立要求的比较维度和时间范围单独提取；"
                    "不能为了抽取共同主题而丢掉对象身份的一部分。"
                    "例如“比较中国和日本的人工智能教育政策”是在共同主题下比较两国，"
                    "应为 object_a=中国、object_b=日本、topic=人工智能教育政策；"
                    "“比较中国双减政策和韩国教育减负政策”直接对照两项具体内容，"
                    "应为 object_a=中国双减政策、object_b=韩国教育减负政策、topic=教育减负政策；"
                    "“比较项目式学习和探究式学习的课堂应用”应保留两个完整做法名称，"
                    "不能都缩成学习；“比较改革前和改革后的评价机制”应保留两侧的阶段限定。"
                    "无法从请求中确定两项对象时，相应字段留空；缺少上下文的指代不得猜测。"
                    "对象名称中的时期或版本是身份限定，不自动转成两侧共用的检索时间范围。"
                    "没有明确时间范围时 year_from/year_to 为 null；没有明确要求来源语言时"
                    " source_language_requirement 为 null 且 source_language_codes 为空；"
                    "明确要求来源语言时，将所有指定语种写入 source_language_codes，使用语言代码，"
                    "例如中文、英文、日文写为 ['ZH','EN','JA']，不限制可用语种。"
                    "来源语言与最终回答语言分开解析；未明确维度时 requested_dimensions 为空。"
                    "根据用户问题的主要语言或明确语言要求设置 output_language；只有用户明确"
                    "要求同时提供中文和英文时才设置为 BILINGUAL。"
                    f"\n请求：{state['query']}"
                ),
                schema=ParsedCompareRequest,
                system_prompt=self.system_prompt,
            )
            route = list(state.get("agent_route", []))
            if not route or route[-1] != "ZHIJIAN":
                route.append("ZHIJIAN")
            return {
                "compare_request": parsed,
                "original_language": BilingualQueryProcessor()
                .process(state["query"])
                .language,
                "output_language": parsed.output_language,
                "intent": "COMPARE",
                "current_agent": "ZHIJIAN",
                "agent_route": route,
            }

        return await self._tracked("ParseCompareRequest", state, operation)

    async def _validate_request(self, state: ZhijianState) -> dict[str, Any]:
        """规范化 A/B 对象并验证歧义、时间和维度，不按来源语种拒绝请求。

        Args:
            state: 包含自然语言解析结果的智鉴状态。

        Returns:
            规范对象与可继续标记，或带澄清问题的受限状态。
        """

        async def operation() -> dict[str, Any]:
            """在同一检查点内固定对象标识并决定是否允许检索。

            Returns:
                始终包含规范对象和请求校验结果的状态增量。
            """
            request = state["compare_request"]
            # 规范对象先于同一对象检查；受限请求仍保留对象供结果页解释。
            normalized = {
                "compare_object_a": normalize_compare_object(request.object_a),
                "compare_object_b": normalize_compare_object(request.object_b),
            }
            if not request.object_a or not request.object_b:
                return {
                    **normalized,
                    "compare_block_status": "NEEDS_CLARIFICATION",
                    "clarification_question": "你希望比较哪两个对象？例如中国和美国。",
                }
            for raw in (request.object_a, request.object_b):
                question = ambiguity_question(raw)
                if question:
                    return {
                        **normalized,
                        "compare_block_status": "NEEDS_CLARIFICATION",
                        "clarification_question": question,
                    }
            if (
                normalized["compare_object_a"].identifier
                == normalized["compare_object_b"].identifier
            ):
                return {
                    **normalized,
                    "compare_block_status": "NEEDS_CLARIFICATION",
                    "clarification_question": "两个比较对象相同，请提供两个不同的对象。",
                }
            requested_dimensions = _requested_dimensions(state)
            if len(requested_dimensions) > MAX_COMPARE_DIMENSIONS:
                return {
                    **normalized,
                    "compare_block_status": "NEEDS_CLARIFICATION",
                    "clarification_question": (
                        f"一次最多支持 {MAX_COMPARE_DIMENSIONS} 个比较维度，"
                        "请保留最重要的维度后重新提问。"
                    ),
                }
            if request.year_from and request.year_to and request.year_from > request.year_to:
                return {
                    **normalized,
                    "compare_block_status": "NEEDS_CLARIFICATION",
                    "clarification_question": (
                        "时间范围的起始年份晚于结束年份，请重新说明时间范围。"
                    ),
                }
            return {**normalized, "compare_block_status": "", "clarification_question": ""}

        return await self._tracked("ValidateCompareRequest", state, operation)

    async def _generate_dimensions(self, state: ZhijianState) -> dict[str, Any]:
        """优先采用用户至多三个明确维度，否则由模型恰好规划三个。

        Args:
            state: 当前比较 Run 的图状态，包含该节点所需输入。

        Returns:
            本次比较采用的维度列表。
        """

        async def operation() -> dict[str, Any]:
            """生成适配当前任务的有限维度列表。

            Returns:
                用户指定或模型规划的比较维度。
            """
            # 用户明确给出的维度允许少于三个；全部无效时回到模型规划，避免把脏标题
            # 继续传入对象×维度检索与矩阵生成。超过上限的请求已在验证节点自然语言澄清。
            dimensions = _requested_dimensions(state)
            if not dimensions:
                plan = await self.gateway.structured(
                    instruction=(
                        f"为以下比较任务恰好规划 {MAX_COMPARE_DIMENSIONS} 个具体、"
                        "互不重复且可由文献证据支撑的维度。"
                        "每个维度必须是 4-30 个字的短标题，不要写解释或完整句子，"
                        "也不要机械套用固定模板；禁止返回逗号、句号等纯标点维度。"
                        f"\n用户原始请求：{state.get('query', '')}"
                        f"\n主题：{state['compare_request'].topic}"
                        f"\n对象A：{state['compare_object_a'].model_dump_json()}"
                        f"\n对象B：{state['compare_object_b'].model_dump_json()}"
                    ),
                    schema=CompareDimensionPlan,
                    system_prompt=self.system_prompt,
                )
                dimensions = plan.dimensions
            return {"compare_dimensions": dimensions[:MAX_COMPARE_DIMENSIONS]}

        return await self._tracked("GenerateCompareDimensions", state, operation)

    async def _build_queries(self, state: ZhijianState) -> dict[str, Any]:
        """为每个对象×维度确定性构造包含主题、别名和时间的查询。

        Args:
            state: 当前比较 Run 的图状态，包含该节点所需输入。

        Returns:
            对象与维度对应的检索查询计划。
        """

        async def operation() -> dict[str, Any]:
            """返回每项只绑定一个对象和一个维度的检索计划。

            Returns:
                逐对象、逐维度的检索查询。
            """
            request = state["compare_request"]
            year_text = ""
            if request.year_from or request.year_to:
                year_text = f" {request.year_from or ''}-{request.year_to or ''}"
            queries: list[CompareQuery] = []
            for key, obj in (
                ("A", state["compare_object_a"]),
                ("B", state["compare_object_b"]),
            ):
                aliases = " ".join(obj.aliases[:5])
                for index, dimension in enumerate(state["compare_dimensions"], start=1):
                    text = " ".join(
                        part
                        for part in (
                            obj.canonical_name,
                            aliases,
                            request.topic,
                            dimension,
                            year_text,
                        )
                        if part
                    ).strip()
                    queries.append(
                        CompareQuery(
                            query_id=f"{key}:{index}",
                            object=key,
                            dimension=dimension,
                            text=text,
                        )
                    )
            return {"compare_queries": queries}

        return await self._tracked("BuildCompareQueries", state, operation)

    def _filters(self, state: ZhijianState) -> RetrievalFilters:
        """把时间及任意来源语种转成检索过滤器，兼容旧的单语种请求字段。

        Args:
            state: 当前比较 Run 的图状态，包含该节点所需输入。

        Returns:
            传递给受控检索服务的时间和语言过滤条件。
        """
        request = state["compare_request"]
        required = request.source_language_codes
        if not required:
            legacy = (request.source_language_requirement or request.language_filter).strip()
            required = [legacy] if legacy.casefold() != "all" else []
        language_codes = list(dict.fromkeys(
            code for value in required if (code := normalize_source_language(value))
        ))
        return RetrievalFilters(
            year_from=request.year_from,
            year_to=request.year_to,
            language_codes=language_codes,
        )

    async def _retrieve_branch(self, state: ZhijianState, branch: str) -> list[dict[str, object]]:
        """依次执行一侧各维度且仍受知识库可见性与 READY 白名单约束的 Hybrid Retrieval。

        Args:
            state: 包含冻结知识库版本和对象×维度查询计划的图状态。
            branch: A 或 B 对象侧。

        Returns:
            当前对象侧按维度排列的检索结果及对应查询上下文。

        Raises:
            RuntimeError: 任一查询发现知识库版本已在 Run 创建后变化。
        """
        results: list[dict[str, object]] = []
        for query in state["compare_queries"]:
            if query.object != branch:
                continue
            # RetrievalService 持有当前后台任务的同步 SQLAlchemy Session；同侧查询按序
            # 执行以避免跨线程共享 Session，A/B 仍由 LangGraph 的独立分支汇合。
            result = await self.retrieval.search(
                knowledge_base_id=uuid.UUID(state["knowledge_base_id"]),
                query=query.text,
                mode="FAST",
                filters=self._filters(state),
                evidence_limit=MAX_QUERY_EVIDENCE,
            )
            if result.knowledge_base_version != state["knowledge_base_version"]:
                raise RuntimeError("知识库版本在运行创建后发生变化，请重新提问")
            results.append(
                {
                    "query": query,
                    "evidence": result.evidence,
                    "citations": result.citations,
                    "candidate_count": result.candidate_count,
                    "expanded_queries": result.expanded_queries,
                }
            )
        return results

    async def _retrieve_a(self, state: ZhijianState) -> dict[str, Any]:
        """执行对象 A 的 Hybrid Retrieval。

        Args:
            state: 当前比较 Run 的图状态，包含该节点所需输入。

        Returns:
            对象 A 的检索结果及查询上下文。
        """

        async def operation() -> dict[str, Any]:
            """返回只属于 A 检索分支的候选集合。

            Returns:
                对象 A 的检索响应。
            """
            return {"object_a_retrieval": await self._retrieve_branch(state, "A")}

        return await self._tracked("RetrieveObjectA", state, operation)

    async def _retrieve_b(self, state: ZhijianState) -> dict[str, Any]:
        """执行对象 B 的 Hybrid Retrieval。

        Args:
            state: 当前比较 Run 的图状态，包含该节点所需输入。

        Returns:
            对象 B 的检索结果及查询上下文。
        """

        async def operation() -> dict[str, Any]:
            """返回只属于 B 检索分支的候选集合。

            Returns:
                对象 B 的检索响应。
            """
            return {"object_b_retrieval": await self._retrieve_branch(state, "B")}

        return await self._tracked("RetrieveObjectB", state, operation)

    async def _merge_candidates(self, state: ZhijianState) -> dict[str, Any]:
        """按真实 chunk_id 去重并保留全部对象×维度召回轨迹和最高分。

        Args:
            state: 当前比较 Run 的图状态，包含该节点所需输入。

        Returns:
            去重并限制数量后的候选及检索统计。
        """

        async def operation() -> dict[str, Any]:
            """归一化各查询 Candidate Chunk，但不把召回来源当作语义归属。

            Returns:
                合并后的候选、召回数和扩展查询。
            """
            # 1. 汇总 A/B 查询结果，以 Chunk 去重并保留每次命中的维度、对象和分数。
            # 召回分支只记录来源，不决定证据最终支持哪一侧；归属留给后续模型判断。
            merged: dict[uuid.UUID, dict[str, object]] = {}
            total_candidates = 0
            expanded: dict[str, str] = {}
            for branch, results in (
                ("A", state["object_a_retrieval"]),
                ("B", state["object_b_retrieval"]),
            ):
                for result in results:
                    query = result["query"]
                    total_candidates += int(result["candidate_count"])
                    for language, text in dict(result["expanded_queries"]).items():
                        expanded[f"{query.query_id}_{language}"] = text
                    citations = {item.chunk_id: item for item in result["citations"]}
                    for item in result["evidence"]:
                        context = CompareRetrievalContext(
                            query_id=query.query_id,
                            object=branch,
                            dimension=query.dimension,
                            relevance_score=item.relevance_score,
                        )
                        current = merged.get(item.chunk_id)
                        if current:
                            current["retrieval_contexts"].append(context)
                            if item.relevance_score > current["evidence"].relevance_score:
                                current["evidence"] = item
                                current["citation"] = citations[item.chunk_id]
                        else:
                            merged[item.chunk_id] = {
                                "evidence": item,
                                "citation": citations[item.chunk_id],
                                "retrieval_contexts": [context],
                            }
            # 2. 同一 Chunk 采用最高相关性结果作为代表，其他命中轨迹仍保留用于审计。
            ordered = sorted(
                merged.values(),
                key=lambda item: item["evidence"].relevance_score,
                reverse=True,
            )
            # 3. 先为每条对象×维度查询保留一个最高分候选，再按全局最高分补足上限，
            # 防止高频维度占满上下文而让低频维度在归属前就完全消失。
            candidates: list[dict[str, object]] = []
            selected_chunks: set[uuid.UUID] = set()
            for query in state["compare_queries"]:
                representative = next(
                    (
                        item
                        for item in ordered
                        if item["evidence"].chunk_id not in selected_chunks
                        and any(
                            context.query_id == query.query_id
                            for context in item["retrieval_contexts"]
                        )
                    ),
                    None,
                )
                if representative is not None:
                    candidates.append(representative)
                    selected_chunks.add(representative["evidence"].chunk_id)
            for item in ordered:
                if len(candidates) >= MAX_MERGED_CANDIDATES:
                    break
                if item["evidence"].chunk_id not in selected_chunks:
                    candidates.append(item)
                    selected_chunks.add(item["evidence"].chunk_id)
            return {
                "merged_candidates": candidates,
                "retrieved_candidates": total_candidates,
                "expanded_queries": expanded,
            }

        return await self._tracked("MergeCandidates", state, operation)

    async def _assign_evidence(self, state: ZhijianState) -> dict[str, Any]:
        """对固定候选生成逐 Chunk 归属，对边界和格式错误分别有界纠正。

        Args:
            state: 当前比较 Run 的图状态，包含该节点所需输入。

        Returns:
            已判定归属的证据、引用和单元格支持关系。
        """

        async def operation() -> dict[str, Any]:
            """分批归属固定候选，并在完整通过结构及边界校验后发布 Evidence。

            Returns:
                已确认归属与 Evidence，或可重入的规则失败状态。

            Raises:
                ValueError: 第二轮仍违反候选或维度边界。
                StructuredSchemaValidationError: 某批次两轮请求仍未通过本地 Schema。
            """
            attempt = state.get("evidence_assignment_attempts", 0) + 1
            if attempt > MAX_EVIDENCE_ASSIGNMENT_ATTEMPTS:
                raise ValueError("智鉴证据归属生成次数超过上限")

            def retry_or_fail(reason: str) -> dict[str, Any]:
                """只把首轮确定性边界错误转为图级重试。

                Args:
                    reason: 不含模型原文的稳定失败原因。

                Returns:
                    RETRY 路由及下一轮纠错原因。

                Raises:
                    ValueError: 第二轮仍违反同一归属边界。
                """
                if attempt >= MAX_EVIDENCE_ASSIGNMENT_ATTEMPTS:
                    raise ValueError(reason)
                return {
                    "evidence_assignment_attempts": attempt,
                    "evidence_assignment_status": "RETRY",
                    "evidence_assignment_error": reason,
                }

            # 1. 空候选直接形成可审计的证据不足输入，不调用外部模型。
            candidates = state["merged_candidates"]
            if not candidates:
                return {
                    "evidence": [],
                    "citations": [],
                    "evidence_assignments": [],
                    "evidence_assignment_attempts": attempt,
                    "evidence_assignment_status": "PASS",
                    "evidence_assignment_error": "",
                }
            # 2. 小批量调用模型，避免十余条长 Chunk 和 UUID 在一个 JSON 数组中被遗漏或截断。
            # 标题、页码和召回轨迹只帮助判断来源语境；最终引用仍绑定原始 Chunk 正文。
            decisions: dict[uuid.UUID, EvidenceAssignmentDecision] = {}
            total_batches = (
                len(candidates) + EVIDENCE_ASSIGNMENT_BATCH_SIZE - 1
            ) // EVIDENCE_ASSIGNMENT_BATCH_SIZE
            semaphore = asyncio.Semaphore(EVIDENCE_ASSIGNMENT_MAX_CONCURRENCY)

            async def classify_batch(start: int) -> list[EvidenceAssignmentDecision]:
                """分类一个候选小批次并校验模型逐项返回真实 Chunk。

                Args:
                    start: 当前批次在合并候选列表中的起始下标。

                Returns:
                    与当前批次候选一一对应的结构化归属决定。

                Raises:
                    ValueError: 模型遗漏、重复或虚构当前批次的 Chunk UUID。
                    StructuredSchemaValidationError: 当前批次两轮请求后仍违反 Schema。
                """
                candidate_batch = candidates[start : start + EVIDENCE_ASSIGNMENT_BATCH_SIZE]
                blocks = [
                    {
                        "chunk_id": str(item["evidence"].chunk_id),
                        "document_title": item["evidence"].document_title,
                        "pages": [
                            item["evidence"].page_start,
                            item["evidence"].page_end,
                        ],
                        "retrieval_contexts": [
                            context.model_dump(mode="json")
                            for context in item["retrieval_contexts"]
                        ],
                        "content": item["evidence"].content,
                    }
                    for item in candidate_batch
                ]
                batch_number = start // EVIDENCE_ASSIGNMENT_BATCH_SIZE + 1
                # 同一批次 Schema 失败只重送本批；成功批次无需再次消耗模型调用。
                # 并发度受信号量限制；批次候选不完整时仍由图级逻辑重做整轮。
                async with semaphore:
                    for batch_attempt in range(MAX_EVIDENCE_ASSIGNMENT_BATCH_ATTEMPTS):
                        instruction = (
                            "对每个候选 Chunk 判断它直接支持哪些对象×比较维度单元格。"
                            "supports 中每项必须同时给出 object=A/B、给定维度原文和"
                            "HIGH/MEDIUM/LOW；不得返回未给定的维度，也不得重复同一"
                            "对象×维度。只有能从 Chunk 正文直接支持某个对象×维度时才"
                            "填写 supports；没有直接支持项时，即使与主题相关或顺带提及"
                            "对象，也必须返回 supports=[] 且 relevance=IRRELEVANT。"
                            "有支持项时 relevance 只能为 HIGH/MEDIUM/LOW。"
                            "对象按原始请求和给定完整名称理解；只提及其所属国家、机构、人物"
                            "或上位概念，不等于直接支持该具体对象。"
                            "直接同时比较双方时，可为同一维度分别返回 A 和 B 两项支持。"
                            f"本批共 {len(blocks)} 个候选，decisions 必须恰好返回"
                            f" {len(blocks)} 项，"
                            "每个输入 chunk_id 原样返回一次且不得新增。"
                            f"\n批次：{batch_number}/{total_batches}"
                            f"\n用户原始请求：{state.get('query', '')}"
                            f"\n主题：{state['compare_request'].topic}"
                            f"\n对象A：{state['compare_object_a'].model_dump_json()}"
                            f"\n对象B：{state['compare_object_b'].model_dump_json()}"
                            "\n允许维度："
                            f"{json.dumps(state['compare_dimensions'], ensure_ascii=False)}"
                            f"\n候选：{json.dumps(blocks, ensure_ascii=False)}"
                            + (
                                "\n上轮归属校验失败："
                                + state["evidence_assignment_error"]
                                + "。请逐项修正，候选 UUID 与维度必须原样对应。"
                                if state.get("evidence_assignment_error")
                                else ""
                            )
                            + (
                                "\n本批上次输出未通过结构与归属规则校验。"
                                "请逐项检查 supports、relevance 和重复单元格后重新输出。"
                                if batch_attempt else ""
                            )
                        )
                        try:
                            batch = await self.gateway.structured(
                                instruction=instruction,
                                schema=EvidenceAssignmentBatch,
                                system_prompt=self.system_prompt,
                            )
                            break
                        except StructuredSchemaValidationError:
                            if batch_attempt + 1 >= MAX_EVIDENCE_ASSIGNMENT_BATCH_ATTEMPTS:
                                raise
                batch_decisions = {item.chunk_id: item for item in batch.decisions}
                expected_batch = {item["evidence"].chunk_id for item in candidate_batch}
                if set(batch_decisions) != expected_batch:
                    raise _AssignmentCoverageError(
                        f"EvidenceAssignment 第 {batch_number} 批未逐项覆盖真实 Candidate Chunk"
                    )
                return list(batch_decisions.values())

            starts = list(range(0, len(candidates), EVIDENCE_ASSIGNMENT_BATCH_SIZE))
            # 收齐全部并发批次后再决定重试，避免上轮未完成的模型调用与下一轮交错。
            batch_results = await asyncio.gather(
                *(classify_batch(start) for start in starts), return_exceptions=True
            )
            for result in batch_results:
                if isinstance(result, Exception) and not isinstance(
                    result, _AssignmentCoverageError
                ):
                    raise result
            for result in batch_results:
                if isinstance(result, _AssignmentCoverageError):
                    return retry_or_fail(str(result))
            for batch_decisions in batch_results:
                decisions.update((item.chunk_id, item) for item in batch_decisions)

            # 3. 全批次结果必须与合并候选一一对应，过滤并选取最终预算后才编排引用编号。
            expected = {item["evidence"].chunk_id for item in candidates}
            if set(decisions) != expected:
                return retry_or_fail("EvidenceAssignment 未逐项覆盖真实 Candidate Chunk")
            allowed_dimensions = set(state["compare_dimensions"])
            for decision in decisions.values():
                if any(
                    support.dimension not in allowed_dimensions for support in decision.supports
                ):
                    return retry_or_fail("EvidenceAssignment 返回了本次任务之外的比较维度")
            evidence: list[EvidenceRead] = []
            citations: list[CitationRead] = []
            assignments: list[dict[str, object]] = []
            final_candidates = _select_final_candidates(
                candidates, decisions, state["compare_dimensions"]
            )
            for candidate in final_candidates:
                source = candidate["evidence"]
                decision = decisions[source.chunk_id]
                if decision.relevance == "IRRELEVANT" or not decision.supports:
                    continue
                index = len(evidence) + 1
                supports_a = any(item.object == "A" for item in decision.supports)
                supports_b = any(item.object == "B" for item in decision.supports)
                assignment = "SHARED" if supports_a and supports_b else "A" if supports_a else "B"
                cell_supports = [item.model_dump(mode="json") for item in decision.supports]
                supported_dimensions = list(
                    dict.fromkeys(item.dimension for item in decision.supports)
                )
                retrieval_contexts = [
                    item.model_dump(mode="json") for item in candidate["retrieval_contexts"]
                ]
                evidence.append(source.model_copy(update={"citation_index": index}))
                citations.append(candidate["citation"].model_copy(update={"citation_index": index}))
                assignments.append(
                    {
                        "chunk_id": source.chunk_id,
                        "document_id": source.document_id,
                        "citation_id": index,
                        "assignment": assignment,
                        "supports_a": supports_a,
                        "supports_b": supports_b,
                        "cell_supports": cell_supports,
                        "supported_dimensions": supported_dimensions,
                        "relevance": decision.relevance,
                        "retrieval_branches": list(
                            dict.fromkeys(item["object"] for item in retrieval_contexts)
                        ),
                        "retrieval_contexts": retrieval_contexts,
                        "assignment_reason": decision.assignment_reason,
                    }
                )
            return {
                "evidence": evidence,
                "citations": citations,
                "evidence_assignments": assignments,
                "evidence_assignment_attempts": attempt,
                "evidence_assignment_status": "PASS",
                "evidence_assignment_error": "",
            }

        return await self._tracked("EvidenceAssignment", state, operation)

    async def _check_balance(self, state: ZhijianState) -> dict[str, Any]:
        """按可配置规则检查全局双侧数量及逐维度单元格覆盖。

        Args:
            state: 当前比较 Run 的图状态，包含该节点所需输入。

        Returns:
            双侧证据平衡状态及覆盖统计。
        """

        async def operation() -> dict[str, Any]:
            """只基于已确认 Evidence 计算平衡状态。

            Returns:
                证据平衡状态、覆盖统计及证据等级。
            """
            # 1. 按模型已确认的归属统计双侧专属证据及文献覆盖。
            # SHARED 证据可解释共同背景，但不代替任一侧的专属材料。
            settings = get_settings()
            evidence_by_chunk = {item.chunk_id: item for item in state["evidence"]}
            # 否则会出现界面显示 B=0、后端却继续生成双边结论的矛盾状态。
            a_items = [item for item in state["evidence_assignments"] if item["assignment"] == "A"]
            b_items = [item for item in state["evidence_assignments"] if item["assignment"] == "B"]
            shared_items = [
                item for item in state["evidence_assignments"] if item["assignment"] == "SHARED"
            ]
            a_documents = {item["document_id"] for item in a_items}
            b_documents = {item["document_id"] for item in b_items}
            # 2. 从最终 Evidence 计算年份、语言和相关性分布，供结果解释使用。
            years = sorted(
                {
                    evidence_by_chunk[item["chunk_id"]].publication_year
                    for item in state["evidence_assignments"]
                    if evidence_by_chunk[item["chunk_id"]].publication_year
                }
            )
            languages: dict[str, set[uuid.UUID]] = {}
            for item in state["evidence"]:
                languages.setdefault(item.language_code, set()).add(item.document_id)
            relevance_distribution = {
                level: sum(item["relevance"] == level for item in state["evidence_assignments"])
                for level in ("HIGH", "MEDIUM", "LOW")
            }
            # 3. 每个维度分别统计 A/B 单元格；同一 Chunk 可支持多个单元格，但在每个
            # 单元格内只计一次。至少一个维度双侧都有证据，才能进入 CrossAnalysis。
            dimension_coverage: list[DimensionEvidenceCoverage] = []
            for dimension in state["compare_dimensions"]:
                a_count = sum(
                    any(
                        support["object"] == "A" and support["dimension"] == dimension
                        for support in item["cell_supports"]
                    )
                    for item in state["evidence_assignments"]
                )
                b_count = sum(
                    any(
                        support["object"] == "B" and support["dimension"] == dimension
                        for support in item["cell_supports"]
                    )
                    for item in state["evidence_assignments"]
                )
                status = (
                    "SUFFICIENT"
                    if a_count and b_count
                    else "PARTIAL"
                    if a_count or b_count
                    else "INSUFFICIENT"
                )
                dimension_coverage.append(
                    DimensionEvidenceCoverage(
                        dimension=dimension,
                        object_a_evidence_count=a_count,
                        object_b_evidence_count=b_count,
                        status=status,
                    )
                )
            # 4. 全局数量、文献数与至少一个双侧可比维度共同决定是否允许比较。
            has_comparable_dimension = any(
                item.status == "SUFFICIENT" for item in dimension_coverage
            )
            sufficient = (
                len(a_items) >= settings.zhijian_min_side_evidence
                and len(b_items) >= settings.zhijian_min_side_evidence
                and len(a_documents) >= settings.zhijian_min_side_documents
                and len(b_documents) >= settings.zhijian_min_side_documents
                and has_comparable_dimension
            )
            balance = EvidenceBalance(
                status="SUFFICIENT" if sufficient else "INSUFFICIENT_EVIDENCE",
                object_a_evidence_count=len(a_items),
                object_b_evidence_count=len(b_items),
                shared_evidence_count=len(shared_items),
                object_a_document_count=len(a_documents),
                object_b_document_count=len(b_documents),
                year_coverage=[years[0], years[-1]] if years else [],
                source_diversity=len({item.document_id for item in state["evidence"]}),
                source_language_coverage={key: len(value) for key, value in languages.items()},
                relevance_distribution=relevance_distribution,
                dimension_coverage=dimension_coverage,
            )
            return {
                "evidence_balance": balance,
                "evidence_level": "MEDIUM" if sufficient else "INSUFFICIENT",
            }

        return await self._tracked("EvidenceBalanceCheck", state, operation)

    def _analysis_instruction(self, state: ZhijianState, issues: list[str] | None = None) -> str:
        """构造只允许使用已确认 Evidence 的结构化比较任务说明。

        Args:
            state: 包含动态维度、证据和运行时归属的图状态。
            issues: 可选的审核修订问题。

        Returns:
            供结构化模型生成核心比较草稿的完整任务说明。
        """
        revision = f"\n必须修正的问题：{json.dumps(issues, ensure_ascii=False)}" if issues else ""
        assignments = {item["citation_id"]: item for item in state["evidence_assignments"]}
        cell_citations = {
            dimension: {"A": [], "B": []} for dimension in state["compare_dimensions"]
        }
        evidence_payload = []
        for item in state["evidence"]:
            assignment = assignments[item.citation_index]
            for support in assignment["cell_supports"]:
                cell_citations[support["dimension"]][support["object"]].append(item.citation_index)
            evidence_payload.append(
                {
                    "citation_id": item.citation_index,
                    "title": item.document_title,
                    "year": item.publication_year,
                    "language": item.language_code,
                    "page": [item.page_start, item.page_end],
                    "assignment": assignment["assignment"],
                    "supports_a": assignment["supports_a"],
                    "supports_b": assignment["supports_b"],
                    "cell_supports": assignment["cell_supports"],
                    "content": item.content,
                }
            )
        return (
            "只根据随后提供的已确认 Evidence 生成专业 A/B 比较分析。"
            "保持用户原始请求中两项对象的完整含义和粒度，使用给定对象名称；"
            "不能把具体对象扩大成所属国家、机构、人物或上位概念，"
            "也不能把一侧的专属名称、时期、版本或其他限定移到另一侧。"
            "证据中的类比名称和背景材料不能自动替换对象名称或当作对象本身的事实。"
            "summary、object_a_summary、object_b_summary 中每个事实结论必须在句末使用 [n]；"
            "句末不得[n,m]要[n][m]"
            "similarities 和 differences 的 citation_ids 必须列出对应引用。"
            "comparison_matrix 必须严格按给定维度逐行返回，不得增删、改名或换序；"
            "每行分别填写支持 A、B、共同点的引用编号，编号必须来自该行维度"
            "对应的对象单元格清单。共同点可综合 A、B 各自材料归纳，"
            "不要求每条证据同时支持双方；shared_citation_ids 是共同点实际使用的"
            "引用集合，合起来必须覆盖该维度的 A、B 双方，也可使用同时支持双方的证据。"
            "应依据原文归纳具体共同点，表述只概括双方证据均支持的范围；"
            "双方有材料但未体现共同特征时，应说明未发现明确共同点，不得强行凑同。"
            "没有某侧证据时，"
            "对应文字填写“证据不足”，对应引用数组返回空数组，绝不能用常识补全。"
            "timeline 只返回对象 A 或 B 的事件；year 必须在所引 Evidence 正文、标题"
            "或出版年份中明确出现，citation_ids 必须支撑对应对象。没有可靠年份就返回空数组。"
            "topic_distribution 返回 2 至 6 个当前 Evidence 中反复出现或具有代表性的研究主题；"
            "object=A/B 时引用必须直接支持对应对象，object=SHARED 时只能引用同时支持双方的证据。"
            "热点只表示本次知识库证据中的可观察主题，不得声称总体占比、流行度或全球趋势；"
            "如果证据不足以归纳主题，可以返回空数组。"
            "不要返回总 citation_ids，该字段由后端按分侧引用确定性合并。"
            f"\n输出语言：{state['output_language']}"
            f"\n用户原始请求：{state.get('query', '')}"
            f"\n主题：{state['compare_request'].topic}"
            f"\n对象A：{state['compare_object_a'].canonical_name}"
            f"\n对象B：{state['compare_object_b'].canonical_name}"
            f"\n比较维度：{json.dumps(state['compare_dimensions'], ensure_ascii=False)}"
            f"\n对象×维度可用引用：{json.dumps(cell_citations, ensure_ascii=False)}"
            f"\n已确认 Evidence：{json.dumps(evidence_payload, ensure_ascii=False)}"
            f"{revision}"
        )

    async def _cross_analysis(self, state: ZhijianState) -> dict[str, Any]:
        """基于固定 Evidence 生成核心比较草稿，校验失败时最多重做一次。

        Args:
            state: 当前比较 Run 的图状态，包含该节点所需输入。

        Returns:
            比较分析草稿或待重试的校验状态。
        """

        async def operation() -> dict[str, Any]:
            """调用结构化模型并记录本轮轮次与可审计纠错反馈。

            Returns:
                核心分析草稿、矩阵原稿和生成轮次。

            Raises:
                ValueError: 图异常重入导致生成超过两轮。
            """
            attempt = state.get("compare_analysis_attempts", 0) + 1
            if attempt > MAX_COMPARE_ANALYSIS_ATTEMPTS:
                raise ValueError("智鉴核心分析生成次数超过上限")
            instruction = self._analysis_instruction(state)
            if correction := state.get("compare_analysis_validation_error"):
                # 只反馈确定性规则原因；不重传上轮草稿，也不改变 Evidence 编号。
                instruction += f"\n上轮引用校验失败：{correction}。请重新生成并修正引用。"
            core = await self.gateway.structured(
                instruction=instruction,
                schema=ZhijianCoreAnalysisDraft,
                system_prompt=self.system_prompt,
                thinking=False,
            )
            draft = ZhijianAnalysisDraft(
                summary=core.summary,
                object_a_summary=core.object_a_summary,
                object_b_summary=core.object_b_summary,
                similarities=core.similarities,
                differences=core.differences,
                comparison_matrix=[],
                timeline=core.timeline,
                topic_distribution=core.topic_distribution,
                limitations=core.limitations,
            )
            return {
                "zhijian_draft": draft,
                "zhijian_matrix_draft": core.comparison_matrix,
                "compare_analysis_attempts": attempt,
            }

        return await self._tracked("CrossAnalysis", state, operation)

    async def _compare_analysis_validation(self, state: ZhijianState) -> dict[str, Any]:
        """检查总览及共同点、差异点引用，首轮失败时要求重新生成。

        Args:
            state: 包含 CrossAnalysis 草稿的智鉴状态。

        Returns:
            PASS 或 RETRY 判定、失败原因及两类条目数量。

        Raises:
            ValueError: 第二轮结论仍缺少或越界引用。
        """

        async def operation() -> dict[str, Any]:
            """用固定 Evidence 编号判定核心分析是否可进入后续整理。

            Returns:
                判定结果、失败原因与两类结论数量。

            Raises:
                ValueError: 第二轮引用仍不满足固定 Evidence 边界。
            """
            draft = state["zhijian_draft"]
            allowed = {item.citation_index for item in state["evidence"]}
            summary_ids = {int(value) for value in re.findall(r"\[(\d+)]", draft.summary)}
            failure = ""
            if not summary_ids or not summary_ids <= allowed:
                failure = "比较正文缺少 Citation 或引用编号越界"
            elif any(
                not item.citation_ids or not set(item.citation_ids) <= allowed
                for item in draft.similarities
            ):
                failure = "共同点缺少 Citation 或引用编号越界"
            elif any(
                not item.citation_ids or not set(item.citation_ids) <= allowed
                for item in draft.differences
            ):
                failure = "差异点缺少 Citation 或引用编号越界"
            if failure:
                if state.get("compare_analysis_attempts", 0) >= MAX_COMPARE_ANALYSIS_ATTEMPTS:
                    raise ValueError(failure)
                return {
                    "compare_analysis_validation_status": "RETRY",
                    "compare_analysis_validation_error": failure,
                }
            return {
                "compare_analysis_validation_status": "PASS",
                "compare_analysis_validation_error": "",
                "intermediate_results": {
                    "similarity_count": len(draft.similarities),
                    "difference_count": len(draft.differences),
                }
            }

        return await self._tracked("CompareAnalysisValidation", state, operation)

    async def _timeline_builder(self, state: ZhijianState) -> dict[str, Any]:
        """按对象归属和 Evidence 中的明确年份校验双轨时间节点。

        Args:
            state: 当前比较 Run 的图状态，包含该节点所需输入。

        Returns:
            经年份与来源约束筛选的时间线。
        """

        async def operation() -> dict[str, Any]:
            """剔除错侧、无来源年份和重复节点，再按年份稳定排序。

            Returns:
                经校验并排序的双侧时间线节点。
            """
            assignment_by_id = {item["citation_id"]: item for item in state["evidence_assignments"]}
            evidence_by_id = {item.citation_index: item for item in state["evidence"]}
            timeline = []
            seen: set[tuple[int, str, str]] = set()
            for item in state["zhijian_draft"].timeline:
                if item.object not in {"A", "B"}:
                    continue
                support_key = "supports_a" if item.object == "A" else "supports_b"
                citation_ids = []
                for citation_id in dict.fromkeys(item.citation_ids):
                    assignment = assignment_by_id.get(citation_id, {})
                    evidence = evidence_by_id.get(citation_id)
                    if evidence is None or assignment.get(support_key) is not True:
                        continue
                    year_is_grounded = (
                        evidence.publication_year == item.year
                        or str(item.year) in evidence.document_title
                        or str(item.year) in evidence.content
                    )
                    if year_is_grounded:
                        citation_ids.append(citation_id)
                key = (item.year, item.object, item.event.strip())
                if not citation_ids or key in seen:
                    continue
                seen.add(key)
                timeline.append(item.model_copy(update={"citation_ids": citation_ids}))
            timeline.sort(key=lambda item: (item.year, item.object, item.event))
            draft = state["zhijian_draft"].model_copy(update={"timeline": timeline})
            return {
                "zhijian_draft": draft,
                "intermediate_results": {"timeline_count": len(timeline)},
            }

        return await self._tracked("TimelineBuilder", state, operation)

    async def _topic_aggregator(self, state: ZhijianState) -> dict[str, Any]:
        """校验研究热点的证据归属并过滤错侧或重复主题。

        Args:
            state: 当前比较 Run 的图状态，包含该节点所需输入。

        Returns:
            经主题和来源约束筛选的热点分布。
        """

        async def operation() -> dict[str, Any]:
            """过滤没有真实分侧引用或语义重复的主题。

            Returns:
                经校验的研究热点分布。
            """
            # 热点结论只能保留与 A/B/SHARED 归属一致的真实 Citation。
            # 模型负责语义归纳，后端负责收紧引用边界，避免把错侧证据包装成热点。
            assignment_by_id = {item["citation_id"]: item for item in state["evidence_assignments"]}
            topics = []
            seen_topics: set[tuple[str, str]] = set()
            for item in state["zhijian_draft"].topic_distribution:
                if item.object == "SHARED":
                    citation_ids = [
                        citation_id
                        for citation_id in dict.fromkeys(item.citation_ids)
                        if assignment_by_id.get(citation_id, {}).get("assignment") == "SHARED"
                    ]
                else:
                    support_key = "supports_a" if item.object == "A" else "supports_b"
                    citation_ids = self._valid_object_citations(
                        item.citation_ids,
                        assignment_by_id,
                        support_key=support_key,
                    )
                topic_key = (item.object, item.topic.strip().casefold())
                if not citation_ids or topic_key in seen_topics:
                    continue
                seen_topics.add(topic_key)
                topics.append(item.model_copy(update={"citation_ids": citation_ids}))

            draft = state["zhijian_draft"].model_copy(
                update={"topic_distribution": topics}
            )
            return {
                "zhijian_draft": draft,
                "intermediate_results": {"topic_count": len(topics)},
            }

        return await self._tracked("TopicAggregator", state, operation)

    async def _matrix_generator(self, state: ZhijianState) -> dict[str, Any]:
        """按运行时维度校验分侧引用及共同点引用集合的双侧覆盖。

        Args:
            state: 当前比较 Run 的图状态，包含该节点所需输入。

        Returns:
            含各维度双侧证据单元格的比较矩阵。
        """

        async def operation() -> dict[str, Any]:
            """补齐缺失维度，并阻止错误归属的 Citation 支撑矩阵单元。

            Returns:
                覆盖本次维度的比较矩阵。
            """
            # 1. 按运行时维度匹配模型原稿，未知和重复维度不进入发布结果。
            assignment_by_id = {item["citation_id"]: item for item in state["evidence_assignments"]}
            source_rows = state.get("zhijian_matrix_draft", [])
            by_dimension: dict[str, ComparisonMatrixDraftRow] = {}
            for row in source_rows:
                if (
                    row.dimension in state["compare_dimensions"]
                    and row.dimension not in by_dimension
                ):
                    by_dimension[row.dimension] = row

            # 2. 逐维度校验对象及共同点的引用，缺少的维度保留明确占位。
            rows: list[ComparisonMatrixRow] = []
            for dimension in state["compare_dimensions"]:
                source = by_dimension.get(dimension)
                if source is None:
                    source = ComparisonMatrixDraftRow(
                        dimension=dimension,
                        object_a="证据不足",
                        object_b="证据不足",
                        similarity="证据不足",
                        difference="证据不足",
                        object_a_citation_ids=[],
                        object_b_citation_ids=[],
                        shared_citation_ids=[],
                    )
                a_ids = self._valid_cell_citations(
                    source.object_a_citation_ids,
                    assignment_by_id,
                    object_code="A",
                    dimension=dimension,
                )
                b_ids = self._valid_cell_citations(
                    source.object_b_citation_ids,
                    assignment_by_id,
                    object_code="B",
                    dimension=dimension,
                )
                # 共同点可由不同材料分别支撑双方，但只核对该结论实际引用的集合，
                # 不从 A/B 描述借用引用；错维度或单侧引用不能组成双边共同点。
                shared_a_ids = set(self._valid_cell_citations(
                    source.shared_citation_ids,
                    assignment_by_id,
                    object_code="A",
                    dimension=dimension,
                ))
                shared_b_ids = set(self._valid_cell_citations(
                    source.shared_citation_ids,
                    assignment_by_id,
                    object_code="B",
                    dimension=dimension,
                ))
                shared_ids = [
                    citation_id
                    for citation_id in dict.fromkeys(source.shared_citation_ids)
                    if citation_id in shared_a_ids or citation_id in shared_b_ids
                ] if shared_a_ids and shared_b_ids else []
                citation_ids = list(dict.fromkeys([*a_ids, *b_ids, *shared_ids]))
                rows.append(
                    ComparisonMatrixRow(
                        dimension=dimension,
                        object_a=source.object_a if a_ids else "证据不足",
                        object_b=source.object_b if b_ids else "证据不足",
                        similarity=source.similarity if shared_ids else "证据不足",
                        difference=source.difference if a_ids and b_ids else "证据不足",
                        object_a_citation_ids=a_ids,
                        object_b_citation_ids=b_ids,
                        shared_citation_ids=shared_ids,
                        citation_ids=citation_ids,
                    )
                )
            # 3. 以校验后的矩阵替换草稿，最终 CitationValidator 再映射真实来源。
            draft = state["zhijian_draft"].model_copy(update={"comparison_matrix": rows})
            return {"zhijian_draft": draft}

        return await self._tracked("MatrixGenerator", state, operation)

    @staticmethod
    def _valid_object_citations(
        citation_ids: list[int],
        assignment_by_id: dict[int, dict[str, object]],
        *,
        support_key: str,
    ) -> list[int]:
        """保留真实存在且经 EvidenceAssignment 支持指定对象的引用编号。

        Args:
            citation_ids: 模型为对象级结论返回的引用编号。
            assignment_by_id: Citation 编号到运行时证据归属的映射。
            support_key: ``supports_a`` 或 ``supports_b``。

        Returns:
            保持原顺序的合法去重引用编号。
        """
        return [
            citation_id
            for citation_id in dict.fromkeys(citation_ids)
            if assignment_by_id.get(citation_id, {}).get(support_key) is True
        ]

    @staticmethod
    def _valid_cell_citations(
        citation_ids: list[int],
        assignment_by_id: dict[int, dict[str, object]],
        *,
        object_code: str,
        dimension: str,
    ) -> list[int]:
        """保留经 EvidenceAssignment 支持指定对象×维度单元格的引用编号。

        Args:
            citation_ids: 模型为一个矩阵单元返回的引用编号。
            assignment_by_id: Citation 编号到运行时证据归属的映射。
            object_code: A 或 B 对象侧。
            dimension: 当前矩阵行的比较维度。

        Returns:
            保持原顺序的合法去重引用编号。
        """
        return [
            citation_id
            for citation_id in dict.fromkeys(citation_ids)
            if ZhijianGraph._supports_cell(
                assignment_by_id.get(citation_id, {}),
                object_code=object_code,
                dimension=dimension,
            )
        ]

    @staticmethod
    def _supports_cell(assignment: dict[str, object], *, object_code: str, dimension: str) -> bool:
        """判断一条运行时 Evidence 是否直接支持指定对象×维度。

        Args:
            assignment: Citation 对应的逐 Chunk 归属记录。
            object_code: A 或 B 对象侧。
            dimension: 当前比较维度。

        Returns:
            存在完全匹配的单元格支持时为真。
        """
        return any(
            item.get("object") == object_code and item.get("dimension") == dimension
            for item in assignment.get("cell_supports", [])
        )

    async def _citation_validator(self, state: ZhijianState) -> dict[str, Any]:
        """校验所有专业结论引用并构造最终智鉴结果。

        Args:
            state: 当前比较 Run 的图状态，包含该节点所需输入。

        Returns:
            通过来源映射校验的结果及实际使用的引用。
        """

        async def operation() -> dict[str, Any]:
            """把引用编号映射回真实 Document/Chunk/page 后允许发布。

            Returns:
                引用核验状态、来源映射与可发布结果。

            Raises:
                ValueError: 任一专业结论缺少 Citation 或编号不属于当前 Evidence。
            """
            # 1. 用本次固定 Evidence 建立允许的编号集合，并汇总所有发布字段的引用。
            allowed = {item.citation_index for item in state["evidence"]}
            draft = state["zhijian_draft"]
            cited_groups = [
                draft.similarities,
                draft.differences,
                draft.timeline,
                draft.topic_distribution,
            ]
            # 结构化总览仍在正文中保留 [n]，便于最终消息和导出结果直接可读。
            used = {int(value) for value in re.findall(r"\[(\d+)]", draft.summary)}
            if not used or not used <= allowed:
                raise ValueError("比较正文缺少 Citation 或引用编号越界")
            # 2. 逐项拒绝缺引和越界；有结论的矩阵行也必须提供来源。
            for group in cited_groups:
                for item in group:
                    indices = set(item.citation_ids)
                    if not indices or not indices <= allowed:
                        raise ValueError("专业结论包含缺失或越界 Citation")
                    used.update(indices)
            for row in draft.comparison_matrix:
                indices = set(row.citation_ids)
                if indices and not indices <= allowed:
                    raise ValueError("矩阵包含越界 Citation")
                has_conclusion = any(
                    value != "证据不足"
                    for value in (row.object_a, row.object_b, row.similarity, row.difference)
                )
                if has_conclusion and not indices:
                    raise ValueError("有结论的矩阵行缺少 Citation")
                used.update(indices)
            # 3. 只把实际使用的编号映射到真实来源，并构造可发布结果。
            references = self._reference_documents(state, used)
            result = self._result_from_draft(
                state,
                draft,
                references=references,
                status="SUCCESS",
            )
            citations = [item for item in state["citations"] if item.citation_index in used]
            return {
                "citation_validation_status": "PASS",
                "validated_citations": citations,
                "used_citation_indices": sorted(used),
                "zhijian_result": result,
                "final_answer": result.summary,
            }

        return await self._tracked("CitationValidator", state, operation)

    @staticmethod
    def _reference_documents(
        state: ZhijianState, used_citation_ids: set[int]
    ) -> list[ReferenceDocumentItem]:
        """按最终使用的 Citation 生成不重复的真实参考文献清单。

        Args:
            state: 包含本次 Run 固定 Evidence 的智鉴状态。
            used_citation_ids: 已通过范围校验的最终引用编号。

        Returns:
            按 Citation 顺序排列、每个 Document 仅保留一次的参考文献。
        """
        references: list[ReferenceDocumentItem] = []
        seen_documents: set[uuid.UUID] = set()
        for item in sorted(state["evidence"], key=lambda evidence: evidence.citation_index):
            if (
                item.citation_index not in used_citation_ids
                or item.document_id in seen_documents
            ):
                continue
            seen_documents.add(item.document_id)
            references.append(
                ReferenceDocumentItem(
                    citation_id=item.citation_index,
                    document_id=item.document_id,
                    title=item.document_title,
                )
            )
        return references

    async def _finalize_restricted(self, state: ZhijianState) -> dict[str, Any]:
        """对澄清或证据不足返回诚实受限结果，兼容历史受限状态。

        Args:
            state: 当前比较 Run 的图状态，包含该节点所需输入。

        Returns:
            明确说明证据边界的受限比较结果。
        """

        async def operation() -> dict[str, Any]:
            """不调用分析模型、不生成无证据比较结论。

            Returns:
                受限结果及不包含虚构来源的空引用列表。
            """
            status = state.get("compare_block_status") or "INSUFFICIENT_EVIDENCE"
            if status == "INSUFFICIENT_EVIDENCE" or state.get("evidence_balance"):
                status = "INSUFFICIENT_EVIDENCE"
                balance = state.get("evidence_balance")
                missing_objects: list[str] = []
                if balance and balance.object_a_evidence_count == 0:
                    missing_objects.append(state["compare_object_a"].canonical_name)
                if balance and balance.object_b_evidence_count == 0:
                    missing_objects.append(state["compare_object_b"].canonical_name)
                if missing_objects:
                    names = "、".join(missing_objects)
                    message = (
                        f"当前知识库没有检索到 {names} 的专属证据，"
                        "无法形成可靠的双边比较。请补充相关文献后重试。"
                    )
                else:
                    message = (
                        "当前知识库的双侧专属证据数量或来源不足，"
                        "无法形成可靠的双边比较。请补充双方相关文献后重试。"
                    )
            else:
                message = state.get("clarification_question", "请补充比较请求。")
            result = ZhijianResult(
                status=status,
                clarification_question=message if status != "INSUFFICIENT_EVIDENCE" else None,
                topic=getattr(state.get("compare_request"), "topic", None),
                object_a=state.get("compare_object_a"),
                object_b=state.get("compare_object_b"),
                compare_dimensions=state.get("compare_dimensions", []),
                summary=message,
                evidence=[
                    ZhijianEvidenceRead(**item) for item in state.get("evidence_assignments", [])
                ],
                evidence_balance=state.get("evidence_balance"),
                limitations=[message],
                evidence_level="INSUFFICIENT",
            )
            return {
                "zhijian_result": result,
                "final_answer": message,
                "reviewer_status": "PASS",
                "citation_validation_status": "PASS",
                "validated_citations": [],
                "used_citation_indices": [],
                "evidence_level": "INSUFFICIENT",
            }

        return await self._tracked("FinalizeRestricted", state, operation)

    @staticmethod
    def _result_from_draft(
        state: ZhijianState,
        draft: ZhijianAnalysisDraft,
        *,
        references: list[ReferenceDocumentItem],
        status: str,
    ) -> ZhijianResult:
        """把已验证草稿与运行时证据事实组合成 API 结果。

        Args:
            state: 当前比较 Run 的图状态，包含该节点所需输入。
            draft: 已经完成规则校验的比较分析草稿。
            references: 从本次实际引用归集的真实文献。
            status: 最终结果的业务状态。

        Returns:
            包含真实证据与引用元数据的智鉴结果。
        """
        request = state["compare_request"]
        time_range = [value for value in (request.year_from, request.year_to) if value]
        return ZhijianResult(
            status=status,
            topic=request.topic,
            object_a=state["compare_object_a"],
            object_b=state["compare_object_b"],
            time_range=time_range,
            compare_dimensions=state["compare_dimensions"],
            summary=draft.summary,
            object_a_summary=draft.object_a_summary,
            object_b_summary=draft.object_b_summary,
            similarities=draft.similarities,
            differences=draft.differences,
            comparison_matrix=draft.comparison_matrix,
            timeline=draft.timeline,
            topic_distribution=draft.topic_distribution,
            references=references,
            evidence=[ZhijianEvidenceRead(**item) for item in state["evidence_assignments"]],
            evidence_balance=state["evidence_balance"],
            limitations=draft.limitations,
            evidence_level=state["evidence_level"],
        )
