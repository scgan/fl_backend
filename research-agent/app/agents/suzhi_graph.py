"""实现溯知 Evidence First 文献溯源与综述 LangGraph。

图固定执行概念解析、跨语言检索、去重、文献整理、主题分析、综述撰写及引用校验。
模型只处理受控 Evidence，不能访问数据库；参考文献格式由确定性代码生成。
"""

from __future__ import annotations

import json
import re
import uuid
from collections import Counter
from collections.abc import Awaitable, Callable
from typing import Any

from langgraph.graph import END, START, StateGraph

from app.agents.gateway import ModelGateway
from app.agents.suzhi_schemas import (
    LiteratureAnalysisDraft,
    LiteratureAnalysisWithGraph,
    ProvenanceGraph,
    ReviewConcept,
    ReviewDraft,
    ReviewQueryPlan,
    SuzhiResult,
)
from app.agents.suzhi_state import SuzhiState
from app.agents.tools import RetrievalTool
from app.retrieval.query_processing import BilingualQueryProcessor
from app.retrieval.schemas import CitationRead, EvidenceRead, RetrievalFilters

NodeOperation = Callable[[], Awaitable[dict[str, Any]]]
StepRunner = Callable[[str, SuzhiState, NodeOperation], Awaitable[dict[str, Any]]]

# 业务层每段最多生成两轮；网关内部的传输或 Schema 重试另行计数。
MAX_ANALYSIS_ATTEMPTS = 2
MAX_REVIEW_ATTEMPTS = 2
# 多查询合并去重后的最终证据上限；每条查询仍沿用 DEEP 的十条返回预算。
MAX_REVIEW_EVIDENCE = 30


class SuzhiGraph:
    """执行研究主题输入到可追溯综述结果的固定工作流。"""

    def __init__(
        self,
        retrieval: RetrievalTool,
        gateway: ModelGateway,
        system_prompt: str,
        step_runner: StepRunner,
    ) -> None:
        """注入受控检索、固定模型、版本化 Prompt 和步骤记录器。

        Args:
            retrieval: 已绑定当前用户权限边界的检索 Tool。
            gateway: 只接收受控输入的结构化模型网关。
            system_prompt: Run 创建时冻结的溯知 Prompt 正文。
            step_runner: 在每个节点前后持久化 AgentStep 的运行回调。
        """
        # 检索 Tool 统一执行知识库权限、READY 文档白名单和 Evidence 质量门槛。
        self.retrieval = retrieval
        # 模型网关只负责受控的概念理解、分析草稿和综述撰写，不直接访问数据源。
        self.gateway = gateway
        # Prompt 从 Run 固定版本读取，运行期间不得切换到当前激活版本。
        self.system_prompt = system_prompt
        # 步骤回调负责节点可观察状态、Evidence 检查点和脱敏失败信息。
        self.step_runner = step_runner
        # 图在构造期编译一次，单次运行的可变数据全部通过 SuzhiState 传递。
        self.graph = self._build()

    def _build(self):
        """编译固定 Evidence、两段有界回退的溯知文献综述图。

        Returns:
            可异步调用并以稳定节点名记录步骤的已编译 LangGraph。
        """
        # 1. 注册从概念理解到引用校验的全部业务节点
        # 节点名称同时用于 SSE 进度和 AgentStep 审计；模型只能处理节点收到的受控
        # Evidence，不能改变拓扑、跳过证据整理阶段或直接构造最终引用。
        builder = StateGraph(SuzhiState)
        # 从用户主题中提取规范概念、输出语言和有限的中英文相关词，不生成文献事实。
        builder.add_node("ConceptExtraction", self._concept_extraction)
        # 将原始主题与规范概念扩展为至多五条检索表达，且始终保留用户原始主题。
        builder.add_node("QueryExpansion", self._query_expansion)
        # 逐条调用受权限约束的 RetrievalTool，获取冻结知识库版本内的候选证据。
        builder.add_node("LiteratureRetrieval", self._literature_retrieval)
        # 按真实 chunk_id 去重、保留最高相关结果，并重新固定连续 Citation 编号。
        builder.add_node("EvidenceDedup", self._evidence_dedup)
        # 把 Chunk 级 Evidence 聚合成文献记录，保留真实元数据和全部引用入口。
        builder.add_node("MetadataNormalization", self._metadata_normalization)
        # 同一次模型调用读取完整 Evidence，同时生成综述分析草稿和主题演进图。
        builder.add_node("TopicClustering", self._topic_clustering)
        # 在撰写正文前一次校验模型生成的概念、分析分区和溯源图引用。
        builder.add_node("AnalysisValidation", self._analysis_validation)
        # 只重组前序已验证的结构化事实，生成带行内引用的文献综述正文和局限说明。
        builder.add_node("ReviewDraft", self._review_draft)
        # 检查完整综述正文的引用格式和编号范围，不重复审核结构化草稿。
        builder.add_node("ResearchReviewer", self._research_reviewer)
        # 两轮模型正文均不合格时，只用已验证事实确定性组织带引用的正文。
        builder.add_node("ReviewFallback", self._review_fallback)
        # 将正文及专业结论的编号映射回真实 Document、Chunk 和页码，组装最终结果。
        builder.add_node("CitationValidator", self._citation_validator)

        # 2. 先理解研究范围并固定 Evidence；后续节点只能消费这批不可扩张的证据
        builder.add_edge(START, "ConceptExtraction")
        builder.add_edge("ConceptExtraction", "QueryExpansion")
        builder.add_edge("QueryExpansion", "LiteratureRetrieval")
        builder.add_edge("LiteratureRetrieval", "EvidenceDedup")

        # 3. 将 Chunk 级证据聚合成文献级事实，保留真实题录和引用入口
        builder.add_edge("EvidenceDedup", "MetadataNormalization")

        # 4. 模型从同一批 Evidence 同时生成分析草稿和溯源图
        # 首轮分析校验失败重做一次生成，第二次失败则停止，绝不重新检索证据。
        builder.add_edge("MetadataNormalization", "TopicClustering")
        builder.add_edge("TopicClustering", "AnalysisValidation")
        builder.add_conditional_edges(
            "AnalysisValidation",
            self._route_analysis_validation,
            {"PASS": "ReviewDraft", "RETRY": "TopicClustering"},
        )

        # 5. 已验证分析先生成正文，再由 Reviewer 决定重写、降级或继续发布
        # Reviewer 最多退回正文生成一次；降级正文仍须经过同一个 Reviewer。
        builder.add_edge("ReviewDraft", "ResearchReviewer")
        builder.add_conditional_edges(
            "ResearchReviewer",
            self._route_research_reviewer,
            {
                "PASS": "CitationValidator",
                "RETRY": "ReviewDraft",
                "FALLBACK": "ReviewFallback",
            },
        )
        builder.add_edge("ReviewFallback", "ResearchReviewer")
        builder.add_edge("CitationValidator", END)
        return builder.compile()

    def _route_analysis_validation(self, state: SuzhiState) -> str:
        """根据分析校验结果选择继续或重做主题分析。

        Args:
            state: 包含本轮分析校验状态的图状态。

        Returns:
            PASS 或 RETRY 路由标识。
        """
        return state["analysis_validation_status"]

    def _route_research_reviewer(self, state: SuzhiState) -> str:
        """根据正文审核结果选择发布、重写或确定性降级。

        Args:
            state: 包含本轮正文审核状态的图状态。

        Returns:
            PASS、RETRY 或 FALLBACK 路由标识。
        """
        return state["reviewer_status"]

    async def run(self, state: SuzhiState) -> SuzhiState:
        """执行溯知图并返回包含结构化结果的最终状态。

        Args:
            state: 已冻结用户、研究范围、知识库版本和输出规格的初始状态。

        Returns:
            包含真实 Evidence、审核状态、综述正文和 SuzhiResult 的最终状态。
        """
        return await self.graph.ainvoke(state)

    async def _tracked(
        self, name: str, state: SuzhiState, operation: NodeOperation
    ) -> dict[str, Any]:
        """通过 Service 回调持久化节点检查点。

        Args:
            name: 与前端阶段和 AgentStep 审计记录一致的稳定节点名。
            state: 当前节点收到的溯知状态。
            operation: 不直接接触数据库的节点业务操作。

        Returns:
            当前节点产生并合并回 LangGraph 的状态增量。

        Raises:
            Exception: 节点操作失败；回调记录脱敏失败检查点后继续向上抛出。
        """
        return await self.step_runner(name, state, operation)

    async def _concept_extraction(self, state: SuzhiState) -> dict[str, Any]:
        """只从用户输入提取研究概念和同义检索词，不生成研究结论。

        Args:
            state: 包含冻结研究主题和可选用户指定概念的溯知状态。

        Returns:
            规范概念、SUZHI 执行路由和 REVIEW 意图组成的状态增量。
        """

        async def operation() -> dict[str, Any]:
            """调用冻结模型解析概念，并记录溯知执行路由。

            Returns:
                概念计划与统一运行路由字段。
            """
            # 模型仅做输入理解，同义词用于后续查询扩展，不能在此阶段生成文献事实。
            concept = await self.gateway.structured(
                instruction=(
                    "从研究任务中提取一个规范核心概念，并给出至多 8 个中英文同义或紧密相关"
                    "检索词。这里只理解输入，不陈述任何研究事实。根据研究主题的主要语言或"
                    "用户明确要求设置 output_language；只有明确要求中英双语时才选择 BILINGUAL。"
                    f"\n研究主题：{state['research_topic']}"
                    f"\n用户指定概念：{state.get('requested_concept') or '未指定'}"
                ),
                schema=ReviewConcept,
                system_prompt=self.system_prompt,
                thinking=False,
            )
            route = list(state.get("agent_route", []))
            # 启明委派时路由已经包含 QIMING；专业入口则从空路由开始。
            # 只在末尾缺失时追加，避免重入或恢复执行产生重复 SUZHI 节点。
            if not route or route[-1] != "SUZHI":
                route.append("SUZHI")
            return {
                "concept_plan": concept,
                "original_language": BilingualQueryProcessor()
                .process(state["research_topic"])
                .language,
                "output_language": concept.output_language,
                "agent_route": route,
                "intent": "REVIEW",
            }

        return await self._tracked("ConceptExtraction", state, operation)

    async def _query_expansion(self, state: SuzhiState) -> dict[str, Any]:
        """基于原主题和已解析概念生成有限的检索计划。

        Args:
            state: 包含研究主题和规范概念计划的溯知状态。

        Returns:
            保留原始主题且去重、最多五条查询的状态增量。
        """

        async def operation() -> dict[str, Any]:
            """生成去重且不超过五条的受控检索表达。

            Returns:
                可直接交给受控 RetrievalTool 的查询计划。
            """
            plan = await self.gateway.structured(
                instruction=(
                    "生成 2 至 5 条用于文献综述的中英文检索表达。每条都必须包含核心主题，"
                    "可分别覆盖源流、理论、方法或争议，但不得引入无关领域。"
                    f"\n主题：{state['research_topic']}"
                    f"\n概念：{state['concept_plan'].canonical_name}"
                    "\n候选词："
                    f"{json.dumps(state['concept_plan'].search_terms, ensure_ascii=False)}"
                ),
                schema=ReviewQueryPlan,
                system_prompt=self.system_prompt,
                thinking=False,
            )
            # 原始主题始终作为第一条查询，模型扩展只能补充覆盖面，不能替换用户意图。
            queries = list(dict.fromkeys([state["research_topic"], *plan.queries]))[:5]
            return {"query_plan": ReviewQueryPlan(queries=queries)}

        return await self._tracked("QueryExpansion", state, operation)

    async def _literature_retrieval(self, state: SuzhiState) -> dict[str, Any]:
        """逐条执行受控 Hybrid Retrieval，保留原始候选供下一节点去重。

        Args:
            state: 包含查询计划、时间范围和冻结知识库版本的溯知状态。

        Returns:
            各查询召回的原始 Evidence、Citation 及候选总数。

        Raises:
            ValueError: 任一检索响应的知识库版本与 Run 冻结版本不一致。
        """

        async def operation() -> dict[str, Any]:
            """执行全部计划查询并阻止知识库版本漂移。

            Returns:
                等待按真实 Chunk 去重的原始检索结果。

            Raises:
                ValueError: 检索期间知识库版本发生变化。
            """
            # 1. 将用户冻结的年份范围转换为 Retrieval 层过滤条件
            evidence: list[EvidenceRead] = []
            citations: list[CitationRead] = []
            filters = RetrievalFilters(
                year_from=state.get("year_from"), year_to=state.get("year_to")
            )

            # 2. 每条扩展查询仍通过同一个受控 Tool 执行权限和 READY 文档检查
            # 多条查询可能召回同一 Chunk，此处完整保留，交由下一节点按最高相关性去重。
            for query in state["query_plan"].queries:
                response = await self.retrieval.search(
                    knowledge_base_id=uuid.UUID(state["knowledge_base_id"]),
                    query=query,
                    mode="DEEP",
                    filters=filters,
                )
                # Run 必须始终基于创建时冻结的 KB Version；版本漂移时不能混合新旧证据。
                if response.knowledge_base_version != state["knowledge_base_version"]:
                    raise ValueError("知识库版本已变化")
                evidence.extend(response.evidence)
                citations.extend(response.citations)

            # 3. 候选总数保留重复召回，用于解释检索规模；最终证据数由去重节点确定
            return {
                "raw_evidence": evidence,
                "raw_citations": citations,
                "retrieved_candidates": len(evidence),
            }

        return await self._tracked("LiteratureRetrieval", state, operation)

    async def _evidence_dedup(self, state: SuzhiState) -> dict[str, Any]:
        """按真实 Chunk 去重并重新建立连续 Citation 编号。

        Args:
            state: 包含所有查询原始 Evidence 与 Citation 的溯知状态。

        Returns:
            最多三十条去重 Evidence 及一一对应的连续 Citation。

        Raises:
            ValueError: 去重后没有可用于综述的 Evidence。
        """

        async def operation() -> dict[str, Any]:
            """选择每个 Chunk 的最高相关结果并固定引用编号。

            Returns:
                可持久化并供所有后续模型节点引用的最终证据集合。

            Raises:
                ValueError: 当前知识库没有有效 Evidence。
            """
            # 1. 同一 Chunk 可能由多条查询重复召回，只保留相关性最高的一次结果
            best: dict[uuid.UUID, EvidenceRead] = {}
            citation_by_chunk = {item.chunk_id: item for item in state["raw_citations"]}
            for item in state["raw_evidence"]:
                current = best.get(item.chunk_id)
                if current is None or item.relevance_score > current.relevance_score:
                    best[item.chunk_id] = item

            # 2. 按最高相关性稳定截取综述证据预算，扩容不改变单次检索配置。
            ordered = sorted(best.values(), key=lambda item: item.relevance_score, reverse=True)[
                :MAX_REVIEW_EVIDENCE
            ]

            # 3. Evidence 与 Citation 使用相同的新编号，后续模型只能引用这一固定映射
            evidence: list[EvidenceRead] = []
            citations: list[CitationRead] = []
            for index, item in enumerate(ordered, start=1):
                evidence.append(item.model_copy(update={"citation_index": index}))
                citation = citation_by_chunk[item.chunk_id]
                citations.append(citation.model_copy(update={"citation_index": index}))
            if not evidence:
                raise ValueError("当前知识库没有足够 Evidence 支撑文献综述")
            return {"evidence": evidence, "citations": citations}

        return await self._tracked("EvidenceDedup", state, operation)

    async def _metadata_normalization(self, state: SuzhiState) -> dict[str, Any]:
        """把 Chunk 聚合为规范文献记录，保留每篇文献的全部引用入口。

        Args:
            state: 包含已固定 Citation 编号 Evidence 的溯知状态。

        Returns:
            按真实 Document 聚合的规范文献记录列表。
        """

        async def operation() -> dict[str, Any]:
            """按文献聚合真实元数据和全部引用入口。

            Returns:
                供主题分析与参考文献格式化使用的文献级记录。
            """
            documents: dict[uuid.UUID, dict[str, Any]] = {}
            for item in state["evidence"]:
                # 每篇文献只创建一条规范记录；多个 Chunk 的 Citation 编号全部保留，
                # 使后续参考文献仍能映射到实际使用的原文片段。
                row = documents.setdefault(
                    item.document_id,
                    {
                        "document_id": item.document_id,
                        "title": item.document_title.strip(),
                        "authors": item.authors,
                        "year": item.publication_year,
                        "journal": item.journal,
                        "language": item.language_code,
                        "citation_ids": [],
                    },
                )
                row["citation_ids"].append(item.citation_index)
            return {"documents": list(documents.values())}

        return await self._tracked("MetadataNormalization", state, operation)

    def _analysis_instruction(self, state: SuzhiState) -> str:
        """构造只包含本次真实 Evidence 的结构化分析指令。

        Args:
            state: 包含研究主题、输出语言和最终 Evidence 的溯知状态。

        Returns:
            限制模型事实来源和 Citation 规则的分析指令。
        """
        # 分析节点读取全部固定 Evidence，保证每个结构化结论都能回到真实 Chunk。
        payload = [item.model_dump(mode="json") for item in state["evidence"]]
        language = {"ZH": "中文", "EN": "English", "BILINGUAL": "中英双语"}[
            state["output_language"]
        ]
        return (
            f"只根据随后提供的已确认 Evidence 分析“{state['research_topic']}”，使用{language}。"
            "concept_definition 中每个事实结论必须在对应位置使用半角 [n]；"
            "concept_citation_ids 必须列出正文实际使用的全部引用编号。"
            "topics、theoretical_paths、methods、controversies、future_directions 每项的 title "
            "只写简短标题，不写 Citation；summary 中每个事实结论必须在对应位置使用半角 [n]，"
            "citation_ids 必须列出 summary 实际使用的全部引用编号。"
            "research_gaps 每项的 gap 只写研究不足的简短标题，不写 Citation；evidence_logic 中每个"
            "事实结论必须在对应位置使用半角 [n]，citation_ids 必须列出 evidence_logic 实际"
            "使用的全部引用编号。evidence_logic 要说明从哪些已有研究的范围、时间或方法分布"
            "归纳出研究不足，明确限于本次检索文献，不得把未检索到直接说成全球不存在。"
            "统一使用研究不足，不使用研究空白或填补空白等断言。"
            "所有 [n] 和 citation_ids 只能使用随后 Evidence 的 citation_id，二者编号集合必须"
            "一致；正文只使用 [n]，不得输出 citation_id 字段名，也不得用常识补全 Evidence "
            "未支持的结论。"
            "不得使用[n,m],要[n][m]。"
            "同时生成 provenance_graph。图中只保留能够代表发展阶段、技术路线、模型架构、"
            "理论范式或关键应用转折的里程碑节点；严禁把 Evidence 文献、作者、研究不足、"
            "争议、未来建议或一般研究主题分类直接画成节点。Evidence 仅通过 citation_ids "
            "支撑里程碑，不出现在主图。优先生成 8 至 18 个有明确继承、扩展、分叉、汇合"
            "或替代关系的关键节点，证据不足时宁可少画，不得用常识补全。演进方向必须从较早"
            "节点指向较晚节点；主干保持连贯，只有存在不同技术路线时才分叉。stage 从 0 开始"
            "表示从左到右的拓扑层级，同一阶段的并行路线使用相同 stage；lane=0 表示主线，"
            "上方分支使用负数，下方分支使用正数。label 只写技术或阶段名称，subtitle 用一行"
            "概括机制或定位，year 没有 Evidence 支撑时必须为 null。节点 id 必须唯一，edge "
            "的 source/target 必须引用已有节点 id；边关系只使用 EVOLVES_TO、BRANCHES_TO、"
            "ENABLES、EXTENDS、MERGES_INTO 或 REPLACES。summary 和 citation_ids 说明 Evidence "
            "如何支撑里程碑或演进关系；citation_ids 只能使用随后 Evidence 的 citation_id。"
            "当前版本不做图关系语义复核或自动补边。"
            f"\n已确认 Evidence：{json.dumps(payload, ensure_ascii=False)}"
        )

    async def _topic_clustering(self, state: SuzhiState) -> dict[str, Any]:
        """基于一次完整 Evidence 输入共同生成分析草稿和溯源图。

        Args:
            state: 包含研究主题、输出语言和最终 Evidence 的溯知状态。

        Returns:
            结构化分析草稿与同源的主题演进图。
        """

        async def operation() -> dict[str, Any]:
            """从受控 Evidence 同时生成综述分析和主题演进图。

            Returns:
                等待后续节点校验引用的分析草稿和溯源图。
            """
            attempt = state.get("analysis_attempts", 0) + 1
            if attempt > MAX_ANALYSIS_ATTEMPTS:
                raise ValueError("分析生成次数超过上限")
            correction = state.get("analysis_validation_error", "")
            instruction = self._analysis_instruction(state)
            if correction:
                # 只反馈确定性校验原因；重用冻结 Evidence，不把上轮草稿重复上传。
                instruction += f"\n上轮校验失败：{correction}。请重新生成并修正引用。"
            output = await self.gateway.structured(
                instruction=instruction,
                schema=LiteratureAnalysisWithGraph,
                system_prompt=self.system_prompt,
                thinking=False,
            )
            return {
                "analysis_draft": output.analysis_draft,
                "provenance_graph": output.provenance_graph,
                "analysis_attempts": attempt,
            }

        return await self._tracked("TopicClustering", state, operation)

    async def _analysis_validation(self, state: SuzhiState) -> dict[str, Any]:
        """在正文生成前统一校验分析草稿和溯源图的 Citation。

        Args:
            state: 包含 TopicClustering 输出和本次固定 Evidence 的溯知状态。

        Returns:
            各分区数量及 PASS 或 RETRY 决策。

        Raises:
            ValueError: 第二轮仍有缺失、错位或越界引用。
        """

        async def operation() -> dict[str, Any]:
            """检查同次模型输出的全部结构化引用边界。

            Returns:
                已校验数量或触发重新生成的失败原因。

            Raises:
                ValueError: 第二轮结构化引用仍不满足本次 Evidence 约束。
            """
            # 1. 概念定义和各分析分区都必须在事实原位标出与数组一致的编号。
            allowed = {item.citation_index for item in state["evidence"]}
            draft = state["analysis_draft"]
            concept_ids = set(draft.concept_citation_ids)
            concept_inline_ids = _review_citation_ids(draft.concept_definition)
            failure = ""
            if not concept_ids <= allowed or concept_inline_ids != concept_ids:
                failure = "概念定义包含缺失、错位或越界 Citation"

            counts: dict[str, int] = {}
            for field_name in (
                "topics",
                "theoretical_paths",
                "methods",
                "controversies",
                "future_directions",
            ):
                items = getattr(draft, field_name)
                for item in items:
                    citations = set(item.citation_ids)
                    if not citations <= allowed or _review_citation_ids(item.summary) != citations:
                        failure = failure or f"{field_name} 包含缺失、错位或越界 Citation"
                counts[f"{field_name}_count"] = len(items)

            # 2. 研究不足的归纳文字也属于模型生成的事实，必须保留原有证据逻辑检查。
            for gap in draft.research_gaps:
                citations = set(gap.citation_ids)
                if (
                    not citations <= allowed
                    or _review_citation_ids(gap.evidence_logic) != citations
                    or not gap.evidence_logic.strip()
                ):
                    failure = failure or "研究不足缺少有效 Evidence 或证据逻辑"
            counts["gap_count"] = len(draft.research_gaps)

            # 3. 溯源图只做既有的 Citation 范围检查，不把模型关系当成已验证事实。
            graph = state["provenance_graph"]
            for item in (*graph.nodes, *graph.edges):
                if not set(item.citation_ids) <= allowed:
                    failure = failure or "溯源图包含越界 Citation"
            counts["graph_node_count"] = len(graph.nodes)
            counts["graph_edge_count"] = len(graph.edges)
            # 4. 首轮失败作为可审计的路由结果；第二轮失败由 Step 记录为 FAILED。
            if failure:
                if state.get("analysis_attempts", 0) >= MAX_ANALYSIS_ATTEMPTS:
                    raise ValueError(failure)
                return {
                    "analysis_validation_status": "RETRY",
                    "analysis_validation_error": failure,
                }
            return {
                **counts,
                "analysis_validation_status": "PASS",
                "analysis_validation_error": "",
            }

        return await self._tracked("AnalysisValidation", state, operation)

    async def _review_draft(self, state: SuzhiState) -> dict[str, Any]:
        """使用已审核结构化事实生成指定语言和长度的完整综述草稿。

        Args:
            state: 包含已审核分析事实、语言和长度规格的溯知状态。

        Returns:
            本轮已规范化的综述草稿及业务尝试次数。
        """

        async def operation() -> dict[str, Any]:
            """从已审核事实生成一轮综述正文，交给 Reviewer 决定是否重写。

            Returns:
                使用半角 ``[n]`` Citation 的 ReviewDraft。
            """
            # 1. 记录业务轮次并将冻结规格转换为模型可理解的要求
            attempt = state.get("review_attempts", 0) + 1
            if attempt > MAX_REVIEW_ATTEMPTS:
                raise ValueError("综述生成次数超过上限")
            target = {"SHORT": "约1000字", "MEDIUM": "2000至3000字", "LONG": "5000字以上"}[
                state["review_length"]
            ]
            language = {"ZH": "中文", "EN": "英文", "BILINGUAL": "中英双语"}[
                state["output_language"]
            ]
            # 溯源图保存在独立状态字段中，不送入正文生成，避免 ReviewDraft 把图节点
            # 说明误写成新的综述事实。
            facts = state["analysis_draft"].model_dump(mode="json")

            # 2. 模型只重组已审核事实；重试时附上前轮确定性失败原因
            correction = state.get("reviewer_error", "")
            retry_instruction = (
                f"\n上轮审核失败：{correction}。请重新撰写并修正引用。" if correction else ""
            )
            draft = await self.gateway.structured(
                instruction=(
                    "【review 字段排版】review 是分章节、分段落的 Markdown 正文，"
                    "正文须有章节和段落换行，不能把整篇综述写成一行。"
                    "不要重复研究主题总标题。根据下方有证据的事实"
                    "组织多个 ## 章节；需要细分时使用 ### 小节。每个标题独占一行，"
                    "标题与正文之间、相邻自然段之间各留一个空行。每个自然段聚焦一个"
                    "研究要点，不能把多个章节的内容连成一个长段落。标题不要附带引用，"
                    "不能把标题标记与正文写在同一行。\n\n"
                    "【版式示意】以下只展示 review 正文的排版，示意文字不得照抄：\n"
                    "## 概念界定\n\n第一段正文。\n\n第二段正文。\n\n"
                    "## 研究主题\n\n第三段正文。\n\n"
                    f"【写作依据】使用{language}撰写{target}的学术文献综述，"
                    "只重组提供的结构化事实。每个事实句都必须在句末保留本次真实 [n] 引用；"
                    "多个引用写成 [n][m]，不能写成 [n,m]。统一使用研究不足，"
                    "不使用研究空白或填补空白等断言。研究不足必须明确标为"
                    "基于当前证据的推断。不要输出 HTML、Markdown 代码围栏或目录。"
                    f"\n\n事实：{json.dumps(facts, ensure_ascii=False)}"
                    f"{retry_instruction}"
                ),
                schema=ReviewDraft,
                system_prompt=self.system_prompt,
                thinking=False,
            )
            # 3. 只做格式规范化，合法性统一由下一节点审核并记录路由决定
            return {
                "review_draft": draft.model_copy(
                    update={"review": _normalize_review_citations(draft.review)}
                ),
                "review_attempts": attempt,
                "review_fallback_used": False,
            }

        return await self._tracked("ReviewDraft", state, operation)

    async def _research_reviewer(self, state: SuzhiState) -> dict[str, Any]:
        """在发布前检查完整综述正文的 Citation 格式与编号范围。

        Args:
            state: 包含已通过分析校验的正文草稿和最终 Evidence 的溯知状态。

        Returns:
            PASS、RETRY 或 FALLBACK 决策。

        Raises:
            ValueError: 确定性降级正文仍不满足 Citation 约束。
        """

        async def operation() -> dict[str, Any]:
            """确认正文只使用本次固定 Evidence 的公开引用编号。

            Returns:
                正文审核状态和失败原因。

            Raises:
                ValueError: 降级正文仍不满足 Citation 发布约束。
            """
            allowed = {item.citation_index for item in state["evidence"]}
            review = state["review_draft"].review
            failure = ""
            if re.search(r"citation_id\s*:", review, flags=re.IGNORECASE):
                failure = "完整综述包含未规范化的内部 Citation 标记"
            review_ids = _review_citation_ids(review)
            if not failure and (not review_ids or not review_ids <= allowed):
                failure = "完整综述缺少合法 Citation"
            if not failure:
                return {"reviewer_status": "PASS", "reviewer_error": ""}
            # 模型最多重写一次；连续失败时改用已校验事实，降级结果仍需再审。
            if state.get("review_fallback_used"):
                raise ValueError(failure)
            status = (
                "RETRY"
                if state.get("review_attempts", 0) < MAX_REVIEW_ATTEMPTS
                else "FALLBACK"
            )
            return {"reviewer_status": status, "reviewer_error": failure}

        return await self._tracked("ResearchReviewer", state, operation)

    async def _review_fallback(self, state: SuzhiState) -> dict[str, Any]:
        """两轮模型正文失败后从已审核事实确定性组织综述。

        Args:
            state: 包含已验证分析事实和输出语言的溯知状态。

        Returns:
            等待同一 Reviewer 复核的降级正文。
        """

        async def operation() -> dict[str, Any]:
            """保留已审核事实与原始 Citation 对应关系，构造降级正文。

            Returns:
                带降级标记的完整综述草稿。
            """
            review = _build_grounded_review(
                state["analysis_draft"], state["output_language"]
            )
            return {
                "review_draft": ReviewDraft(review=review),
                "review_fallback_used": True,
            }

        return await self._tracked("ReviewFallback", state, operation)

    async def _citation_validator(self, state: SuzhiState) -> dict[str, Any]:
        """映射最终所有引用到真实 Document/Chunk/Page 并组装结果。

        Args:
            state: 已通过 Reviewer 且包含全部专业派生结果的溯知状态。

        Returns:
            真实 Citation、参考文献、Evidence 摘要和可发布 SuzhiResult。

        Raises:
            ValueError: 任一使用中的 Citation 无法映射到当前 Run 的真实文献分块。
        """

        async def operation() -> dict[str, Any]:
            """完成真实来源映射、参考文献格式化和最终结果组装。

            Returns:
                通过最终发布闸门的状态增量。

            Raises:
                ValueError: 使用中的 Citation 与真实来源集合不一致。
            """
            # 1. 汇总正文和结构化结论实际使用的 Citation 编号
            review_ids = _review_citation_ids(state["review_draft"].review)
            used = review_ids | _structured_citation_ids(
                state["analysis_draft"], state["provenance_graph"]
            )

            # 2. 将编号严格投影回 EvidenceDedup 固定的真实 Document/Chunk/Page 引用
            citations = [item for item in state["citations"] if item.citation_index in used]
            if {item.citation_index for item in citations} != used:
                raise ValueError("Citation 无法映射到真实文献分块")

            # 3. 只为实际使用的文献生成指定格式参考文献，并计算可解释覆盖摘要
            references = [
                _format_reference(item, state["citation_style"])
                for item in state["documents"]
                if item["citation_ids"][0] in used
            ]
            years = sorted({item["year"] for item in state["documents"] if item["year"]})
            languages = Counter(item["language"] for item in state["documents"])
            summary = {
                "evidence_count": len(state["evidence"]),
                "document_count": len(state["documents"]),
                "year_range": [years[0], years[-1]] if years else [],
                "language_coverage": dict(languages),
            }

            # 4. 组装唯一专业结果；最终回答直接采用已审核正文，不再调用模型改写
            analysis = state["analysis_draft"]
            limitations: list[str] = []
            if state.get("review_fallback_used"):
                limitations.append(
                    "模型正文连续未满足 Citation 格式，已使用审核通过的结构化事实确定性组织综述。"
                )
            result = SuzhiResult(
                status="SUCCESS",
                research_topic=state["research_topic"],
                concept=state["concept_plan"].canonical_name,
                language=state["output_language"],
                review_length=state["review_length"],
                citation_style=state["citation_style"],
                concept_definition=analysis.concept_definition,
                concept_citation_ids=analysis.concept_citation_ids,
                topics=analysis.topics,
                theoretical_paths=analysis.theoretical_paths,
                methods=analysis.methods,
                controversies=analysis.controversies,
                research_gaps=analysis.research_gaps,
                future_directions=analysis.future_directions,
                provenance_graph=state["provenance_graph"],
                review=state["review_draft"].review,
                references=references,
                evidence_summary=summary,
                limitations=limitations,
                evidence_level="HIGH" if len(state["documents"]) >= 5 else "MEDIUM",
            )
            return {
                "citation_validation_status": "PASS",
                "validated_citations": citations,
                "used_citation_indices": sorted(used),
                "references": references,
                "evidence_summary": summary,
                "evidence_level": result.evidence_level,
                "suzhi_result": result,
                "final_answer": result.review,
            }

        return await self._tracked("CitationValidator", state, operation)


def _structured_citation_ids(
    draft: LiteratureAnalysisDraft, provenance_graph: ProvenanceGraph
) -> set[int]:
    """收集结构化分析草稿和溯源图中的全部 Citation 编号。

    Args:
        draft: 已通过 Pydantic 结构校验的溯知分析草稿。
        provenance_graph: 与分析草稿同次生成的溯源图。

    Returns:
        定义、所有分析分区及溯源图节点和关系引用编号的去重集合。
    """
    used = set(draft.concept_citation_ids)
    for field_name in (
        "topics",
        "theoretical_paths",
        "methods",
        "controversies",
        "research_gaps",
        "future_directions",
    ):
        for item in getattr(draft, field_name):
            used.update(item.citation_ids)
    for node in provenance_graph.nodes:
        used.update(node.citation_ids)
    for edge in provenance_graph.edges:
        used.update(edge.citation_ids)
    return used


def _build_grounded_review(draft: LiteratureAnalysisDraft, language: str) -> str:
    """把已审核结构化事实确定性组织为带真实 Citation 的综述。

    该函数仅在两轮模型正文都违反 Citation 契约时使用。它不会新增研究事实，
    每个段落的编号直接来自对应结构化字段已经通过审核的 ``citation_ids``。

    Args:
        draft: 已完成各分区 Citation 范围审核的结构化分析草稿。
        language: ZH、EN 或 BILINGUAL 输出语言。

    Returns:
        每个事实段落都带原始 Citation 编号的 Markdown 综述。
    """
    headings = {
        "ZH": (
            "概念界定",
            "研究主题",
            "理论路径",
            "研究方法",
            "主要争议",
            "研究不足",
            "未来方向",
        ),
        "EN": (
            "Concept definition",
            "Research topics",
            "Theoretical paths",
            "Research methods",
            "Controversies",
            "Research limitations",
            "Future directions",
        ),
        "BILINGUAL": (
            "概念界定 / Concept definition",
            "研究主题 / Research topics",
            "理论路径 / Theoretical paths",
            "研究方法 / Research methods",
            "主要争议 / Controversies",
            "研究不足 / Research limitations",
            "未来方向 / Future directions",
        ),
    }[language]
    sections: list[str] = [
        f"## {headings[0]}",
        _fact_with_citations(draft.concept_definition, draft.concept_citation_ids),
    ]
    fields = (
        (headings[1], draft.topics),
        (headings[2], draft.theoretical_paths),
        (headings[3], draft.methods),
        (headings[4], draft.controversies),
    )
    for heading, findings in fields:
        if not findings:
            continue
        sections.append(f"## {heading}")
        sections.extend(
            f"### {item.title}\n\n{_fact_with_citations(item.summary, item.citation_ids)}"
            for item in findings
        )
    if draft.research_gaps:
        sections.append(f"## {headings[5]}")
        sections.extend(
            (f"### {item.gap}\n\n{_fact_with_citations(item.evidence_logic, item.citation_ids)}")
            for item in draft.research_gaps
        )
    if draft.future_directions:
        sections.append(f"## {headings[6]}")
        sections.extend(
            f"### {item.title}\n\n{_fact_with_citations(item.summary, item.citation_ids)}"
            for item in draft.future_directions
        )
    return "\n\n".join(sections)


def _fact_with_citations(text: str, citation_ids: list[int]) -> str:
    """保留事实文本中的引用位置，并限制其只能使用已审核 Citation。

    Args:
        text: 结构化分析中的事实或证据逻辑。
        citation_ids: 与该文本一起通过分区审核的引用编号。

    Returns:
        原位引用已规范化且只包含已审核编号的事实段落。
    """
    normalized = _normalize_review_citations(text)
    allowed = set(citation_ids)

    # 结构化事实偶尔已经包含准确的句级引用，此时保留原位置，不能为了补齐
    # citation_ids 而把全部编号再次汇总到段末。
    with_valid_markers = re.sub(
        r"\[(\d+)]",
        lambda match: match.group(0) if int(match.group(1)) in allowed else "",
        normalized,
    ).strip()
    if _review_citation_ids(with_valid_markers):
        return with_valid_markers

    # 文本完全没有句级引用时才附加该结构化事实自身的已审核编号，保证确定性
    # 降级仍然 Evidence First，而不是猜测 Citation 应插入哪个句子。
    markers = "".join(f"[{citation_id}]" for citation_id in dict.fromkeys(citation_ids))
    return f"{with_valid_markers} {markers}".strip()


def _normalize_review_citations(review: str) -> str:
    """把模型 Citation 原位规范为系统唯一的 ``[n]`` 形式。

    Args:
        review: 模型生成的原始综述正文。

    Returns:
        内部标记和全角方括号已原位转换、重复段末汇总已清除的正文。
    """
    normalized = re.sub(r"【\s*(\d+)\s*】", r"[\1]", review)
    normalized = re.sub(r"［\s*(\d+)\s*］", r"[\1]", normalized)

    def replace_internal_markers(match: re.Match[str]) -> str:
        """把一个内部 Citation 列表按原位置转换为连续标准编号。

        Args:
            match: 匹配到的内部 Citation 标记及其编号列表。

        Returns:
            保持原位置的连续 ``[n]`` 引用文本。
        """
        indices = re.findall(r"citation_id\s*:\s*(\d+)", match.group(0), flags=re.IGNORECASE)
        return "".join(f"[{index}]" for index in indices)

    normalized = re.sub(
        r"\[\s*citation_id\s*:\s*\d+"
        r"(?:\s*,\s*citation_id\s*:\s*\d+)*\s*]",
        replace_internal_markers,
        normalized,
        flags=re.IGNORECASE,
    )

    # 旧降级正文形如“事实[1]。 [1][2]”：段末编号与同段正文引用重叠时，
    # 可确定它是旧版本追加的汇总并整组删除；普通句末 Citation 不受影响。
    paragraphs = []
    for paragraph in normalized.split("\n\n"):
        trailing = re.search(r"([。！？.!?])\s+((?:\[\d+])+)$", paragraph)
        if trailing:
            prior = paragraph[: trailing.start(2)]
            prior_ids = _review_citation_ids(prior)
            trailing_ids = _review_citation_ids(trailing.group(2))
            if trailing_ids & prior_ids:
                paragraph = paragraph[: trailing.start(2)].rstrip()
        paragraphs.append(paragraph)
    return "\n\n".join(paragraphs)


def _review_citation_ids(review: str) -> set[int]:
    """提取规范化综述正文中的去重 Citation 编号。

    Args:
        review: 使用半角 ``[n]`` 格式的综述正文。

    Returns:
        正文实际使用的 Citation 编号去重集合。
    """
    return {int(value) for value in re.findall(r"\[(\d+)]", review)}


def _format_reference(document: dict[str, Any], style: str) -> str:
    """按用户选择的 GB/T 7714、APA 或 MLA 输出知识库元数据引用。

    Args:
        document: 从真实 Evidence 聚合出的文献级元数据。
        style: 用户冻结的 APA、MLA 或 GBT7714 引用格式编码。

    Returns:
        不补造缺失作者、年份或来源信息的参考文献文本。
    """
    names = [str(item.get("name") or "").strip() for item in document["authors"]]
    authors = ", ".join(name for name in names if name) or "佚名"
    year = str(document["year"] or "n.d.")
    title = document["title"]
    journal = document["journal"] or "来源未注明"
    if style == "APA":
        return f"{authors}. ({year}). {title}. {journal}."
    if style == "MLA":
        return f'{authors}. "{title}." {journal}, {year}.'
    return f"{authors}. {title}[J]. {journal}, {year}."
