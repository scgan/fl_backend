"""实现Quill THEORY 阶段（理论依据与文献推荐）的 LangGraph。

图固定执行概念解析、跨语言检索、去重、文献整理、理论综合及引用校验。
模型只处理受控 Evidence，不能访问数据库；理论要点和推荐理由必须原位
使用真实 [n] Citation，参考文献由确定性代码生成。
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
from app.agents.quill_schemas import (
    QuillResult,
    TheoryAnalysisDraft,
    TheoryConcept,
    TheoryQueryPlan,
)
from app.agents.quill_state import QuillState
from app.agents.tools import RetrievalTool
from app.retrieval.query_processing import BilingualQueryProcessor
from app.retrieval.schemas import CitationRead, EvidenceRead, RetrievalFilters

NodeOperation = Callable[[], Awaitable[dict[str, Any]]]
StepRunner = Callable[[str, QuillState, NodeOperation], Awaitable[dict[str, Any]]]

# 理论综合每段最多生成两轮；网关内部的传输或 Schema 重试另行计数。
MAX_ANALYSIS_ATTEMPTS = 2
# EvidenceDedup 固定的 DEEP 模式证据上限，与溯知保持一致。
MAX_EVIDENCE = 20


class QuillGraph:
    """执行研究主题输入到可追溯 THEORY 结果的固定工作流。"""

    # 子类（RQ/PLAN 图）复用检索管线时覆盖：运行意图编码与检索目的措辞。
    _stage_intent = "THEORY"
    _retrieval_purpose = "理论依据与文献推荐"

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
            system_prompt: Run 创建时冻结的Quill Prompt 正文。
            step_runner: 在每个节点前后持久化 AgentStep 的运行回调。
        """
        # 检索 Tool 统一执行知识库权限、READY 文档白名单和 Evidence 质量门槛。
        self.retrieval = retrieval
        # 模型网关只负责受控的概念理解、理论综合草稿，不直接访问数据源。
        self.gateway = gateway
        # Prompt 从 Run 固定版本读取，运行期间不得切换到当前激活版本。
        self.system_prompt = system_prompt
        # 步骤回调负责节点可观察状态、Evidence 检查点和脱敏失败信息。
        self.step_runner = step_runner
        # 图在构造期编译一次，单次运行的可变数据全部通过 QuillState 传递。
        self.graph = self._build()

    def _build(self):
        """编译固定 Evidence、两段有界回退的Quill THEORY 图。

        Returns:
            可异步调用并以稳定节点名记录步骤的已编译 LangGraph。
        """
        # 1. 注册从概念理解到引用校验的全部业务节点
        # 节点名称同时用于 SSE 进度和 AgentStep 审计；模型只能处理节点收到的受控
        # Evidence，不能改变拓扑、跳过证据整理阶段或直接构造最终引用。
        builder = StateGraph(QuillState)
        # 从用户主题中提取规范概念、输出语言和有限的中英文相关词，不生成研究事实。
        builder.add_node("ConceptExtraction", self._concept_extraction)
        # 将原始主题与规范概念扩展为至多五条检索表达，且始终保留用户原始主题。
        builder.add_node("QueryExpansion", self._query_expansion)
        # 逐条调用受权限约束的 RetrievalTool，获取冻结知识库版本内的候选证据。
        builder.add_node("LiteratureRetrieval", self._literature_retrieval)
        # 按真实 chunk_id 去重、保留最高相关结果，并重新固定连续 Citation 编号。
        builder.add_node("EvidenceDedup", self._evidence_dedup)
        # 把 Chunk 级 Evidence 聚合成文献记录，保留真实元数据和全部引用入口。
        builder.add_node("MetadataNormalization", self._metadata_normalization)
        # 同一次模型调用读取完整 Evidence，生成理论依据、文献推荐与研究空白草稿。
        builder.add_node("TheorySynthesis", self._theory_synthesis)
        # 在组装结果前一次校验模型生成的理论、推荐和空白的引用。
        builder.add_node("AnalysisValidation", self._analysis_validation)
        # 将结论编号映射回真实 Document、Chunk 和页码，组装最终结果。
        builder.add_node("CitationValidator", self._citation_validator)

        # 2. 先理解研究范围并固定 Evidence；后续节点只能消费这批不可扩张的证据
        builder.add_edge(START, "ConceptExtraction")
        builder.add_edge("ConceptExtraction", "QueryExpansion")
        builder.add_edge("QueryExpansion", "LiteratureRetrieval")
        builder.add_edge("LiteratureRetrieval", "EvidenceDedup")

        # 3. 将 Chunk 级证据聚合成文献级事实，保留真实题录和引用入口
        builder.add_edge("EvidenceDedup", "MetadataNormalization")

        # 4. 理论综合与校验构成有界回路：首轮失败重做一次生成，
        # 第二次失败则停止，绝不重新检索证据。
        builder.add_edge("MetadataNormalization", "TheorySynthesis")
        builder.add_edge("TheorySynthesis", "AnalysisValidation")
        builder.add_conditional_edges(
            "AnalysisValidation",
            self._route_analysis_validation,
            {"PASS": "CitationValidator", "RETRY": "TheorySynthesis"},
        )
        builder.add_edge("CitationValidator", END)
        return builder.compile()

    def _route_analysis_validation(self, state: QuillState) -> str:
        """根据分析校验结果选择组装结果或重做理论综合。

        Args:
            state: 包含本轮分析校验状态的图状态。

        Returns:
            PASS 或 RETRY 路由标识。
        """
        return state["analysis_validation_status"]

    async def run(self, state: QuillState) -> QuillState:
        """执行Quill图并返回包含结构化结果的最终状态。

        Args:
            state: 已冻结用户、工作阶段、知识库版本和输出规格的初始状态。

        Returns:
            包含真实 Evidence、审核状态和 QuillResult 的最终状态。
        """
        return await self.graph.ainvoke(state)

    async def _tracked(
        self, name: str, state: QuillState, operation: NodeOperation
    ) -> dict[str, Any]:
        """通过 Service 回调持久化节点检查点。

        Args:
            name: 与前端阶段和 AgentStep 审计记录一致的稳定节点名。
            state: 当前节点收到的Quill状态。
            operation: 不直接接触数据库的节点业务操作。

        Returns:
            当前节点产生并合并回 LangGraph 的状态增量。

        Raises:
            Exception: 节点操作失败；回调记录脱敏失败检查点后继续向上抛出。
        """
        return await self.step_runner(name, state, operation)

    async def _concept_extraction(self, state: QuillState) -> dict[str, Any]:
        """只从用户输入提取研究概念和同义检索词，不生成研究结论。

        Args:
            state: 包含冻结研究主题的Quill状态。

        Returns:
            规范概念、QUILL 执行路由和 THEORY 意图组成的状态增量。
        """

        async def operation() -> dict[str, Any]:
            """调用冻结模型解析概念，并记录Quill执行路由。

            Returns:
                概念计划与统一运行路由字段。
            """
            # 模型仅做输入理解，同义词用于后续查询扩展，不能在此阶段生成研究事实。
            concept = await self.gateway.structured(
                instruction=(
                    "从研究主题中提取一个规范核心概念，并给出至多 8 个中英文同义或紧密相关"
                    "检索词。这里只理解输入，不陈述任何研究事实。根据研究主题的主要语言或"
                    "用户明确要求设置 output_language；只有明确要求中英双语时才选择 BILINGUAL。"
                    f"\n研究主题：{state['research_topic']}"
                ),
                schema=TheoryConcept,
                system_prompt=self.system_prompt,
                thinking=bool(state.get("thinking")),
            )
            route = list(state.get("agent_route", []))
            # 专业入口从空路由开始；只在末尾缺失时追加，避免重复 QUILL 节点。
            if not route or route[-1] != "QUILL":
                route.append("QUILL")
            return {
                "concept_plan": concept,
                "original_language": BilingualQueryProcessor()
                .process(state["research_topic"])
                .language,
                "output_language": concept.output_language,
                "agent_route": route,
                "intent": self._stage_intent,
            }

        return await self._tracked("ConceptExtraction", state, operation)

    async def _query_expansion(self, state: QuillState) -> dict[str, Any]:
        """基于原主题和已解析概念生成有限的检索计划。

        Args:
            state: 包含研究主题和规范概念计划的Quill状态。

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
                    f"生成 2 至 5 条用于{self._retrieval_purpose}的检索表达。每条都必须包含核心"
                    "主题，可分别覆盖理论基础、研究方法或相关文献，但不得引入无关领域。"
                    f"\n主题：{state['research_topic']}"
                    f"\n概念：{state['concept_plan'].canonical_name}"
                    "\n候选词："
                    f"{json.dumps(state['concept_plan'].search_terms, ensure_ascii=False)}"
                ),
                schema=TheoryQueryPlan,
                system_prompt=self.system_prompt,
                thinking=bool(state.get("thinking")),
            )
            # 原始主题始终作为第一条查询，模型扩展只能补充覆盖面，不能替换用户意图。
            queries = list(dict.fromkeys([state["research_topic"], *plan.queries]))[:5]
            return {"query_plan": TheoryQueryPlan(queries=queries)}

        return await self._tracked("QueryExpansion", state, operation)

    async def _literature_retrieval(self, state: QuillState) -> dict[str, Any]:
        """逐条执行受控 Hybrid Retrieval，保留原始候选供下一节点去重。

        Args:
            state: 包含查询计划、时间范围和冻结知识库版本的Quill状态。

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
                    mode=str(state.get("mode") or "DEEP"),
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

    async def _evidence_dedup(self, state: QuillState) -> dict[str, Any]:
        """按真实 Chunk 去重并重新建立连续 Citation 编号。

        Args:
            state: 包含所有查询原始 Evidence 与 Citation 的Quill状态。

        Returns:
            最多二十条去重 Evidence 及一一对应的连续 Citation。

        Raises:
            ValueError: 去重后没有可用于理论分析的 Evidence。
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

            # 2. 按最高相关性稳定截取 DEEP 模式的二十条证据上限
            ordered = sorted(best.values(), key=lambda item: item.relevance_score, reverse=True)[
                :MAX_EVIDENCE
            ]

            # 3. Evidence 与 Citation 使用相同的新编号，后续模型只能引用这一固定映射
            evidence: list[EvidenceRead] = []
            citations: list[CitationRead] = []
            for index, item in enumerate(ordered, start=1):
                evidence.append(item.model_copy(update={"citation_index": index}))
                citation = citation_by_chunk[item.chunk_id]
                citations.append(citation.model_copy(update={"citation_index": index}))
            if not evidence:
                raise ValueError("当前知识库没有足够 Evidence 支撑本阶段分析")
            return {"evidence": evidence, "citations": citations}

        return await self._tracked("EvidenceDedup", state, operation)

    async def _metadata_normalization(self, state: QuillState) -> dict[str, Any]:
        """把 Chunk 聚合为规范文献记录，保留每篇文献的全部引用入口。

        Args:
            state: 包含已固定 Citation 编号 Evidence 的Quill状态。

        Returns:
            按真实 Document 聚合的规范文献记录列表。
        """

        async def operation() -> dict[str, Any]:
            """按文献聚合真实元数据和全部引用入口。

            Returns:
                供理论综合与参考文献格式化使用的文献级记录。
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

    def _writing_requirements_instruction(self, state: QuillState) -> str:
        """将独立的写作规格作为用户数据提供给各阶段，不能覆盖阶段及引用边界。"""
        requirements = state.get("writing_requirements", "").strip()
        if not requirements:
            return ""
        return (
            "\n用户写作要求（仅调整本阶段成果的篇幅、层级和侧重，不能改变任务或证据规则；"
            "与默认篇幅冲突时按用户要求，在本阶段 Schema 支持的范围内执行）："
            + json.dumps(requirements, ensure_ascii=False)
            + "\n"
        )

    def _analysis_instruction(self, state: QuillState) -> str:
        """构造只包含本次真实 Evidence 的理论综合指令。

        Args:
            state: 包含研究主题、输出语言、推荐上限和最终 Evidence 的Quill状态。

        Returns:
            限制模型事实来源和 Citation 规则的理论综合指令。
        """
        # 理论综合节点读取全部固定 Evidence，保证每个结构化结论都能回到真实 Chunk。
        payload = [item.model_dump(mode="json") for item in state["evidence"]]
        language = {"ZH": "中文", "EN": "English", "BILINGUAL": "中英双语"}[
            state["output_language"]
        ]
        return (
            f"只根据随后提供的已确认 Evidence 为“{state['research_topic']}”整理理论依据"
            f"与文献推荐，使用{language}。"
            "theoretical_basis 每项的 theory 只写理论、框架或概念视角的简短名称，不写 "
            "Citation；description 中每个事实结论必须在对应位置使用半角 [n]，"
            "citation_ids 必须列出 description 实际使用的全部引用编号。"
            f"literature_recommendations 不得超过 {state['literature_limit']} 条；每项的 "
            "document_title 必须原样使用 Evidence 对应文献的 document_title，不得推荐"
            "知识库之外的文献；relevance_reason 中每个事实结论必须在对应位置使用半角 "
            "[n]，citation_ids 必须列出 relevance_reason 实际使用的全部引用编号。"
            "research_gaps 每项的 gap 只写空白结论标题，不写 Citation；evidence_logic 中"
            "每个事实结论必须在对应位置使用半角 [n]，citation_ids 必须列出 evidence_logic "
            "实际使用的全部引用编号。evidence_logic 要说明从哪些已有研究的范围、时间或方法"
            "分布推导出空白，不得把未检索到直接说成全球不存在。"
            "所有 [n] 和 citation_ids 只能使用随后 Evidence 的 citation_id，二者编号集合必须"
            "一致；正文只使用 [n]，不得输出 citation_id 字段名，也不得用常识补全 Evidence "
            "未支持的结论。不得使用[n,m]，要[n][m]。"
            + self._writing_requirements_instruction(state)
            + f"\n已确认 Evidence：{json.dumps(payload, ensure_ascii=False)}"
        )

    async def _theory_synthesis(self, state: QuillState) -> dict[str, Any]:
        """基于一次完整 Evidence 输入生成理论依据、文献推荐与研究空白草稿。

        Args:
            state: 包含研究主题、输出语言和最终 Evidence 的Quill状态。

        Returns:
            等待后续节点校验引用的结构化分析草稿。
        """

        async def operation() -> dict[str, Any]:
            """从受控 Evidence 生成理论综合草稿。

            Returns:
                等待校验的分析草稿及本轮业务尝试次数。
            """
            attempt = state.get("analysis_attempts", 0) + 1
            if attempt > MAX_ANALYSIS_ATTEMPTS:
                raise ValueError("理论综合生成次数超过上限")
            correction = state.get("analysis_validation_error", "")
            instruction = self._analysis_instruction(state)
            if correction:
                # 只反馈确定性校验原因；重用冻结 Evidence，不把上轮草稿重复上传。
                instruction += f"\n上轮校验失败：{correction}。请重新生成并修正引用。"
            output = await self.gateway.structured(
                instruction=instruction,
                schema=TheoryAnalysisDraft,
                system_prompt=self.system_prompt,
                thinking=bool(state.get("thinking")),
            )
            return {
                "analysis_draft": _normalize_draft(output),
                "analysis_attempts": attempt,
            }

        return await self._tracked("TheorySynthesis", state, operation)

    async def _analysis_validation(self, state: QuillState) -> dict[str, Any]:
        """在组装结果前统一校验理论、推荐和空白的 Citation。

        Args:
            state: 包含 TheorySynthesis 输出和本次固定 Evidence 的Quill状态。

        Returns:
            各分区数量及 PASS 或 RETRY 决策。

        Raises:
            ValueError: 第二轮仍有缺失、错位或越界引用。
        """

        async def operation() -> dict[str, Any]:
            """检查同次模型输出的全部结构化引用边界和文献真实性。

            Returns:
                已校验数量或触发重新生成的失败原因。

            Raises:
                ValueError: 第二轮结构化引用仍不满足本次 Evidence 约束。
            """
            allowed = {item.citation_index for item in state["evidence"]}
            known_titles = {item.document_title.strip() for item in state["evidence"]}
            draft = state["analysis_draft"]
            failure = ""

            # 1. 理论依据必须在事实原位标出与数组一致的编号。
            for item in draft.theoretical_basis:
                citations = set(item.citation_ids)
                if not citations <= allowed or _theory_citation_ids(item.description) != citations:
                    failure = failure or "理论依据包含缺失、错位或越界 Citation"

            # 2. 文献推荐不得引用知识库之外的文献标题。
            for item in draft.literature_recommendations:
                citations = set(item.citation_ids)
                inline = _theory_citation_ids(item.relevance_reason)
                if not citations <= allowed or inline != citations:
                    failure = failure or "文献推荐包含缺失、错位或越界 Citation"
                if item.document_title.strip() not in known_titles:
                    failure = failure or "文献推荐包含知识库外的文献标题"

            # 3. 研究空白的推导文字也属于模型生成的事实，必须保留证据逻辑检查。
            for gap in draft.research_gaps:
                citations = set(gap.citation_ids)
                if (
                    not citations <= allowed
                    or _theory_citation_ids(gap.evidence_logic) != citations
                    or not gap.evidence_logic.strip()
                ):
                    failure = failure or "研究空白缺少有效 Evidence 或证据逻辑"

            counts = {
                "theory_count": len(draft.theoretical_basis),
                "recommendation_count": len(draft.literature_recommendations),
                "gap_count": len(draft.research_gaps),
            }
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

    async def _citation_validator(self, state: QuillState) -> dict[str, Any]:
        """映射最终所有引用到真实 Document/Chunk/Page 并组装结果。

        Args:
            state: 已通过分析校验且包含全部专业派生结果的Quill状态。

        Returns:
            真实 Citation、参考文献、Evidence 摘要和可发布 QuillResult。

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
            draft = state["analysis_draft"]
            # 1. 文献推荐按用户冻结的上限确定性截断，保留模型原始顺序。
            limit = max(1, int(state["literature_limit"]))
            recommendations = draft.literature_recommendations[:limit]
            trimmed = len(draft.literature_recommendations) - len(recommendations)

            # 2. 汇总截断后各分区实际使用的 Citation 编号
            used: set[int] = set()
            for item in draft.theoretical_basis:
                used.update(item.citation_ids)
            for item in recommendations:
                used.update(item.citation_ids)
            for gap in draft.research_gaps:
                used.update(gap.citation_ids)

            # 3. 将编号严格投影回 EvidenceDedup 固定的真实 Document/Chunk/Page 引用
            citations = [item for item in state["citations"] if item.citation_index in used]
            if {item.citation_index for item in citations} != used:
                raise ValueError("Citation 无法映射到真实文献分块")

            # 4. 只为实际使用的文献生成指定格式参考文献，并计算可解释覆盖摘要
            references = [
                _format_reference(item, state["citation_style"])
                for item in state["documents"]
                if any(citation_id in used for citation_id in item["citation_ids"])
            ]
            years = sorted({item["year"] for item in state["documents"] if item["year"]})
            languages = Counter(item["language"] for item in state["documents"])
            summary = {
                "evidence_count": len(state["evidence"]),
                "document_count": len(state["documents"]),
                "year_range": [years[0], years[-1]] if years else [],
                "language_coverage": dict(languages),
                "theory_count": len(draft.theoretical_basis),
                "recommendation_count": len(recommendations),
                "gap_count": len(draft.research_gaps),
            }

            # 5. 组装唯一专业结果；最终回答由已校验结构化事实确定性组织，不再调用模型。
            limitations: list[str] = []
            if trimmed:
                limitations.append(
                    f"按 literature_limit={limit} 截断了 {trimmed} 条模型推荐的文献。"
                )
            result = QuillResult(
                status="SUCCESS",
                stage="THEORY",
                research_topic=state["research_topic"],
                language=state["output_language"],
                citation_style=state["citation_style"],
                theoretical_basis=draft.theoretical_basis,
                literature_recommendations=recommendations,
                research_gaps=draft.research_gaps,
                references=references,
                evidence_summary=summary,
                limitations=limitations,
                evidence_level="HIGH" if len(state["documents"]) >= 5 else "MEDIUM",
            )
            answer = _build_theory_answer(result)
            return {
                "citation_validation_status": "PASS",
                "validated_citations": citations,
                "used_citation_indices": sorted(used),
                "references": references,
                "evidence_summary": summary,
                "evidence_level": result.evidence_level,
                "quill_result": result,
                "final_answer": answer,
            }

        return await self._tracked("CitationValidator", state, operation)


def _normalize_draft(draft: TheoryAnalysisDraft) -> TheoryAnalysisDraft:
    """把模型草稿各正文中的全角 Citation 原位规范为半角 [n]。

    Args:
        draft: 模型返回的结构化理论综合草稿。

    Returns:
        正文 Citation 标记已规范化的同结构草稿。
    """
    return draft.model_copy(
        update={
            "theoretical_basis": [
                item.model_copy(
                    update={"description": _normalize_theory_text(item.description)}
                )
                for item in draft.theoretical_basis
            ],
            "literature_recommendations": [
                item.model_copy(
                    update={"relevance_reason": _normalize_theory_text(item.relevance_reason)}
                )
                for item in draft.literature_recommendations
            ],
            "research_gaps": [
                item.model_copy(
                    update={"evidence_logic": _normalize_theory_text(item.evidence_logic)}
                )
                for item in draft.research_gaps
            ],
        }
    )


def _normalize_theory_text(text: str) -> str:
    """把模型输出的全角方括号 Citation 转换为半角 [n] 形式。

    Args:
        text: 理论要点、推荐理由或证据逻辑正文。

    Returns:
        【n】与［n］已原位转换为 [n] 的正文。
    """
    normalized = re.sub(r"【\s*(\d+)\s*】", r"[\1]", text)
    return re.sub(r"［\s*(\d+)\s*］", r"[\1]", normalized)


def _theory_citation_ids(text: str) -> set[int]:
    """提取结构化正文中去重后的半角 Citation 编号。

    Args:
        text: 使用半角 ``[n]`` 格式的理论综合正文。

    Returns:
        正文实际使用的 Citation 编号去重集合。
    """
    return {int(value) for value in re.findall(r"\[(\d+)]", text)}


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


def _build_theory_answer(result: QuillResult) -> str:
    """把已通过引用校验的结构化结果组织为最终 Markdown 回答。

    Args:
        result: 已完成 Citation 投影的Quill THEORY 结果。

    Returns:
        每个事实段落都带真实 Citation 编号的 Markdown 文档。
    """
    headings = {
        "ZH": ("理论依据", "推荐文献", "研究空白与后续提示", "参考文献"),
        "EN": (
            "Theoretical basis",
            "Recommended literature",
            "Research gaps and next steps",
            "References",
        ),
        "BILINGUAL": (
            "理论依据 / Theoretical basis",
            "推荐文献 / Recommended literature",
            "研究空白与后续提示 / Research gaps and next steps",
            "参考文献 / References",
        ),
    }[result.language]
    sections: list[str] = [
        f"# {result.research_topic}",
        f"## {headings[0]}",
    ]
    sections.extend(
        f"### {item.theory}\n\n{item.description}" for item in result.theoretical_basis
    )
    if result.literature_recommendations:
        sections.append(f"## {headings[1]}")
        for index, item in enumerate(result.literature_recommendations, start=1):
            sections.append(
                f"### {index}. {item.document_title}\n\n推荐理由：{item.relevance_reason}"
            )
    if result.research_gaps:
        sections.append(f"## {headings[2]}")
        sections.extend(
            f"### {gap.gap}\n\n{gap.evidence_logic}" for gap in result.research_gaps
        )
    if result.references:
        sections.append(f"## {headings[3]}")
        sections.extend(f"- {reference}" for reference in result.references)
    if result.limitations:
        sections.extend(f"> {note}" for note in result.limitations)
    sections.append(f"> {result.academic_notice}")
    return "\n\n".join(sections)
