"""声明智鉴专业比较图使用的 LangGraph State。

该状态继承所有 Agent 共用的运行、Evidence 和发布契约，并补充自然语言比较请求
解析、A/B 双路检索、证据归属、比较草稿及结构化结果字段。
"""

from __future__ import annotations

from app.agents.state import AgentState
from app.agents.zhijian_schemas import (
    CompareObject,
    CompareQuery,
    ComparisonMatrixDraftRow,
    EvidenceBalance,
    ParsedCompareRequest,
    ZhijianAnalysisDraft,
    ZhijianResult,
)


class ZhijianState(AgentState, total=False):
    """保存智鉴从请求解析到专业比较结果发布的完整图状态。"""

    # 从自然语言解析出的内部比较请求，不作为前端参数表单。
    compare_request: ParsedCompareRequest
    # 经过确定性规则规范化的两个比较对象。
    compare_object_a: CompareObject
    compare_object_b: CompareObject
    # 本次 Run 最终采用的动态比较维度。
    compare_dimensions: list[str]
    # 对象×维度形成的显式查询计划；每项只包含一个对象和一个维度。
    compare_queries: list[CompareQuery]
    # A/B 两侧各维度检索结果及合并后的候选，仅在受控图内流转。
    object_a_retrieval: list[dict[str, object]]
    object_b_retrieval: list[dict[str, object]]
    merged_candidates: list[dict[str, object]]
    # EvidenceAssignment 的逐 Chunk 运行时派生结果。
    evidence_assignments: list[dict[str, object]]
    # 归属输出的图级尝试次数与固定候选校验结果，重试不得重新检索。
    evidence_assignment_attempts: int
    evidence_assignment_status: str
    evidence_assignment_error: str
    # 双侧证据充分性与来源覆盖统计。
    evidence_balance: EvidenceBalance
    # CrossAnalysis 形成、等待 Validator 的结构化草稿。
    zhijian_draft: ZhijianAnalysisDraft
    # 核心分析图级生成轮次，以及引用校验给下轮模型的确定性纠错原因。
    compare_analysis_attempts: int
    compare_analysis_validation_status: str
    compare_analysis_validation_error: str
    # 模型矩阵原稿；MatrixGenerator 会按证据归属重建最终矩阵。
    zhijian_matrix_draft: list[ComparisonMatrixDraftRow]
    # 智鉴专业工作台读取的最终结构化结果。
    zhijian_result: ZhijianResult
    # 无法安全执行时的自然语言澄清或边界状态。
    compare_block_status: str
    clarification_question: str
