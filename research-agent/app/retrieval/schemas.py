"""定义混合检索请求、Evidence 与可定位 Citation 的稳定 API 契约。

请求模型只允许用户缩小已授权知识库范围，不能扩大权限；响应模型明确区分完整
证据正文与展示用短引文，并携带固定知识库版本。Citation 的文档、切块和页码均
来自数据库真实记录，文件地址还会在读取 PDF 时再次执行资源权限校验。
"""

from __future__ import annotations

import uuid

from pydantic import BaseModel, Field, model_validator


class RetrievalFilters(BaseModel):
    """限制检索候选文档的结构化元数据条件。

    这些条件最终由 MySQL 与知识库授权文档集合取交集，再下推到 Qdrant 和
    OpenSearch。客户端传入的 ``document_ids`` 因此只是更窄的白名单，不能绕过
    知识库成员关系、READY 状态或三路索引成功条件。
    """

    # 允许的文献语言代码集合；空列表表示不限制。
    language_codes: list[str] = Field(default_factory=list, max_length=8)
    # 发表年份下界，包含边界值。
    year_from: int | None = Field(default=None, ge=1000, le=2100)
    # 发表年份上界，包含边界值。
    year_to: int | None = Field(default=None, ge=1000, le=2100)
    # 作者名称模糊过滤条件。
    authors: list[str] = Field(default_factory=list, max_length=20)
    # 期刊名称模糊过滤条件。
    journals: list[str] = Field(default_factory=list, max_length=20)
    # 用户显式选择的文档白名单，仍会与知识库授权集合取交集。
    document_ids: list[uuid.UUID] = Field(default_factory=list, max_length=100)

    @model_validator(mode="after")
    def validate_year_range(self) -> RetrievalFilters:
        """拒绝起止年份颠倒的过滤条件。

        Returns:
            已完成范围校验的当前对象。

        Raises:
            ValueError: 起始年份晚于结束年份。
        """
        if self.year_from and self.year_to and self.year_from > self.year_to:
            raise ValueError("year_from 不能晚于 year_to")
        return self


class RetrievalRequest(BaseModel):
    """一次限定知识库范围的 Evidence First 检索请求。

    ``candidate_limit`` 控制每个召回分支的成本，``evidence_limit`` 控制通过质量
    门槛后的响应规模；两者都不能改变授权范围。服务开始检索时读取并回传知识库
    当前版本，使调用方知道本次证据对应哪个可见集合。
    """

    # 检索范围对应的知识库主键。
    knowledge_base_id: uuid.UUID
    # 用户原始中英文检索问题。
    query: str = Field(min_length=2, max_length=1000)
    # 可选结构化候选过滤器。
    filters: RetrievalFilters = Field(default_factory=RetrievalFilters)
    # 每个外部索引分支最多召回的候选数量。
    candidate_limit: int | None = Field(default=None, ge=5, le=100)
    # 经过 Evidence First 门槛后最多返回的证据数量。
    evidence_limit: int | None = Field(default=None, ge=1, le=20)


class CitationRead(BaseModel):
    """映射真实 Document、Chunk、页码和鉴权原始 PDF 的引用。

    Citation 不接受生成模型自报的来源，而是由服务根据已复核 CandidateChunk
    构造。``quote`` 只用于快速展示，完整证据仍在 Evidence 中；``file_url`` 指向
    会重新校验知识库权限的接口，并通过 URL fragment 定位 1-based 原文页码。
    """

    # 与 Evidence 一一对应的展示序号，起始值为 1。
    citation_index: int
    # 原始文献主键。
    document_id: uuid.UUID
    # 支撑引用的真实切块主键。
    chunk_id: uuid.UUID
    # 引用列表展示标题。
    document_title: str
    # 原文起始页，使用 1-based 页码。
    page_start: int
    # 原文结束页，使用 1-based 页码。
    page_end: int
    # 切块所属章节标题，解析失败时允许为空。
    section: str | None
    # 从真实切块规范空白后截取的短引文。
    quote: str
    # 再次执行知识库权限校验的 PDF 地址，包含页码片段。
    file_url: str


class EvidenceRead(BaseModel):
    """经过授权复核、融合、重排和绝对门槛过滤的证据片段。

    Evidence 保留完整真实 Chunk，既不改写也不由模型补全。``relevance_score``
    是可解释的本地重排分值，不等同于 Qdrant 相似度或 BM25 原始分；两路名次
    单独返回，便于调用方审计证据来自哪个召回分支。
    """

    # 对应 Citation 的展示序号。
    citation_index: int
    # 证据所属原始文献主键。
    document_id: uuid.UUID
    # 证据对应的真实切块主键。
    chunk_id: uuid.UUID
    # 原始文献标题。
    document_title: str
    # 原始文献作者对象，直接来自 MySQL 主记录。
    authors: list[dict[str, str]]
    # 可选出版年份，用于 Agent Evidence Level 的时间覆盖解释。
    publication_year: int | None
    # 可选期刊名称，用于 Evidence Panel 解释来源多样性。
    journal: str | None
    # 切块识别语言代码。
    language_code: str
    # 证据起始 1-based 页码。
    page_start: int
    # 证据结束 1-based 页码。
    page_end: int
    # 证据所属章节标题。
    section: str | None
    # 未改写的完整切块正文。
    content: str
    # RRF 与确定性重排得到的 0 到 1 相关性分值。
    relevance_score: float
    # 只依赖两路名次计算的原始 RRF 融合分值，用于运行审计。
    retrieval_score: float
    # 切块在向量分支的名次；未被该分支召回时为空。
    vector_rank: int | None
    # 切块在词法分支的名次；未被该分支召回时为空。
    lexical_rank: int | None


class RetrievalResponse(BaseModel):
    """返回可审计查询处理结果、知识库版本及可引用证据。

    该响应只提供检索基座，不生成研究结论。没有候选或候选未通过绝对质量门槛
    时，``insufficient_evidence`` 必须为真、Evidence/Citation 必须为空，并返回
    明确拒答信息，防止后续 Agent 用预训练记忆补造事实。
    """

    # 完成本次授权的知识库主键。
    knowledge_base_id: uuid.UUID
    # 检索开始时固定的知识库当前版本。
    knowledge_base_version: int
    # 用户原始查询。
    original_query: str
    # 检测到的 zh、en 或 mixed。
    detected_language: str
    # 去除问句外壳后的可审计查询。
    rewritten_query: str
    # 实际提交给索引的中英文扩展表达式。
    expanded_queries: dict[str, str]
    # 外部召回并经数据库授权复核后保留的候选数。
    candidate_count: int
    # 通过绝对证据门槛的正文分块。
    evidence: list[EvidenceRead]
    # 与证据一一对应的真实引用。
    citations: list[CitationRead]
    # 是否没有任何证据通过门槛。
    insufficient_evidence: bool
    # 证据不足时的拒答说明；有证据时为空。
    message: str | None
