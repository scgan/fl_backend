# AI 问答统一入口

AI 页面 `/agent` 使用 Research 服务的 `/agents/QIMING/runs`，固定 `entry_source=AI`，不再调用 Flask 的 `/ai_search/query` 或搜索历史接口。未新增数据库表或字段。现有 `entry_source` 字符串增加 AI 取值，数据库无需结构迁移。

## 会话与任务

- 会话、消息、运行、范围、证据、引用分别复用现有 Research 表。
- `options.corpus_ids` 为空表示冻结创建时可访问的全库文献；非空表示固定选文，HTTP 最多二十篇。
- 全库包含已审核公共文献和本人个人材料；正文检索仍要求当前双路索引就绪、附件摘要和构建版本一致。
- `SEARCH` 返回可勾选的题录候选；`QA` 复用启明及专业委派；`READING` 在原 AI 会话复用文献阅读图；`WRITING` 继续使用现有知识库确认与写作流程。
- `research_run.result_payload` 使用 `result_type=AI_ASSISTANT`、`schema_version=1`，载荷包含意图、候选、引用、检索词和原始问题。候选数量表示本轮召回结果，不宣称全库相关文献总数。
- AI 历史查询使用现有 `/conversations?entry_source=AI`；详情的 `assistant_runs` 恢复每轮结构化结果，未结束运行继续轮询。

## 专业页面交接

选中文献进入 `/user/literature-detail?ids=...`，详细分析与追问继续使用独立 LITERATURE 会话，保留原问题供编辑。文献阅读仍先完成分析，再允许追问。

## 自然语言文献分析

无需跳转即可输入“基于上述文献分析”“比较第二篇和第四篇的方法”“你刚才引用的文献有什么共同结论”。整体分析执行 `ANALYZE`，具体问题执行 `ASK`；AI 入口允许直接基于原文回答具体问题，无需先生成整套分析。后续相关问题沿用阅读集合，结果和原文引用保存在同一 AI Run / conversation，独立阅读页面的规则不变。

- 搜索候选、回答实际引用、已有阅读材料分别恢复为有序来源；研究回答的 Evidence 候选不冒充实际引用。来源独立于聊天消息截断预算，默认提供最近十个有文献的轮次，显式操作的旧卡片额外保留。
- `options.reading_context` 包含 `source_run_id`、`page`、`selected_ids`，仅提示页面当前显示状态。前端记录最近操作的材料卡，重新勾选或翻页旧结果后优先提交该卡；新的材料结果完成后成为当前来源。服务从同用户同会话成功 Run 校验来源，分页复用原搜索条件，跨页勾选也核验是否命中原查询。
- 分类器按语义识别整体解读、梳理、归纳、评述和综合比较等阅读请求，不只匹配固定问法。`reading_reference` 将材料指代与任务分开：`AUTO` 泛指优先使用同来源勾选集合，无勾选沿用当前列表；`SELECTED` 明确所选但未勾选时询问；`LIST` 明确整页/整组；`SPECIFIC` 按篇名、序号或引用编号定位。明确限定覆盖默认勾选，范围优先级由服务执行。分类器不生成文献 ID。来源缺失、歧义、篇名未定位、序号越界或超过二十篇时询问用户；全部搜索结果不能被静默替换成当前页。
- `ResolveReadingScope` 在正文读取前保存来源、顺序和材料版本；子材料工具只能读取父 Run 授权范围内的文献。读取及发布均复核权限和版本，追问引用进入既有证据和引用发布事务。
- `AI_ASSISTANT.payload.reading_result` 保存分析结构、推荐问题、有序 `source_documents`、`corpus_ids` 及 `scope_version`。前端执行中展示可展开的材料范围，成功后展示分析与追问，刷新从历史快照恢复。无需数据库结构迁移。
- 阅读卡片推荐问题通过 `options.reading={source_run_id, task:"ASK"}` 提交。此操作绕过自然语言任务分类，后端按同用户同会话的成功阅读卡片恢复原材料、版本及分析来源链，再调用 `LiteratureGraph` 的 `ASK`；即使问题本身是完整的新句子，也不会切换到全库研究。普通输入继续使用意图识别。
- 文献追问参照启明、溯知、智鉴的Evidence First、引用归属及“带具体错误反馈重生成”方式。指代规划读取完整历史，回答生成只接收独立问题、旧问题、已有分析及本轮原文，避免照抄旧答案。引用数量由提示词指导，不再因数量阈值直接判整轮失败。格式、引用编号或定位校验失败时最多进行三轮业务生成，每轮复用同一份材料并反馈校验原因；不重传失败草稿。耗尽后返回明确标注的原文摘录受限结果，不伪造比较结论。网关既有的传输和格式重试保持独立。
- 原阅读集合保留用于权限和版本核验，本轮计数、公开材料表及后续READING来源统一按 `target_corpus_ids` 从 `source_documents` 保序投影。旧快照同样投影，不改写历史正文。显式继续追问先复核原集合版本，再固定该卡片实际目标；“第一篇”按当前卡片显示列表解析。阅读子图不得用历史旧列表替换当前目标，越界或格式错误带当前允许ID重试，耗尽时明确询问篇目。

AI 入口写作继续通过现有任务卡确认知识库。本次自然语言阅读集合不自动转为写作范围，写作参数不能混入阅读上下文。

## 旧链路退出

已删除 `SearchHistory` 模型、旧查询及 `/ai_search/search/history` 接口和前端封装。保留题录搜索、时间线及旧阅读兼容接口。未自动删除实际数据库内的 `search_history` 表，也未迁移旧记录；旧记录不出现在新历史中。部署前后端需同步切换，确认没有旧客户端依赖后再按数据库运维流程清理旧表。

## 验证

内存库测试覆盖权限、幂等、并发、入口隔离、任务分流、HTTP 历史及消息/证据/引用发布。`tests/test_ai_reading.py` 额外验证自然语言阅读分支、来源序号、真实引用、分页勾选、旧卡片恢复、范围版本及同 AI 会话发布。外部模型、Qdrant 和 OpenSearch 的真实生产联调不属于离线验证。

## Agent 代码组织

AI 与启明、智鉴等并列放在 `app/agents/`：

- `ai_assistant_graph.py`：`AssistantGraph`，显式 StateGraph、条件边和逐节点步骤回调。
- `ai_assistant_state.py`：`AssistantState`，只继承公共 AgentState，不扩展 QimingState。
- `ai_assistant_schemas.py`：意图、候选文献及结果载荷 Schema。

`foreign_integration/service.py` 只构造已授权的输入状态、注入工具并调用 `graph.run`，继续承担运行和证据发布。旧 `foreign_integration/ai_assistant.py` 已移除；启明的 AI 专用 `run_general` 及状态字段已撤回。现有 API、表结构、模型配置复用方式及前端结果契约不变。

图拓扑：AuthenticateContext → IntentClassifier → RoutingDecision。SEARCH 执行 ExtractSearchConditions → DiscoverLiterature；QA 调用 QimingQA；READING 调用 LiteratureReading → ResolveReadingScope → LiteratureGraph；WRITING 调用 WritingTask。所有分支最终汇合到 Synthesizer，由服务统一发布结果。
