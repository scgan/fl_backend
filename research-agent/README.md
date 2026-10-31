# EduResearch 共库后端

当前唯一服务入口：app.foreign_integration.main:app。

- app/foreign_integration：Foreign 认证、表映射、research_* 迁移、只读召回、运行及会话 API。
- app/agents：四个研究模块及独立文献阅读 LangGraph、状态、Schema、模型网关。
- app/retrieval：双语查询处理、RRF/确定性重排与 Evidence/Citation 数据契约。
- app/common：错误响应、模型和智鉴证据门槛配置。

文献上传、解析、分块、索引生产及数据库主表由 Foreign Backend 管理。
研究检索使用 Qdrant + OpenSearch 2.9.0 BM25，两个分支共享授权文献白名单。
只接受 `corpus_index_job` 总状态与两路状态均为 `SUCCESS` 的当前附件快照。
BM25 独有命中从 Qdrant 按 point ID 回取正文，再验证构建版本、正文哈希、附件与页码。
查询结束后重新回库核验权限与版本，最终以 RRF 分和确定性重排分生成证据。
旧 app.main、Alembic、独立 ORM/API、Celery 与存储代码已删除。
必须保留 python -m app.foreign_integration.migrate，研究记录仍需持久化。
requirements.foreign.txt 与 pyproject.toml 的生产依赖保持一致。

```bash
python -m pip install -e ".[dev]"
pytest tests
ruff check .
```

本目录是迁入 Foreign/Backend 后的 Agent 维护入口；原 Eduresearch/EduResearch/backend 暂留作迁移核对副本。
请在本目录执行上面的安装与测试命令，并使用独立 Python 3.12 环境，避免与上级 Flask 后端的同名 app 包及依赖混用。
测试使用隔离数据和模型替身，不调用真实付费模型。

AI 题录检索首次生成最多五百篇的有序 ID 快照，首屏只返回十篇详情，随运行结果持久化。
后续翻页只按快照读取当前页详情并复核权限，不重复执行主题匹配、计数和排序。
无快照的旧任务在首次翻页时补存一次（含空结果）；快照不会绕过资源成员或材料可见性。

文献比对不按中英文白名单拒绝来源语言。解析结果的 `source_language_codes` 可指定
多个来源语种（例如 `ZH`、`EN`、`JA`），检索按这些语种的并集过滤；未指定时不限制。
旧的 `source_language_requirement` 与 `language_filter` 单语种字段仍可使用。
来源标签统一处理语言名称、历史代码和地区变体（例如 `日语`、`jp`、`ja-JP` 均为 `ja`）。
该筛选不保证结果引用覆盖每个语种；指定语种没有有效证据时仍返回证据不足。

## Docker 部署

### 数据库研究入口

业务菜单的「智能体研究」按有效根数据库生成「数据库名＋智能体」子菜单，使用
`/database-research/:databaseId` 路由复用四个研究模块。工作台固定数据库范围，不提供切换选择框；
旧的不含数据库 ID 的链接会按原会话绑定（空入口按首个根库）跳转。
原 `/research` 知识库入口保持原行为。数据库列表接口为
`GET /api/v1/research-databases`，仅向已认证的数据收集者返回
`database.parent_id = -1` 且未删除的根节点。

创建数据库研究任务时，使用原 `/api/v1/agents/{code}/runs` 接口，提交
`entry_source: "DATABASE"` 和 `database_id`，不提交 `knowledge_base_id`。
会话固定绑定根数据库，不能在后续提问中更换来源；历史列表和详情使用
`entry_source=DATABASE` 隔离，专业委派保持原会话来源。

`foreign_integration/research_scope.py` 集中负责两种来源的授权与范围解析。
根数据库的范围包括全部有效子孙节点，通过 `corpus_database` 关联文献并在 SQL 层去重；
删除的分类分支及未审核、删除的文献不进入范围。检索继续复核现有双路索引与附件快照，
任务创建、检索及回答发布使用同一来源规则，不复制智能体图或新建索引。
`database` 与 `corpus_database` 仅读取，迁移不会创建或修改这些业务表。

在 Foreign 目录执行：

```bash
docker compose -f compose.local.yaml build backend research-agent
docker compose -f compose.local.yaml up -d opensearch backend index-worker research-agent frontend
```

Compose 的 research-agent 构建目录是 ./Backend/research-agent，Dockerfile 是 Dockerfile.foreign。
Agent 仍使用 18000 端口，前端 Nginx 的 /research-api/ 转发规则不变。
启动迁移仍由 python -m app.foreign_integration.migrate 执行，不在目录迁移时直接操作数据库。
Backend 启动时负责幂等扩展 `corpus_index_job`；Agent 就绪检查要求新增字段已存在。
`FOREIGN_OPENSEARCH_URL` 在 Compose 中为 `http://opensearch:9200`，宿主调试为
`http://127.0.0.1:19200`；两边 `FOREIGN_OPENSEARCH_INDEX` 必须一致，默认 `foreign_chunks_v1`。
排序默认 `FOREIGN_RRF_K=60`、`FOREIGN_MIN_EVIDENCE_SCORE=0.18`、
`FOREIGN_MIN_VECTOR_SCORE=0.35`、`FOREIGN_MIN_LEXICAL_COVERAGE=0.5`。
研究检索的向量请求默认超时 30 秒，可用 `FOREIGN_QDRANT_TIMEOUT_SECONDS` 调整（1～120 秒）。
向量查询及结果回读仅在 HTTP 超时时重试一次；失败步骤区分查询向量编码、向量检索、
词法检索和向量结果回读，不记录凭据或供应商响应正文。
智鉴 A/B 检索步骤按实际返回的去重 Chunk 统计 `evidence_count`，另记 `query_count`
和 `candidate_count`；没有新增证据的后续步骤沿用当前证据数，避免将增量空值误报为零。
每路候选 FAST=30、DEEP=50；证据默认 5/10 条，上限 20 条。
历史只建向量的文献须先回填双路索引，操作见 `../docs/corpus_index_worker.md`。

文献入库、普通搜索和研究召回统一使用 `qwen3.7-text-embedding-flash`（1024 维）。
入库使用 `text_type=document`，查询使用 `query`；新片段携带模型和维度字段。
召回只接受当前模型的片段，旧模型索引需要重新构建后才会进入证据集合。
Backend 的 `app/utils/bailian_client.py` 和 Agent 的 `app/common/bailian_client.py` 读取 `DASHSCOPE_API_KEY`（兼容 `QWEN_API_KEY`）
和原生 `FOREIGN_BAILIAN_API_BASE`（默认 https://dashscope.aliyuncs.com/api/v1）。
该地址与生成回答的 OpenAI 兼容地址分别配置。
OCR 默认使用 `qwen-vl-ocr` 的高精识别，成功页缓存由本地 Docker volume 持久化。
如固定模型只返回位置框而未返回文字，适配器使用同一模型的通用文字识别，
并明确标记 `location_precision=page`（分区域重试时为 region），只提供真实页/区域边界。
不会将无法匹配的检测框伪装成文字行坐标。单文档 HTTP 并发默认为 4，可通过
`FOREIGN_OCR_CONCURRENCY` 设置为 1..8；PDF 解析/渲染不在线程池并行执行。
本地 Qwen3 tokenizer 仅估算分块预算，`token_count_kind=chunking_estimate`；
实际模型 token 用量从百炼响应 usage 记录，不再套用旧模型的精确计数断言。

## 全局搜索文献阅读

`POST /api/v1/agents/literature/runs` 直接接收所选公开已审核文献，不要求加入知识库。
首次提交 `query` 和 `options={"task":"ANALYZE","corpus_ids":[1,2],"template_version":1}`；
追问传入同一 `conversation_id`，将 task 改为 `ASK`。会话内文献顺序和范围保持固定。
首次分析按用户、文献/附件/索引版本、Prompt 和模型配置幂等复用；重新分析使用已有会话。

复用现有 research_* 表，仅 research_conversation 保存可空的 knowledge_base_id，
research_run 通过 conversation_id 取得会话知识库，接口中的 knowledge_base_id 为派生字段。
原四个模块仍要求有效知识库。阅读历史通过
`/api/v1/literature/conversations` 获取，限定会话 entry_source=LITERATURE。

### 会话入口隔离

只有 `research_conversation` 增加 `entry_source` 和可空的 `database_id`；
database_id 仅预留，尚不参与创建参数或检索。三个入口固定为：文献问答 `LITERATURE`、
辅助创作 `WRITING`、启明/智鉴/溯知/融通工作台 `RESEARCH`。
辅助创作调用 QIMING 或由启明委派专业子图时，会话仍保持 WRITING。

创建运行时传入顶层 `entry_source`；同一会话后续请求不能切换来源，幂等重试也校验来源。
旧调用未传值时，LITERATURE/QUILL 分别默认阅读/写作，其他模块默认研究。
`GET /conversations` 和 `GET /agent-runs` 按入口过滤后分页；会话列表默认 RESEARCH，
运行列表未传来源时按模块推导。`GET /conversations/{id}?entry_source=...`
拒绝恢复其他入口的会话，前端三个入口均显式传入来源，包括写作页的缓存恢复。

这些分类不替代本人身份、知识库权限等现有校验。

启动迁移幂等增加字段及历史查询索引，并只回填未标记会话：未绑定知识库或数据库的旧会话
归为 LITERATURE，含 QUILL（或旧名 DIAOLONG）运行的归为 WRITING，其余归为 RESEARCH。
仅调用启明的旧写作会话没有入口标记，无法可靠识别，暂归 RESEARCH；已明确分类的
会话不会在后续启动时被重分。写作页缓存若指向其他入口会话，会清除该缓存引用。

图完整读取当前构建版本的原文片段，默认同时分析 4 篇，每篇内部仍分批阅读，结果按
所选顺序收集，再生成固定六分区分析和推荐问题。`FOREIGN_LITERATURE_ANALYSIS_CONCURRENCY`
可设置每轮并发数（1～20，默认 4）；该限制按运行计数，多用户共享模型额度需据额度调整。
任一篇失败会取消并等待其他任务结束，已开始的步骤记录为 SUCCESS、FAILED 或 CANCELLED。
服务器 `../start-dev.sh` 默认导出 `FOREIGN_LITERATURE_ANALYSIS_CONCURRENCY=4`，
`../deploy-dev.sh` 将该变量传入 Agent 容器；`Dockerfile.foreign` 也设有默认值 4。
上传后重新构建、部署 Research Agent 即可使用；服务器已有的同名环境变量会覆盖默认值。

逐篇分析结束后，`RecommendLiterature` 与 `SynthesizeReading` 并行，汇合后统一收尾。
综合分析提出 3～5 个潜在创新方向，每项包含 `title` 小标题及 `text` 正文，提示词要求正文
约 200～300 中文字，展开文献依据、具体问题、方法与数据、验证路径及预期价值和边界；
材料不足时允许适当缩短，不为凑字数虚构依据。前端以小标题和正文直接完整展示，Markdown
快照同步保留各项标题。旧历史无标题时显示编号方向，重新分析后生成具体小标题。
库内分支从已审核、未删除文献中召回，排除当前选择，按文献聚合 BM25 与向量排名，
并以题录关键词匹配补充尚未全文索引的文献，最多返回 10 篇，保留 25 秒预算。
同一 `RecommendLiterature` 节点并行执行联网分支：使用 Run 冻结的模型及接口调用
Responses `web_search`（最多三次工具调用，请求至多十篇候选），只接受供应商真实搜索
来源中的 URL。不调用 Crossref，不要求 DOI；作者、年份及期刊由模型按搜索结果提取，
缺失则留空，这些题录字段未经独立登记机构核验。

随后查整个未删除文献库，包含未审核、未索引记录，按中英文题名、轻微排版差异和
原始来源中的 DOI 排除重复；不向用户返回这些排除记录。未登记的译名或别名无法保证
识别为同一论文。联网推荐至多追加 10 篇，库内与联网各最多 10 篇；不写入回答 Evidence
或自动导入文献库。联网 API 只收到最多五条有限主题线索，不接收原文全文或全库题录，
不抓取模型自报的网址。

题录写入结果 JSON 的 `recommended_documents`，`origin` 区分 LIBRARY/WEB；
外部文献 `corpus_id=null`，用可选的 `doi` 和 `external_url`（搜索来源入口）标识。
整体状态及两路独立状态保存在结果快照，历史恢复直接读取，不重复联网。
联网请求总预算 120 秒（读取空闲 60 秒、连接 10 秒），联网分支预算 135 秒，整个推荐节点
最多 140 秒；任一路失败仍保留另一条分支结果。`web_recommendation_details` 在结果和步骤
摘要中记录搜索/去重阶段、候选/排除/保留数量、耗时及脱敏错误类型；超时不伪装成零候选。
推荐不影响已生成的阅读分析。当前 Provider 不支持带来源的搜索时，降级到库内结果。
前端新增“文献推荐”标签，历史结果缺少该字段时提示重新分析。新流程使用独立的
首次分析幂等版本 v6，避免重新分析时复用旧分析结构的结果。无需新增业务表或数据库迁移。

继续提问的 `AnswerFollowup` 使用带 `citation_id` 的原文 chunk，模型按段返回
`citation_ids`；后端核验真实片段并按首次出现顺序生成 `[1][2]`。只将实际引用的片段
写入 `research_evidence`，引用关系写入 `research_citation`，与回答一起发布。
会话接口恢复每条消息自己的编号、文献、附件和页码；点击编号切换中间 PDF 并跳到
chunk 起始页，并根据完整证据快照高亮原文。引用字段 `content` 来自已保存的
`content_snapshot`，不需要重新调用模型；跨页正文在对应页继续高亮，缩放时同步坐标。
无法完整匹配时仅高亮已匹配内容并提示，无文字层时保留正常阅读和未匹配提示。
无编号映射的旧回答保留普通文本，失效附件或页码明确提示。
启动时登记文献 Prompt v2 并归档内置 v1，保留历史记录，不新增业务表或字段。
不增加逐条事实审核或语义 Reviewer；检查访问权限、材料版本、输入预算、输出结构
与 chunk 引用定位。使用支持 1M 上下文的模型，每次输入采用保守的 UTF-8 字节预算并分批，
超长追问使用所选范围内的相关片段，不静默截断全文。

`scripts/evaluate_literature_live.py <report-path>` 可测试真实模型；默认仅发送脚本内
虚构研究材料，不读取真实文献、不连接或写入数据库。模型配置沿用项目环境。

`python -m scripts.evaluate_literature_concurrency_live <report-dir> --discover-only`
只读列出真实“双减”候选文献与配置的模型目的地，不读取原文或调用模型。去掉
`--discover-only` 后，核验并分析 10 篇真实文献，验证并发上限、结果顺序及汇总时机；
会将原文发送给配置的模型服务。远程数据库始终只读，步骤写入报告目录的 SQLite。

### Run 知识库字段升级

升级前停止旧版 research-agent 并等待在途任务结束，再使用新版执行
`python -m app.foreign_integration.migrate`。迁移会检查全部历史 Run（包括软删除记录）
与所属会话的知识库 ID 是否一致；不一致或缺少会话时停止删列，需先核对异常数据。
一致时移除 research_run.knowledge_base_id 及其 MySQL 外键，重复执行不会重复删列。
升级后再启动新版服务，旧版服务不能继续读写已删除的字段。
知识库绑定仍固定在会话层；database_id 仍为预留，本次升级不启用数据库检索。

固定数据库的历史列表使用原 `GET /api/v1/conversations` 和 `GET /api/v1/agent-runs`，
同时传递 `entry_source=DATABASE&database_id=<根库ID>`。后端先按所有者、入口和数据库筛选，
再统计总数及分页；省略 `database_id` 保持原查询范围。两个详情接口也接受可选
`database_id`，绑定不符返回 404，避免当前智能体恢复另一根库的记录。
