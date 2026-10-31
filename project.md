# Foreign Backend 迁移记录

## 2026-10-28 Backend Worker 跳过被内容审核拒绝的 OCR 页

- AICP HTTP 400 明确返回 `data_inspection_failed` / `DataInspectionFailed` 时跳过该页 OCR，继续其他页；保留原页号，分区拒绝时舍弃整页 OCR。其他请求错误仍沿用既有重试或失败处理。
- 剩余正文完成 Qdrant 与 OpenSearch 双路索引后发布 `SUCCESS`，将脱敏的跳页摘要保留在既有 `corpus_index_job.error_message`（最多 2000 字符）；无需迁移表结构或新增状态。下一次构建重新收集摘要，不沿用旧警告，全篇无可索引文字仍失败。
- Python 3.9 一次性容器禁网回归：master 共 147 项（146 通过、1 项在线测试跳过），dev 共 138 项（137 通过、1 项在线测试跳过）；覆盖拒绝页后续处理、原页号、分区拒绝、非审核错误及成功摘要保留/重建清理。`git diff --check` 通过，尚未部署服务器或重试线上任务。
- 修复适用于 dev/master 共用的 Backend Worker，保留两分支各自的部署配置。已检查源码副本：当前工作区没有 `Index/`；`ProductionWorker/` 是独立交付实现，使用不同的 HTTP 客户端且不被 Backend 镜像加载，本次未同步或重打独立交付包。

## 2026-10-17 Research 后端注释审计

- 检查当前维护入口 `research-agent/app` 的 57 个 Python 文件、426 个函数/方法（含嵌套回调），补齐 6 个文件中 42 处缺失的函数说明，并补充阅读模型字段、Embedding 配置、材料来源校验和运行发布的职责说明。
- 为文献推荐、逐篇分析、综合阅读、追问、全量材料回取及运行执行补充业务阶段、输入输出、并发顺序、失败降级和版本复核注释。Pydantic 模型使用源码注释说明职责，保留既有 JSON Schema 和模型提示词。
- 核对修改前后去除文档字符串的可执行 AST 一致，六个阅读模型 JSON Schema 一致；未改变业务逻辑、接口、字段约束、检索门槛或部署配置，无需重启服务。
- 在当前 Research 镜像中以只读源码、禁网方式执行回归：244 项通过、4 项网关测试失败。四项均已用修改前的 HEAD 源码在相同环境复现，涉及 `test_agent_gateway.py` 中空内容重试、逐次关闭思考、Qwen 思考参数和截断重试测试；现有模拟未覆盖实际进入的 OpenAI SDK 流式请求，禁网环境下产生连接错误。本次仅补注释，未改写这些既有测试或模型传输实现。

## 2026-10-14 Quill 收尾：Word 导出、产物历史、会话历史与整篇连播起草

- 新增 `docx_export.py`：纯函数把产物列表组装为 Word 文档——整篇导出以最新 PLAN 为骨架、按 section_ref 逐节插入版本号最高的 DRAFT 草稿、未起草章节回退写作要点，并渲染 RQ 与参考文献；requirements.foreign.txt 增加 python-docx。
- api 新增 `GET /writing-artifacts/{conversation_id}`（会话产物列表，含阶段/版本/section_ref/内容）与 `POST /quill/export-docx`（conversation_id 整篇或 artifact_id 单产物，StreamingResponse 返回 .docx）。
- 前端（本地未提交）：会话 ID 持久化 + 刷新后从 /conversations/{id} 恢复写作历史（修复"对话没有记录"）；「新建对话」按钮；PLAN 卡片新增「整篇连播起草」（按大纲顺序循环 DRAFT 运行，逐节落消息，完成后可导出全文/复制）；「产物历史」弹窗（列表 + 单产物导出）；导出经 blob 下载。
- 新增 test_quill_export.py（2 项）；全量 238 项测试通过；前端 eslint+build 通过。
- dev 服务器需再次部署生效（requirements 变更会触发镜像重建）。

## 2026-10-14 Quill Phase 3：逐节起草（DRAFT）、学术修改（REVISE）与可点击引用

- 五阶段全部开放。新增 DRAFT（`quill_draft_graph.py`，继承检索管线）：引用 PLAN 产物（artifact_id 必填 + section_ref 必须命中大纲章节，create_run 快速失败校验），章节事实句原位 [n] 引用、证据不足处 `[待补充文献：主题线索]` 占位并计入 evidence_summary；REVISE（`quill_revise_graph.py`）：待改文本走 query 字段 + revise_instruction，剥离旧运行编号、新增论断必须挂本次真实引用，纯表达性修改按 INSUFFICIENT 发布。
- api options 白名单：DRAFT 要求 artifact_id+section_ref；REVISE 要求 2-500 字修改要求；六阶段全部开放。
- service：_load_writing_artifact 泛化 expect_stage（PLAN 引 RQ、DRAFT 引 PLAN）；产物持久化扩展到 RQ/PLAN/DRAFT/REVISE 四阶段并记录 section_ref；发布映射补 draft_result/revise_result。
- 前端（本地未提交）：[n] 引用可点击——按运行缓存证据面板，点击打开带页码锚点的原文 PDF；PLAN 结果新增「逐节起草」按钮（弹窗选章节、校验编号）；划词工具条新增「学术修改」。
- 新增 test_quill_draft.py（4 项）与 test_quill_revise.py（2 项），更新 DRAFT/REVISE 白名单旧断言。全量 236 项测试通过；前端 eslint+build 通过。
- dev 服务器需再次部署生效；前端仍待用户决定提交。

## 2026-10-14 雕龙更名为 Quill（面向外语学院的英文品牌名）

- 应用户要求，第五/六个业务 Agent「雕龙·论文写作」更名为「Quill·论文写作」，编码 DIAOLONG 全量改为 QUILL（含 API 路径 /agents/QUILL/runs、种子 stable_id、系统提示词开头）。
- 文件与标识符机械化重命名：diaolong_* → quill_*（schemas/state/graph/polish/rq/plan 共 9 个模块 + 4 个测试文件），catalog/migrate/service/api 同步更新；残留检查确认零遗漏。
- 全量 228 项测试通过；ruff 通过（retrieval.py 两处历史 E501 属队友提交，未改动）。
- 迁移基于 stable_id("agent","QUILL") 注册新种子行；旧 DIAOLONG 种子行需在部署新版后手工停用（is_deleted=1）。
- 前端本地同步更名（api/quill.js、WritingAssistant.vue 引用与文案），按用户要求暂不提交。

## 2026-10-14 雕龙 Phase 2：研究问题生成（RQ）与写作方案（PLAN）

- 新增 `research_writing_artifact` 表（会话内阶段产物版本化：stage/version/parent/content/citation_ids），随 RESEARCH_TABLES 启动自动建表；RQ/PLAN 发布时由 `_publish` 原子写入产物并将 artifact_id 回填进 result_payload。
- RQ 阶段（`diaolong_rq_graph.py`，继承 THEORY 检索管线并参数化 _stage_intent/_retrieval_purpose）：候选研究问题表述为创作、rationale 为硬证据（原位 [n] 与 citation_ids 一致），theory/method hints 为待查证创作线索。
- PLAN 阶段（`diaolong_plan_graph.py`）：章节结构与写作要点为纯创作（禁止夹带任何 Citation 标记），suggested_citations 为硬证据且必须命中本次 Evidence；接受同会话 RQ 产物（artifact_id 32 位十六进制，create_run 快速失败校验归属与阶段），PLAN 无建议引用时按 INSUFFICIENT 发布。
- api options 白名单重构为 RQ/PLAN/THEORY 通用研究类校验器 + POLISH 校验器；DRAFT/REVISE 仍显式 422。
- service create_run 快速失败校验 PLAN 产物归属；RunExecutor 阶段路由 RQ/PLAN；产物持久化新表。
- 新增 tests/test_diaolong_rq.py（4 项）、test_diaolong_plan.py（4 项）；更新 RQ/PLAN 白名单相关旧断言。全量 228 项测试通过；新增文件 ruff 通过（retrieval.py/bailian_client 等存在队友提交的历史 lint 项，未改动）。
- 前端 WritingAssistant.vue 已同步点亮「选题推荐」「文章大纲」标签（RQ 运行返回 artifact_id 并支持一键"基于这些问题生成大纲"），按用户要求前端暂不提交。
- 本地与测试全绿；dev 服务器尚未二次部署（需再次 git pull + deploy-dev.sh 使新表与新阶段生效）。

## 2026-10-14 辅助创作接入真实前端（Frontend 仓库）

- 前端仓库已克隆至工作区同级 `Frontend/`（cnb.cool/ccnu_teamlcy_projects/foreign_language/Frontend，Vue3 + Vite + Element Plus）。
- 新增 `src/api/diaolong.js`：研究智能体专用 axios 实例（baseURL `/research-api`，复用 localStorage Foreign Bearer Token，401 跳登录），暴露创建运行与读取运行两个服务。
- 新增 `src/views/UploadView/FileManage/WritingAssistant.vue`：知网风格辅助创作页——左侧富文本编辑器 + 绿色划词工具条（续写/扩写/缩写/改写/润色/中译英/英译中 + 概念解释/文献推荐 + 复制），右侧助手面板（快捷标签占位 Phase 2、消息流、创建运行 + 2 秒轮询直至终态、润改结果替换选区/复制、THEORY 结果 marked 渲染并显示证据等级）。知识库 ID 本地持久化。
- 接线：`router/upload.js` 新增 `/user/writing`（角色守卫沿用 /user 的 role=2，与 Agent 仅对数据收集者开放一致）；`UploadView/index.vue` 侧边栏新增「辅助创作」菜单；`vite.config.js` 开发代理 `/research-api → 10.222.9.151:18000 /api/v1`；`default.conf` 与 `default-dev.conf` 补充 `/research-api/` location（SSE 友好配置）。
- `npm run build` 通过（8.78s），新增/修改文件 ESLint 通过。
- 注意：Agent 服务端启动需设置 `FOREIGN_AGENT_CORS_ORIGINS` 的情况仅限独立演示页直连；本页经 vite/nginx 同源代理，无需 CORS。真实模型端到端验证待 dev 环境部署雕龙后进行。

## 2026-10-14 雕龙 Phase 1.5：POLISH 划词润改与知网风格演示页

- 模拟知网“辅助创作”：新增雕龙 `POLISH` 阶段承接编辑器划词工具条的七个纯创作操作（续写/扩写/缩写/改写/润色/中译英/英译中），`DiaolongStage` 扩展为六阶段；概念解释与文献推荐属硬证据操作，复用 THEORY 阶段。
- 新增 `diaolong_polish_state.py` 与 `diaolong_polish_graph.py`（单节点 TextTransform，结构化输出 PolishOutput，不检索、不产生 Citation，`evidence_level` 固定 INSUFFICIENT）；`DiaolongPolishResult` 附带学术诚信提示。
- `api.py` 拆分 THEORY/POLISH 白名单校验（action 七选一、上下文各限 3000 字、选区走 query 字段限 4000 字）；`service.py` 按 stage 路由到润改图或理论图，发布仍走统一 Citation/范围复核闸门（POLISH 无引用合法通过）。
- `main.py` 新增 `FOREIGN_AGENT_CORS_ORIGINS` 环境变量控制的可选 CORS 中间件（默认关闭，仅为独立编辑器演示页开放 GET/POST/OPTIONS）。
- 新增 `docs/辅助创作演示.html`：知网风格双栏演示页——左侧富文本编辑器+绿色划词工具条，右侧助手面板（快捷标签占位 Phase 2、创建运行+轮询读取结果、润改结果一键替换选区/复制、THEORY 结果 Markdown 渲染）。
- 新增 `tests/test_diaolong_polish.py` 五项：纯创作发布契约、指令携带前后文与诚信边界、七操作映射完整、POLISH options 白名单（未开放阶段 422）、THEORY 原契约回归。research-agent 全量 199 项通过；新增文件 ruff 通过。
- 本机仅完成代码与测试验证；未重建/部署镜像，未用真实模型端到端验证。

## 2026-10-14 雕龙（DIAOLONG）Phase 1：学术研究辅助骨架与 THEORY 阶段

- 按 `docs/学术研究辅助设计方案.md` 实施第一阶段：新增第五个业务 Agent「雕龙·论文写作」（code=DIAOLONG），Phase 1 仅开放 THEORY 阶段（理论依据与文献推荐），RQ/PLAN/DRAFT/REVISE 阶段编码已在 API 白名单冻结并显式 422 拒绝。
- 新增 `app/agents/diaolong_schemas.py`（阶段参数、TheoryConcept/TheoryQueryPlan/CitedTheory/LiteratureRecommendation/TheoryGap/TheoryAnalysisDraft/DiaolongResult）、`diaolong_state.py` 和 `diaolong_graph.py`（固定八节点工作流：ConceptExtraction → QueryExpansion → LiteratureRetrieval → EvidenceDedup → MetadataNormalization → TheorySynthesis → AnalysisValidation → CitationValidator；两轮有界引用重试，绝不重新检索证据）。
- Evidence First 契约：理论要点、推荐理由与研究空白的每个事实句必须原位使用真实 [n] Citation 且与 citation_ids 数组一致；文献推荐标题必须命中知识库真实文献，库外标题直接拒绝；最终回答由已校验结构化结果确定性组装，不再调用模型；附带 AI 辅助学术规范提示。
- 接线：`catalog.py` 注册 Agent 与安全 Prompt v1；`api.py` agents 集合、`_validate_options` 雕龙白名单（stage/citation_style/literature_limit 5–20/year 范围）、历史查询 pattern；`service.py` AGENT_CODES、RunExecutor 雕龙分支、发布结果映射 DIAOLONG_WRITING_ASSISTANCE v1；`migrate.py` 启动种子登记（雕龙与其他专业 Agent 一致要求 OPENAI_COMPATIBLE 模型）。
- Phase 1 产物持久化复用 research_message 与 research_run.result_payload，未新增数据表；`research_writing_artifact` 表按方案推迟到 Phase 2。
- 新增 `tests/test_diaolong.py` 十项：全图发布契约、引用越界重试/停止、库外文献拒绝、literature_limit 裁剪联动引用投影、全角引用规范化、年份契约、Schema 字段描述、options 白名单与注册接线。research-agent 全量 189 项通过；新增文件 ruff 全部通过（test_foreign_integration.py 存在与本次无关的历史 lint 项，未改动）。
- 本机仅完成代码与测试验证；未重建/部署 Agent 镜像，未连接真实模型或业务知识库执行端到端运行。

## 2026-10-07 OCR 允许无文字页

- 移除“每个待 OCR 页面必须返回文字”的校验，允许空白页、纯插图页及整批无文字结果；文字块仍严格校验返回格式、原文页码和坐标。合并所有分片及本地文字后，整份附件无可索引文字才失败。
- 缓存支持页码子集和成功的空列表，避免空白分片在索引重试时重复 OCR；未返回 OCR 文字的候选页保留已有真实文字。保留原文页码，不为无文字页生成占位内容。
- 回归覆盖空白页定位、部分页缓存、空分片后续页识别、原生标题保留、整份无文字仍失败及错误结果仍拒绝。独立 Foreign/Index 未修改。
- Windows 本地及 Docker Linux/Python 3.9 回归各 50 项，49 项通过、1 项在线 OpenSearch 测试跳过。已通过 local-docker 构建并更新本地 Backend/worker，Backend 健康、八路 worker 正常、前端代理接口 HTTP 200；运行容器确认接受空 OCR 结果。部署前无在途索引任务，未自动重建业务文献。

## 2026-10-07 Backend Worker 恢复 MinerU OCR

- PDF 解析对扫描页调用官方 MinerU v4，支持混合 PDF、分片后的原文页码映射、归一化坐标转换和成功结果缓存；保留本地文字提取及现有双路索引流程。
- 旧 Token 已过期，改由 `MINERU_API_TOKEN` 配置；dev 启动及部署脚本透传 OCR 配置。未恢复硬编码密钥。
- 过期任务回收先检查 MySQL 文献命名锁，避免长时间 OCR 被误回收；不增加数据库字段。独立 Foreign/Index 未修改。
- Windows 隔离 Python 3.13 环境及 Docker Linux/Python 3.9 镜像回归均为 45 项，44 项通过，1 项 OpenSearch 在线测试因未配置独立地址跳过；覆盖真实生成 PDF + 模拟 MinerU HTTP、三次重试和命名锁回收。dev Shell 语法与配置覆盖检查通过；镜像测试使用断网容器。
- 按用户要求，本地 Backend 构建入口统一为 `Foreign/local-docker/Backend.Dockerfile`，配套 Dockerfile 专用排除规则避免打包旧镜像归档；`Foreign/compose.local.yaml` 已指向该文件。
- 已构建 `foreign-backend-local:dev` 并部署到 Docker Desktop，更新 `ForeignBackend-local` 和 `foreign-index-worker-1`，worker 按 Compose 配置启动八个消费进程。Backend 健康、容器源码摘要与工作区一致；前端和 Research Agent 未重建。旧 worker 无在途任务但未正常退出，核实队列后结束旧容器完成替换。
- 用户补充 Token 后，已在本地 Docker 用实际 MinerU 官方 v4 接口验证：临时生成两页混合 PDF（第一页文字、第二页纯扫描图），扫描页无文字层，中英文及编号 73921 识别正确，首次耗时 4.29 秒，缓存复用 0.02 秒；原文页码和坐标检查通过。只测试临时样例，未写业务索引、未重建历史失败文献、未部署远程服务器。

## 2026-10-03 不可访问知识库的历史清理

- Research Agent 删除单次运行与整个会话时，独立检查当前用户有效、记录属于本人且任务已结束，不再复用要求知识库权限的 read_run。知识库被删除或转属后，仍可清理自己的历史；结果、步骤、证据等读取权限保持原样。
- 前端四个研究页面及相关类型注释移除“永久/真删除”的错误表述，仍执行既有软删除。未修改数据库结构，也未删除实际业务记录。
- 新增回归覆盖：知识库删除/撤权后的本人任务与会话删除、结果读取仍拒绝、他人记录/停用或删除账号拒绝、PENDING/RUNNING 禁止删除。Agent 全套测试通过；前端类型检查及 35 项测试通过。服务器尚未更新，需提交并部署 Backend 下 Agent 与 Frontend 后生效。

## 2026-10-03 dev 停止超时调整

- deploy-dev.sh 将 Worker 停止等待及容器默认停止超时改为 60 秒，Backend 与 Research Agent 改为 15 秒；停止前打印容器名和等待上限。超时仍在执行的任务会被强制中断，索引任务沿用现有 30 分钟过期回收机制。
- 此调整仅针对原 dev 部署脚本，不变更本地 Compose。Shell 语法检查通过；尚未推送或部署服务器。

## 2026-10-03 Research Agent 服务器 CPU 兼容修复

- 服务器日志确认 Agent 导入 qdrant_client 间接加载 NumPy 时，因 wheel 要求 X86_V2 而启动失败并反复重启，导致研究接口 502。日志未报告实际 NumPy 版本，不推断具体版本号或 CPU 型号。
- requirements.foreign.txt 与 pyproject.toml 固定 numpy==1.26.4；Dockerfile.foreign 在构建时执行 pip check 与 NumPy/Qdrant 导入检查。Python 3.12 隔离容器实测 wheel 基线为 SSE/SSE2/SSE3，依赖检查通过，170 项测试及 7 个子测试通过。
- 仅修改本地源码，尚未推送或修复服务器运行容器；需服务器更新并重新构建部署，最终 CPU 兼容性以服务器导入检查与服务就绪结果为准。

## 2026-10-03 原 dev 脚本四服务部署

- 沿用现有 docker build/run 风格，未新增部署文件。Backend/deploy-dev.sh 构建 Backend 与 research-agent 镜像，创建共用 foreign-dev 网络，部署 ForeignBackend-dev（9503）、ForeignIndexWorker-dev（四进程、无端口）及 ForeignResearchAgent-dev（18000）。三个容器共享 start-dev.sh 导出的配置，直接连接既有服务器数据库、Qdrant 和 OpenSearch。
- start-dev.sh 支持仅加载配置、传入自定义命令和默认迁移后启动 Flask；保留原 dev 默认值并允许环境变量覆盖。部署先构建两个镜像并验证索引表迁移，再停止旧容器；Worker 停止宽限期现为 60 秒，Backend/Agent 为 15 秒。代码使用镜像内版本，保留 FileDev 附件挂载。
- Frontend/deploy-dev.sh 保留宿主机 npm 构建，加入同一网络并设置自动重启；default-dev.conf 将 /api/ 转发到 Backend，将 /research-api/ 转发到 Agent /api/v1/，启用动态 Docker DNS 与研究流式响应支持。正式部署脚本未修改。
- 在一次性无网络容器中通过三个 Shell 脚本语法、配置覆盖/命令透传和 nginx -t 检查；git diff --check 通过。未执行服务器部署。服务器更新代码后先运行 Backend/deploy-dev.sh，再运行 Frontend/deploy-dev.sh；需具备 Docker（Backend）及 npm/sudo Docker（Frontend）权限。

## 2026-10-03 排队显示排查

- 文献 3380 前两次 Embedding 返回 `Receive batching backend response failed!`，第三次于北京时间 16:48 成功，共 23 个片段。并非 Worker 未领取；待重试复用 PENDING 导致页面一直显示“排队中”。
- 索引状态响应新增 `retry_pending`，前端显示“等待重试”及已尝试次数；轮询调度移到 finally，临时请求失败不再永久停止刷新。
- 任务表审查待改进项：实际 `corpus_id` 被建为 AUTO_INCREMENT，但语义应为已有文献 ID；缺少心跳/租约，当前按 started_at 超过 30 分钟回收可能误判长任务；失败重试和首次排队共用 PENDING（本次仅改善展示）；仅保存当前任务，错误在下一次构建时被清空，没有持久尝试历史；过期扫描与领取在同一事务且没有跳过锁行，扩大并发后可能增加竞争。本次未改表或更换队列架构。

## 2026-10-03 索引 Worker 四进程并发

- 参考 EduResearch Celery Worker 的独立消费进程与单条预取设计，在现有 MySQL 队列上运行四个 spawn 子进程；不增加消息代理依赖。每个进程拥有独立客户端与数据库连接，保留原子领取、文献锁和 generation 发布校验。
- 新增 `--concurrency` / `INDEX_WORKER_CONCURRENCY`（默认 4），Compose 显式设置四路并发、init 和 10 分钟停止宽限期。正常停止等待在途任务；子进程异常使容器重启，遗留任务按原有超时规则回收。
- 新增真实多进程测试，覆盖四个不同 PID 同时运行、在途任务收尾、异常子进程退出传播及无效配置。隔离容器运行 Backend 测试共 32 项：31 项通过，1 项可选 OpenSearch 在线测试未配置而跳过；Compose 校验与 git diff --check 通过。
- 已重建镜像并仅替换 index-worker 容器，启动日志及进程列表确认四个消费子进程（slot 1–4）存活；后端验证码与研究就绪接口均返回 200。当前业务队列为空，没有额外提交付费索引重建；真实四路并行已由隔离测试验证。

## 2026-09-30 新版部署与“测试2”索引重建

- 已重建并替换 Docker Desktop 的 Backend、index-worker、research-agent。OpenSearch 固定为 2.9.0；Backend 与 Agent 健康检查通过，前端代理的研究就绪和验证码接口均为 HTTP 200。唯一任务表 `corpus_index_job` 的新增字段已完成生产库迁移。
- 重建范围为知识库 ID 5（名称“测试2”，owner_id=2）的 74 篇当前文献。全部任务均进入终态：58 篇 SUCCESS，共 2291 个 chunk；16 篇 FAILED，无 PENDING/RUNNING。临时增加的两个 Worker 已停止并自动移除，恢复一个常驻 Worker。
- 逐篇核验了数据库状态、附件指纹、构建版本和 Qdrant/OpenSearch 片段数量。58 篇成功文献两路数量一致且为当前版本；16 篇失败文献不可检索且两路均无残留。详细报告：`docs/test2_index_rebuild_20260930.json`。
- 失败原因：14 篇原附件由文件服务返回 404（79、316、317、318、326、346、347、348、349、1712、1774、1775、1776、1777）；2958 为无文字层扫描 PDF；3297 的附件 JSON 为 `[]`。这些任务已完成三次重试，需要补齐原文件、文字层/OCR 或附件后再索引。
- 3389 的 PDF 不被 pdfminer 正常打开，但 PyMuPDF 可读取其 10 页真实文字。为此增加仅在主解析器打开失败时触发的 PyMuPDF 回退，保持 0-based 内部页码和 PDF 左下原点坐标，不执行 OCR、不改写原文。该文献重试成功，生成 15 个片段。
- 最终 Backend 回归 25 项通过（26 项中 1 项可选真机测试未配置地址而跳过）；新 Agent 镜像 170 项及 7 个子测试通过。实际业务双路写入和逐篇计数验证已完成。
- 在已部署 Agent 中实际检索“生成式人工智能如何支持教师教学设计？”，58 个候选返回 5 条证据，包含同时具备 vector_rank 与 lexical_rank 的结果，真实引用页码检查通过；未调用回答生成 LLM。

## 2026-09-30 双路索引与混合召回

- 按用户要求只使用既有 `corpus_index_job`，没有新增第二张索引状态表。增加 `vector_status`、`lexical_status`、`build_id`、`lexical_index`、`chunk_count`，复用 generation、附件指纹和错误说明。两路都写入成功并通过发布前复核后，总状态才为 `SUCCESS`。
- `app/indexing/service.py` 统一控制重建、删除与成功发布；旧单篇及批量函数名保留为兼容入口。`opensearch_store.py` 管理 Foreign 独立版本化索引，正文和标题使用 CJK/standard 双字段，无须安装中文插件。
- 同一次分块写入 Qdrant 与 OpenSearch，共用整数 point ID、构建令牌、附件指纹和 SHA-256 正文哈希。payload 升级为 v4；任一附件无文本、向量数量不符、bulk 部分失败均阻止成功发布。既有整数 ID 编码最多支持 100 个附件/分片、每个 10000 个 chunk，超出时明确拒绝，防止覆盖相邻片段。
- MySQL 独立连接命名锁串行化同篇索引操作；过期任务回收递增 generation，旧执行者不得覆盖新任务状态。失败时撤销可检索状态并尽力清理两路，保留有限重试。
- Agent 增加只读 OpenSearch 召回，沿用 Foreign text-embedding-v4/1024 维向量空间。双语查询 → 两路授权过滤 → 版本/附件/哈希/页码复核 → RRF(k=60) → 确定性重排 → 绝对支持门槛。BM25 独有结果从 Qdrant 按 ID 获取正文，保留既有研究证据落库及引用协议。
- 查询后再次校验知识库权限与构建版本；外部检索故障显式失败。旧 is_vectorized=True 但未建 BM25 的文献不进入新版研究检索。普通 AI 搜索仍使用既有向量召回。
- Backend 启动幂等扩展任务表；Agent 的就绪检查确认字段已存在。Compose 已设置两侧共享的 OpenSearch URL/索引名及启动依赖。
- 新增 `scripts/backfill_hybrid_index.py`：默认预览，`--apply` 才入队，跳过已就绪及相同附件的活动任务。实际回填会重新解析并调用 Embedding。代码交付阶段未执行部署和回填；随后完成的部署及“测试2”重建结果见上节。
- 目录扫描批量脚本仍需匹配真实数据库文献及附件，不能用合成文献 ID 绕过校验；推荐使用数据库批量入口或上述回填脚本。
- 验证：隔离 Python 3.9 Backend 容器通过 25 项测试，含真实 Docker Desktop OpenSearch 2.9.0 的临时索引创建、中英文 BM25、批量可见性和删除；测试索引已清理。隔离 Python 3.12 Agent 容器通过 170 项测试及 7 个子测试。Agent Ruff、Compose 配置校验、git diff --check 通过。没有调用真实 Embedding 或执行业务 Qdrant/MySQL 的端到端重建。

## 2026-09-30 知识库索引状态与重建操作

- 文献列表新增索引状态及说明、单篇重建按钮和手动刷新；当前页有排队或处理中的任务时每 4 秒刷新，离开页面停止轮询。
- 列表批量读取既有 `corpus_index_job`。仅当前附件、目标索引及两路状态均符合成功条件时显示“已就绪”；旧向量标志不能单独代表索引成功。
- 新增 `POST /knowledge-bases/<base_id>/documents/<corpus_id>/reindex`，校验知识库归属、有效文献关联及附件来源；复用相同附件的活动任务，其他情况异步入队，返回 202。
- Backend 测试共 29 项，28 项通过、1 项可选在线测试跳过；前端类型检查、现有测试及生产构建通过。已更新本地 Backend、Worker 和 Frontend 容器，首页、研究就绪及验证码接口返回 200。
- 部署后只读核对“测试2”列表：62 篇，58 篇 SUCCESS、4 篇 FAILED；未为界面验证额外提交重建。浏览器没有已登录会话，未进行登录后的视觉验收。

## 2026-09-30 OpenSearch 本地部署

- `Foreign/compose.local.yaml` 新增 `opensearch` 服务，固定镜像 `opensearchproject/opensearch:2.9.0`，容器名 `ForeignOpenSearch-local`，使用独立持久化卷 `foreign-opensearch-data`。
- Docker Desktop 单节点运行，JVM 堆为 512 MiB，关闭本地开发实例的安全插件；宿主端口仅绑定 `127.0.0.1:19200`，Compose 网络内地址为 `http://opensearch:9200`。
- 已验证容器 healthy、HTTP 返回版本 2.9.0。原 EduResearch 2.19.1 容器及数据未改动。
- 该部署步骤仅启用索引基础设施；后续双路索引与混合召回代码及验证状态见上节。

## 2026-09-29 Research Agent 迁入

- 当前 Agent 维护入口为 research-agent/；由 Eduresearch/EduResearch/backend 复制 57 个有效文件，保留原源码、测试、数据集和构建配置，排除缓存。原目录暂留核对，请在新目录开发。
- Foreign/compose.local.yaml 和 EduResearch 历史 Compose 的 Agent 构建目录已指向新位置。
- Flask 后端和 Agent 继续使用各自 Dockerfile、Python 环境及 app 包；后端 .dockerignore 排除 research-agent/，避免重复打包。
- Agent 的独立 CI 已迁入 .github/workflows/research-agent.yml，测试须在 research-agent 目录单独执行，避免同名 app 包混用。
- 未修改业务代码、数据库或文件存储位置；启动迁移仍由 Agent Dockerfile 执行。

## 验证状态

- 复制时已逐文件核对 57 个有效文件；迁移后再次核对 54 个源码、测试、数据集与依赖文件，内容与原目录一致。README 增补新部署说明；Ruff 与 git diff --check 通过。
- 两份 Compose 的 Agent 构建路径均正确解析到新目录；统一 Compose 配置校验通过。
- Docker Desktop 已成功构建 Foreign 后端与 Agent 镜像；一次性容器确认 Foreign 镜像不含 research-agent 目录。
- Agent 在 Python 3.12 镜像中通过 157 项测试和 7 个子测试；Foreign 后端在 Python 3.9 镜像中通过 15 项测试。默认本机 Python 缺少依赖，改用 Docker 隔离测试，未修改全局 Python 环境。
- 用户随后要求重新部署：已从新目录重建镜像并替换 backend、index-worker、research-agent 容器；前端继续运行原镜像。部署前研究及索引任务均无 PENDING/RUNNING，启动迁移随容器正常执行。
- 部署后 backend 与 research-agent 健康检查通过，index-worker 正常运行；8002 首页、研究就绪接口直连与 Nginx 转发、验证码接口均返回 HTTP 200。未重新执行登录、上传及付费模型完整问答流程。


### 2026-10-03 删除知识库后的历史查看

四个研究模块允许用户查看本人已保存的历史结果和证据，不再以知识库存在作为历史读取前提。启明在知识库删除后禁用输入和发送；创建运行接口仍校验知识库，禁止继续提问。前端移除历史打开拦截并显示仅查看历史提示。通过隔离数据库和前端生命周期测试验证，不修改业务数据。
# 2026-10-20 文献问答个人 PDF 与混合分析

- 数据模型沿用 `corpus`：新增整数状态 `4`（个人上传）和可空 `owner_id`；原有非个人文献保持 `owner_id = NULL`。个人文件状态不进入公共审核、文献管理、搜索、统计或推荐。所有者只从登录身份写入，客户端无法指定。
- Flask 新增 `app/api/literature_files.py`，当前仅收集者角色可用。`POST /literature/files` 接收单个 multipart `file`，仅支持不超过 50 MB 的 PDF，核对扩展名与文件头；内容解析失败由异步索引状态反馈。文件保存到既有文件服务，使用随机文件名保留原名；同一数据库事务保存附件并登记 `corpus_index_job`，上传失败回滚，提交失败尽力补偿清理附件。
- `GET /literature/files` 为本人文件，`GET /literature/library` 为公开已审核候选，支持 `q/page/page_size`。`GET /literature/documents/<id>` 统一授权读取题录及双路状态；`GET /literature/documents/<id>/file?attachment_path=...` 校验所有权和当前附件清单，私有原文禁止公共缓存。`POST /literature/files/<id>/retry` 复用活跃任务；`DELETE /literature/files/<id>` 先软删除并撤销任务代次，再尽力清理双路索引和原件，返回 `cleanup_pending`；清理未完成可重复 DELETE，当前没有自动清理补偿队列。
- Worker 的构建、发布资格统一为“公共已审核且无所有者，或有有效所有者的个人文件”；保留附件指纹、generation、双路发布和删除校验。Qdrant 新片段带 `personal_upload` 标记，旧公共检索先排除个人片段，再回库复核。`Index/` 和 `ProductionWorker/` 已同步资格与载荷，保留其队列、扫描、迁移及生产配置差异。
- Research 阅读授权、冻结范围、逐篇材料读取、追问引用发布与证据链接支持“公共已审核 + 本人个人文件”，继续使用统一 `corpus_ids`，最多 20 篇。个人材料的范围指纹额外包含来源和所有者，公共材料保持旧版指纹兼容；运行期间删除、撤权或材料版本变化会拒绝发布或继续读取原件。历史消息仍按会话所有者保留，删除原件不等于删除既有回答快照。
- 旧管理/编辑/下载入口排除个人文件；旧 NL2SQL 的 corpus 表读取注入公共审核子查询，拒绝非 SELECT、多语句、注释和库名前缀，避免聚合数量泄漏私有记录。旧按路径读取片段接口也必须先匹配当前可读文献的真实附件。
- 前端 `LiteratureDetail.vue` 加入“上传本地 PDF”“我的文件与选文”抽屉；个人与库内材料可以混选，显示排队、解析、失败、重试和可分析状态。默认等待全部所选材料就绪，也可明确仅分析已就绪部分；调整材料范围开启新分析，不改变旧会话范围。预览、下载统一走授权接口。
- 部署顺序：先在配置好的 Backend 环境执行 `python -m scripts.migrate_personal_files`，再更新 Backend/Worker、Research Agent 与 Frontend。迁移幂等增加 `corpus.owner_id BIGINT NULL` 和索引，不改旧文献数据。dev 已由 Worker `--migrate-only` 接入；正式 `start.sh` 启动 Flask 前执行迁移，失败停止启动。独立 ProductionWorker 仍不自行改表，须先执行上述 Backend 迁移。Agent 主表映射依赖该字段，不能先于迁移切换新 Agent。
- 验证范围：后端离线共 112 项，111 项通过、1 项未配置在线 OpenSearch 跳过；前端现有 78 项测试、研究类型检查、生产构建通过。Research Agent 的完整离线检查发现 4 项原有 `test_agent_gateway.py` 用例失败：HTTPX 测试替身未覆盖既有 SDK 思考流式分支，尝试访问模型地址后被无网络环境拒绝；本次未修改网关或这组测试。其余研究测试 291 项通过，追加公共会话版本兼容用例后，文献阅读专项 36 项通过；修改的研究代码 Ruff 检查通过。独立 Index 全套 30 项、ProductionWorker 消费专项 16 项通过。Playwright 在 1440×1000 与 390×844 下使用真实构建产物及模拟 API，验证上传排队、就绪后混合选择、分析请求、结果、PDF 预览和手机控件；控制台无错误或警告。首次实现阶段未执行真实文件服务/模型联调、运行数据库迁移或服务部署；随后完成的本地迁移与部署见下条记录。
- 2026-10-20 后续本地部署：已通过 `compose.local.yaml` 和 `local-docker` 的构建配置更新 Docker Desktop 中的 Backend、八进程 Worker、Research Agent 与 Frontend；前端镜像内类型检查、78 项测试和构建通过。迁移已在既有 `foreign_language_test` 测试库执行，`corpus.owner_id` 为可空 BIGINT 且索引存在，非个人记录 owner_id 非空数量为 0。容器源码摘要与工作区一致；前端/业务代理/研究就绪返回 200，个人文件接口未登录返回 401。部署前后索引任务维持 SUCCESS 86、FAILED 17，没有创建上传或重建业务索引。旧镜像回滚标签后缀 `before-personal-files-20261020-151123`，报告位于工作区 `outputs/personal-files-20261020/`；详细操作见 `local-docker/README.md`。真实文件上传与模型分析联调本次未执行。


## 2026-10-20 个人题录由 Backend 同步提取

- 普通采集与个人文件统一由 Backend 请求进程调用 app/utils/literature_metadata.py 的提取逻辑。题名、关键词、摘要等保存到 corpus。Worker 只执行正文解析、必要 OCR 和双路索引，不提取题录。
- 个人上传先将文件、corpus 和 corpus_index_job 同事务持久化；job 暂为 EXTRACTING，Worker 不会领取。Backend 同步提取后复核所有者、generation、附件指纹和撤销状态，将题录与待领取状态在同一事务保存。本地待领取状态仍为 PENDING_TOKENS，不增加数据库列或异步框架。
- 模型失败保留原件并标记 FAILED，由上传接口返回可重试材料；重试接口也在 Backend 同步提取。活动请求复用，Backend 崩溃留下的 EXTRACTING 超过十五分钟后可手动重试。完整题录的索引重试直接排队，不重复调用模型。页面将准备态投影为正在解析。
- 共享采集提取保留前五页/最多五千字快速提取、整文件回退和 JSON 校验。关键词按列容量保存完整 JSON 数组，原始文件名仍保留。ProductionWorker 已撤回题录调用及额外 OpenAI 依赖，保留其队列和进程回收策略；本轮未调整 Index 目录。
- scripts/backfill_personal_metadata.py 默认预览；--apply 在 Backend 运行环境逐篇同步提取，成功后再排队，不能将缺题录文件直接交给 Worker。
- 修正前经用户明确授权处理的 collector 3444–3450 七篇，题录与双路索引已全部成功（attempts=1）。本次保留这些数据，不重复调用模型；历史会话仍需重新分析才使用新题录。此前 Worker 提取版本已被本节 Backend 同步方案取代。
- 验证：Backend Python 3.9 全套 123 项，122 通过、1 项在线 OpenSearch 跳过；其中个人材料/题录专项 23 项验证了同步请求、准备任务不可被 Worker 领取、失败保留原件及重试、所有者与版本竞态、Worker 不调用题录模型。ProductionWorker 消费专项 16 项通过。

- 已更新 Docker Desktop 的 Backend 与本地 Worker，运行源码与工作区一致；Backend 路由同步题录调用、实际 Worker 无题录调用，HTTP 健康检查通过。此前七篇关键词、摘要与 SUCCESS 索引全部保留，没有重新调用模型。部署验收见 outputs/personal-files-20261020/backend-process-deployment-verification.json。


### Worker 题录代码清理

题录业务编排迁至 Backend 专用 app/services/personal_metadata.py；删除旧 app/indexing/metadata.py。Worker 的题录调用、专用模块与新增 OpenAI 依赖均已撤回，正文读取入口恢复原实现。Backend 的个人 PDF 下载临时文件仅在同步题录服务内处理，不再改动 Worker 原件读取函数。ProductionWorker 保留其已有正文解析、OCR、索引、队列和进程回收逻辑；本轮未调整 Index 目录。

- 清理后 Python 3.9 容器验证：Backend 123 项（122 通过、1 跳过）、ProductionWorker 消费专项 16 项通过，Ruff 与差异检查通过。本地 Backend/Worker 已重新部署，旧模块不存在、Worker 无题录调用、原读取入口恢复，HTTP 健康正常，原七篇题录和索引结果完整保留。报告位于 outputs/personal-files-20261020/worker-cleanup-*.json。
