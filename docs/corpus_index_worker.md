# Foreign 文献审核后自动索引

## 流程

1. 采集者上传附件时，Backend 把文件转发给 `FOREIGN_FILE_URL` 指向的独立文件服务，并把附件路径写入 `corpus.attachment`。文件服务负责存储，不是 Backend 的索引 Worker。
2. 文献保持待审核状态；通过普通或恢复文献审核时，Backend 在同一 MySQL 事务中将 `corpus.status` 改为已通过，并在 `corpus_index_job` 登记 `PENDING`。该表包含 `is_deleted` 软删除字段。
3. `index-worker` 轮询任务，从文件服务取得原文。PDF 优先使用 `pdfplumber` 提取词框并恢复阅读顺序；脚本部署默认对扫描页调用 AICP `qwen/qwen-vl-ocr-2025-11-20`，保留原文页码和明确的定位精度；DOCX、TXT 提取文字。同一次分块生成 Qdrant 向量和 OpenSearch BM25 文档，共用 point ID、`build_id`、附件指纹和正文哈希。两路写入完成并可查询后，再次检查审核状态、附件及任务 generation，才将任务标记 `SUCCESS` 并设置 `corpus.is_vectorized=True`。
4. 附件变更会取消旧任务并使旧片段失效。公开片段检索还会从 MySQL 复核文献未删除、已审核和索引完成，避免旧片段或处理中的片段被返回。

## PDF 阅读顺序

原生文字不再直接按整页同高度合并成行。`app/utils/pdf_layout.py` 从词框寻找多行稳定的栏间空白，
在每个版面区域内按左栏、右栏读取。跨栏标题/正文、有边框表格、与正文明显分隔的页眉/页脚
保持纵向位置；缺少足够分栏证据的页面保留逐行读取。pdfplumber 与 PyMuPDF 备用路径共用规则，
保留真实行框和原附件页码。原文片段查询采用相同顺序，字符偏移对齐新的分块正文。

OCR 返回可靠独立行坐标时也恢复分栏顺序；只有页/区域定位的结果不猜测行位置或改写文字。
OCR 不再读取或写入缓存，每次任务按当前 provider 和模型重新识别。复杂无边框表格、非标准版式或 OCR 已合并的
整页文字仍需逐例核查，不能仅凭此规则保证所有版式都正确。

修改解析顺序不会自动改写已存储片段，需要重建对应文献的 Qdrant 和 OpenSearch 索引。
测试知识库可用 `scripts/rechunk_test_knowledge_base.py rebuild --base-id 11 --corpus-id 223
--corpus-id 3400 --workers 1 --report <path>` 限定重建范围；脚本保留测试库和存储目标校验。

## 运行与排障

任务领取先无锁读取最多 32 个候选，再按主键有条件地将仍到期的待处理任务更新为 `RUNNING`，只有更新成功的进程才能领取。过期任务同样逐条复核 generation、状态、过期时间和文献命名锁，并逐条提交，避免批量回收与范围锁定领取在同一事务中产生死锁。MySQL 死锁（1213）和锁等待超时（1205）最多尝试三次，每次使用新的应用上下文和随机短暂退避；其他数据库错误直接交给原轮询错误处理。该实现不依赖 `SKIP LOCKED`，无需调整表结构。

Worker 默认以八个独立进程同时处理八篇文献，参考 EduResearch 的进程隔离与单条预取方式；保留 MySQL 任务队列，无需额外部署 Celery/RabbitMQ。`--concurrency 8` 优先于环境变量 `INDEX_WORKER_CONCURRENCY`（默认 8）。每个进程逐条原子领取任务，沿用文献命名锁和 generation 校验；spawn 启动隔离数据库连接、PDF 解析器与索引客户端。单篇文献内部仍顺序解析和写入，八路并发提高批量吞吐，不保证单篇提速。

消费进程每处理完一篇文献（成功、取消或登记重试）就正常退出，释放该进程的解析器、分词器及 Python 内存分配器驻留内存。父进程只补齐退出的槽位，其他进程继续处理各自任务；空闲进程保持轮询，不按空轮询反复重启。日志会记录文献任务结束、旧进程回收和新 PID 启动。该机制与 ProductionWorker 一致，不限制单篇处理峰值，八路同时工作的内存仍需计入容量。

父进程收到 SIGTERM/SIGINT 后停止领取新任务，等待在途任务结束。部署脚本为 Worker 设置 `--init` 和 60 秒停止宽限期；超时会强制终止进程，未完成任务沿用原有 30 分钟过期回收机制。子进程异常退出时监督进程非零退出，由 Docker 重启。

服务器运行 `sh deploy-dev.sh` 部署 Backend、八进程 Worker 和研究服务，使用已配置的 Qdrant/OpenSearch。Backend 启动时幂等创建/升级 `corpus_index_job`，Worker 以独立进程运行。只使用这一个索引状态表，新增 `vector_status`、`lexical_status`、`build_id`、`lexical_index`、`chunk_count`，复用现有 `error_message` 描述失败。旧行的两路状态默认 `PENDING`，不能误判为混合索引就绪。

任务总状态沿用 `PENDING/RUNNING/SUCCESS/FAILED/CANCELLED`。分支状态使用 `PENDING/RUNNING/SUCCESS/FAILED`；由于两路按批交错写入，完整文献写完前两路都保持 `RUNNING`，失败补偿后均为 `FAILED`。只有两路都为 `SUCCESS` 且最终版本复核通过，总状态才能为 `SUCCESS`。

`app/indexing/service.py` 统一编排重建、清理和发布；`qdrant_utils.py` 保留旧函数名作为兼容入口，单篇/批量入口也走同一服务。MySQL 命名锁覆盖同篇文献整个重建周期，过期领取会递增 generation，防止旧执行者发布或误删新快照。SQLite 仅用于离线测试，不提供生产跨进程命名锁。

`GET /ai_search/internal/corpus/<id>/vector_status` 返回向量化标志、任务状态、两路状态、构建版本、片段数、尝试次数和错误信息。任务失败最多尝试三次；耗尽后保留 `FAILED` 与错误信息供排查。索引开始前会删除该文献旧片段，所以重建期间该文献暂不可被片段检索返回。某个附件没有文本、向量数量不匹配或任一 bulk 子项失败，均不允许部分成功。

## 历史回填与部署顺序

先启动 OpenSearch，再构建和启动新版 Backend/Worker（先完成表字段迁移），最后启动新版 Agent。旧版本 Agent 仍可使用已就绪的向量数据；新版 Agent 要求任务表两路均成功。

在 Backend 工作目录运行：

```bash
python -m app.workers.corpus_index_worker --migrate-only
python -m scripts.backfill_hybrid_index --limit 20
python -m scripts.backfill_hybrid_index --apply --corpus-id 42
```

回填默认只预览，`--apply` 才登记任务；执行时跳过已就绪及同附件正在处理的任务。
Worker 会重新解析并调用 Embedding，回填会产生模型调用。移除 `--corpus-id` 可处理全部候选。
索引版本变化必须使用新的 `FOREIGN_OPENSEARCH_INDEX` 并重新回填，不原地修改分析器。
部署不会自动登记所有历史文献回填。2026-09-30 按用户请求已重建“测试2”（ID 5）：58/74 篇成功，2291 个片段；16 篇因原文件缺失、空附件或无文字层失败，见同目录重建报告。

可用 `FOREIGN_TEST_OPENSEARCH_URL=http://opensearch:9200 python -m unittest discover -s tests -p test_opensearch_live.py` 执行真实 2.9.0 兼容性测试；该测试只操作自身随机命名的临时索引，完成后清理。

## AICP 扫描 PDF 识别

`start-dev.sh` 默认设置 `FOREIGN_OCR_PROVIDER=aicp`，`deploy-dev.sh` 将 OCR 配置传入容器。
脚本已配置独立的 `FOREIGN_OCR_API_KEY` 默认值，直接执行 `sh deploy-dev.sh` 即可加载；部署环境可覆盖该变量。最终密钥为空时，部署脚本在构建和替换旧容器前退出。
该密钥与百炼 Embedding、业务问答、Research Agent 的配置独立，不使用其他模型密钥兜底。

| 环境变量 | 用途 / 默认值 |
| --- | --- |
| `FOREIGN_OCR_PROVIDER` | 脚本默认 `aicp`；也支持百炼 `qwen`、官方 `mineru`；直接运行 Python 未配置时沿用 `qwen` |
| `FOREIGN_OCR_BASE_URL` | `https://aicp.ccnu.edu.cn/v1`，SDK 基础地址，不包含 `/chat/completions` |
| `FOREIGN_OCR_API_KEY` | `start-dev.sh` 已配置独立 OCR 密钥，支持环境覆盖 |
| `FOREIGN_OCR_MODEL` | `qwen/qwen-vl-ocr-2025-11-20` |
| `FOREIGN_OCR_DPI` / `FOREIGN_OCR_TIMEOUT` | `150` / `120` 秒；范围分别为 72..300 / 1..600 |
| `FOREIGN_OCR_CONCURRENCY` | 每篇文献最多四个 HTTP 请求并发，范围 1..8；八个 Worker 的总并发可能达到 32，应按令牌分组的 RPM/TPM 调整 |

新接口使用已有 OpenAI SDK，同一文档共享连接，文档完成后关闭客户端。扫描页按现有规则渲染为
PNG，文字输出转换为既有 `text/page_idx/bbox/block_id/ocr_model/location_precision` 契约。
整页识别使用真实页边界和 `location_precision=page`；输出截断时递归切分实际图像区域，保留原页码，
使用 `location_precision=region`。不推算文字行坐标，因此不提供旧高精模式的行级定位保证。

网络错误、408/409/429 和 5xx 在请求层最多尝试三次；SDK 自动重试关闭，Retry-After 最多等待 30 秒。
401/403 不在请求层重试，原始远程错误及密钥不写入任务错误字段。超过截断恢复上限、异常响应或
非正常结束均明确失败。输出上限为 8192 token；空白页在重复任务中也重新识别，混合 PDF 保留其他原生文字，
整份附件无可索引文字时仍失败。文献级 Worker 重试机制继续沿用原逻辑。

HTTP 400 明确返回 `data_inspection_failed`（或 `DataInspectionFailed`）时，仅跳过被拒绝页的 OCR，
继续解析其他页并保留原页码。剩余正文完成双路索引后任务仍为 `SUCCESS`，跳页及脱敏原因直接
保存在 `corpus_index_job.error_message`，成功收尾不会清空；下一次构建会重新收集，不沿用旧警告。
分区识别遇到该拒绝时舍弃整页 OCR，不发布半页结果。其他 HTTP 400、鉴权、网络及截断错误仍按原规则处理。

非成功 HTTP 响应会记录模型、输出上限、请求图片尺寸、尝试次数及脱敏后的上游
`type/code/param/message` 和请求 ID。最终异常附带原附件页号、provider、模型与页面尺寸，
沿用原有流程写入 `corpus_index_job.error_message`。不记录原始图片、请求正文、密钥或未筛选的响应正文。

切换只影响后续新建或重建任务，不自动回填历史索引。回退百炼时同时设置 provider 为 `qwen`、
model 为 `qwen-vl-ocr`；回退 MinerU 时设置 provider 为 `mineru` 和有效的 `MINERU_API_TOKEN`。

## MinerU 扫描 PDF 识别

设置 `FOREIGN_OCR_PROVIDER=mineru` 后，Backend 解析层使用 MinerU 官方 v4 文件上传接口；使用现有 requests/PyMuPDF，
不安装本地 OCR 模型或 GPU 运行时。Backend worker、单篇和批量入口共用此解析层。
独立 `Foreign/Index` 不在本次修改范围内。

普通文字 PDF 继续本地提取。pdfplumber 无法打开时改用 PyMuPDF，两条路径均检测
图片扫描页：无文字且有图片，或少于 80 字且单张图片占页面面积至少 50%。
纯空白页与正常文字页共存时跳过；整份无文字且没有候选图片页时尝试全部页面。
这是扫描页检测启发式，不保证发现所有图文混排/损坏文字层。图片中只有图形没有文字时，
OCR 也可能无法生成可索引内容。

只上传候选页，每批默认最多 200 页；分片响应页码映射回原附件，不产生虚拟附件路径。
content_list 的 0..1000 左上原点 bbox 转成 PDF 左下原点坐标，OCR 高亮精度为块级。
保留表格文字、列表、公式和图片标题。OCR 成功完成后，允许个别页面或整个分片没有文字，
不生成占位文字块，不压缩原文页码；若候选页原本有少量真实文字，OCR 未返回文字时保留原文。
每次任务重新识别候选页，不复用成功或无文字分片。仅在合并本地文字与所有 OCR 分片后，整份附件仍无可索引文字时
报错；接口失败、结果格式或文字块页码/坐标异常仍明确失败。OCR 失败沿用现有三次重试和双路清理流程。

| 环境变量 | 默认值 / 用途 |
| --- | --- |
| `MINERU_API_TOKEN` | 必须提供有效官方 Token；旧代码中的 Token 已过期，不恢复硬编码密钥 |
| `MINERU_BASE_URL` | `https://mineru.net/api/v4`；须兼容官方 v4 上传、轮询和结果格式 |
| `MINERU_MODEL_VERSION` | `vlm`，也支持 `pipeline` |
| `MINERU_LANGUAGE` | `ch`，可按文献语言调整 |
| `MINERU_TIMEOUT` | `1200` 秒，整个附件本次 OCR 的总时间预算，包含所有分片 |
| `MINERU_MAX_PAGES` | `200`，范围 1..600；单个上传分片仍不得超过 200MB |

`is_ocr=True` 位于每个上传文件对象中；表格和公式识别开启。API Token 不传给上传和下载
存储地址。OCR 缓存机制已移除；Embedding/索引写入失败后重试也会重新识别候选页，
不读取旧缓存，不创建新的缓存文件，也无需缓存目录或持久卷配置。

运行 Backend worker 的容器需能访问 MinerU API 及其返回的上传/下载地址，环境变量必须
传入 worker 容器。`start-dev.sh` / `deploy-dev.sh` 已透传以上配置；其他容器启动方式应
自行传入 `MINERU_API_TOKEN` 等变量。例如在设置有效 Token 后运行原 `deploy-dev.sh`。
本地原文件目录继续由 `INDEX_FILE_BASE_DIR` 配置，缺少挂载时仍从文件服务下载。

MySQL 回收超过 30 分钟的 RUNNING 任务时，会先确认对应文献命名锁无人持有，避免长时间
OCR 仍在运行却被递增 generation。进程退出或连接断开会释放锁，遗留任务仍可恢复。
SQLite 离线测试保留原过期规则。没有新增任务表或修改数据库字段。

离线回归：`python -m unittest discover -s tests -v`。真实识别仍需配置有效 Token 后，
通过既有单篇重建入口对扫描件执行验收；代码更新不会自动重建历史失败任务。
