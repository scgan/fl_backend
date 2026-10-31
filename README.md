# Foreign Backend

Foreign 多语种教育文献数据库的后端，提供账号与权限、文献采集和审核、附件访问、统计、知识库、AI 搜索及文献问答。业务接口由 Flask 提供；独立索引 Worker 负责解析附件并写入检索索引；`research-agent/` 提供智能研究 API。

配套前端见 [Frontend README](../Frontend/README.md)，变更记录见 [project.md](project.md)。

## 服务与技术栈

| 服务 | 技术 / 入口 | 职责 |
| --- | --- | --- |
| 业务后端 | Python 3.9（当前 Docker 镜像）、Flask、SQLAlchemy；`app/app.py` | 账号、文献、审核、知识库、普通 AI 搜索与问答 |
| 索引 Worker | `python -m app.workers.corpus_index_worker` | 消费 MySQL 任务，解析、OCR、分块，写入 Qdrant 与 OpenSearch |
| Research Agent | 独立 Python 3.12 环境、FastAPI、LangGraph；`app.foreign_integration.main:app` | 启明、智鉴、溯知、融通及全局搜索文献阅读 |
| 外部依赖 | MySQL、独立文件服务、Qdrant、OpenSearch、百炼 API | 业务数据、附件、向量 / BM25 检索和模型调用 |

业务后端和 Agent 共用 Foreign 数据库与 JWT 密钥。附件上传由后端转发给 `FOREIGN_FILE_URL` 对应的独立文件服务。索引任务使用 `corpus_index_job`，无需额外部署消息队列。

## 目录结构

```text
Backend/
├── app/
│   ├── app.py                  # Flask 入口、蓝图注册和身份校验
│   ├── config.py               # 数据库、JWT、文件服务配置
│   ├── models.py               # 业务与索引任务模型
│   ├── api/                    # 各业务模块接口
│   ├── indexing/               # 双路索引编排、任务状态、OpenSearch
│   ├── workers/                # 索引消费与进程管理
│   └── utils/                  # PDF/OCR、分块、Embedding 等工具
├── research-agent/             # 独立研究服务、依赖与测试
├── docs/                       # 索引、分块、问答接口文档
├── scripts/                    # 回填、重建与数据维护工具
├── tests/                      # Flask / 索引离线回归
├── requirements.txt
├── Dockerfile / Dockerfile.dev
├── start.sh / start-dev.sh
└── deploy.sh / deploy-dev.sh
```

`research-agent` 与 Flask 后端都包含名为 `app` 的包，必须在各自目录和独立 Python 环境内安装、启动及测试。

## 一键脚本部署

在 Linux 服务器使用现有 `deploy-dev.sh` 部署完整业务后端、索引 Worker 和研究服务。

### 部署前准备

- 服务器具备 Docker 与构建、运行容器的权限；部署前端还需 Node.js 22、npm 和 sudo Docker 权限。
- MySQL 已有 Foreign 业务表和可登录账号，文件服务、Qdrant、OpenSearch 及模型 API 可从容器访问。
- 核对 `start-dev.sh` 中的数据库、JWT、附件、模型及索引配置；这些变量允许通过部署终端的环境变量覆盖。
- 默认 OCR 使用 AICP 网关；`start-dev.sh` 已配置独立的 OCR 地址、模型和 `FOREIGN_OCR_API_KEY`，可直接执行部署脚本，环境变量可覆盖默认配置。
- 确认服务器附件目录 `/home/ccnu/Foreign/FileDev`；脚本将其挂载到 Backend 和 Worker 的 `/foreign-files`，目录不同时需调整脚本中的挂载路径。

启动迁移只创建 / 升级索引任务表与研究表，不负责完整业务库初始化。密码、JWT 密钥和 API Key 使用部署环境自己的值。

### 执行部署

在 `Backend/` 目录执行：

```bash
sh deploy-dev.sh
```

脚本会构建 `Dockerfile.dev` 和 `research-agent/Dockerfile.foreign`，创建 `foreign-dev` 网络，先验证索引任务表迁移，再替换 Backend、Worker 和 Agent 容器。迁移失败时不会执行旧容器删除；Agent 启动时执行研究表迁移。

后端完成后，在 `Frontend/` 目录执行其 `sh deploy-dev.sh`。也可以在 `Foreign/` 目录依次执行两份脚本：

```bash
sh Backend/deploy-dev.sh
sh Frontend/deploy-dev.sh
```

脚本自动完成镜像构建和容器更新，无需手动安装后端 Python 依赖或分别启动三个后端进程。

| 服务 | 容器名 | 服务器端口 / 入口 |
| --- | --- | --- |
| 业务后端 | `ForeignBackend-dev` | `9503` → 容器 `5000`；直接请求时不加 `/api` |
| 索引 Worker | `ForeignIndexWorker-dev` | 无 HTTP 端口，默认八进程 |
| Research Agent | `ForeignResearchAgent-dev` | `18000`；API 前缀 `/api/v1` |
| 统一前端 | `foreign-frontend-dev` | `8002`；浏览器访问 `http://服务器地址:8002` |

更新已有后端时重新执行 Backend 的 `deploy-dev.sh`；只更新前端时执行 Frontend 的同名脚本。停止时 Worker 最多等待 60 秒完成在途任务，Backend 和 Agent 最多等待 15 秒。

### 查看运行状态

```bash
docker ps --filter network=foreign-dev
docker logs --tail=100 ForeignBackend-dev
docker logs --tail=100 ForeignIndexWorker-dev
docker logs --tail=100 ForeignResearchAgent-dev
```

Agent 的存活检查为 `/api/v1/health`，就绪检查为 `/api/v1/health/ready`；通过统一前端访问时分别使用 `/research-api/health` 和 `/research-api/health/ready`。

`deploy.sh` / `start.sh` 是正式入口：部署 Flask（宿主端口 `9502`）、研究服务（默认 `18001`）
和八进程索引 Worker（不发布宿主端口），沿用 `corpus_chunks_prod_v1` / `foreign_chunks_prod_v1`。
脚本在替换容器前完成业务与研究表迁移，不自动回填索引。
生产配置、前端代理、Worker 选择与回退见 [正式部署说明](docs/production-deployment.md)。

## 主要配置

配置由 `start-dev.sh` 加载，`deploy-dev.sh` 按脚本中的 `-e` 列表传入容器。已有变量可在执行脚本前通过 `export` 覆盖；新增配置需要同步检查脚本是否透传。下面列出主要配置含义：

| 环境变量 | 用途 / 默认值 |
| --- | --- |
| `FOREIGN_SECRET_KEY` | JWT 签名与验证；Backend 和 Agent 必须一致 |
| `FOREIGN_USERNAME` / `FOREIGN_PASSWORD` / `FOREIGN_HOST` / `FOREIGN_DBNAME` | MySQL 连接；必须指向已有业务库 |
| `FOREIGN_FILE_URL` | 独立文件服务基础地址 |
| `QDRANT_HOST` / `QDRANT_PORT` | Backend/Worker 的 Qdrant 地址；按部署环境设置 |
| `FOREIGN_QDRANT_URL` | Agent 的 Qdrant 地址；未设置时由上述 host/port 组合 |
| `FOREIGN_QDRANT_COLLECTION` | 向量集合，默认 `corpus_chunks`；两侧需一致 |
| `FOREIGN_OPENSEARCH_URL` / `FOREIGN_OPENSEARCH_INDEX` | OpenSearch 地址与索引名；索引名默认 `foreign_chunks_v1`，两侧需一致 |
| `DASHSCOPE_API_KEY` / `QWEN_API_KEY` | Embedding 和 Qwen OCR 优先读取前者，兼容后者；普通业务 AI 接口使用 `QWEN_API_KEY` |
| `FOREIGN_BAILIAN_API_BASE` | 百炼原生 API，默认 `https://dashscope.aliyuncs.com/api/v1` |
| `FOREIGN_EMBEDDING_MODEL` / `FOREIGN_EMBEDDING_DIMENSION` | 默认 `qwen3.7-text-embedding-flash` / `1024`；索引与召回配置一致 |
| `FOREIGN_EMBEDDING_BATCH_SIZE` | 每批 Embedding 数量，默认 `20` |
| `QWEN_BASE_URL` / `LLM_MODEL_NAME` / `QA_MODEL_NAME` | 普通 AI 的兼容接口、搜索回答模型（默认 `qwen3.8-flash`）和旧文献问答模型 |
| `OPENAI_API_KEY` / `OPENAI_BASE_URL` / `MODEL_NAME` | Agent 的回答生成配置；正式 `start.sh` 默认 `qwen3.8-max`，`start-dev.sh` 默认 `qwen3.8-flash` |
| `FOREIGN_OCR_PROVIDER` | `start-dev.sh` 默认 `aicp`；也支持 `qwen` 和 `mineru`。直接运行 Python 未设置时仍使用 `qwen` |
| `FOREIGN_OCR_BASE_URL` / `FOREIGN_OCR_API_KEY` | AICP 专用配置；默认基础地址 `https://aicp.ccnu.edu.cn/v1`，部署脚本已配置独立 OCR 密钥，可通过环境覆盖 |
| `FOREIGN_OCR_MODEL` / `FOREIGN_OCR_DPI` / `FOREIGN_OCR_TIMEOUT` | AICP 默认模型 `qwen/qwen-vl-ocr-2025-11-20`；DPI `150`、请求超时 `120` 秒；百炼模式使用 `qwen-vl-ocr` |
| `FOREIGN_OCR_CONCURRENCY` | 单文档 OCR HTTP 并发默认 `4`（范围 1–8） |
| `INDEX_FILE_BASE_DIR` | 容器附件挂载目录，默认 `/foreign-files`；文件不存在时走文件服务 |
| `INDEX_WORKER_CONCURRENCY` | Worker 默认八进程；`--concurrency` 参数优先 |
| `FOREIGN_INDEX_QUEUE_STATUS` | 队列待处理状态，默认 `PENDING`；API 和 Worker 需一致 |
| `FOREIGN_PUBLIC_API_BASE` | Agent 返回附件链接的基础地址；统一前端使用 `/api` |

全局搜索的 AI 增强搜索使用 `LLM_MODEL_NAME`。Python 源码、正式与 dev 启动/部署脚本及批处理
脚本默认统一为 `qwen3.8-flash`。上传服务器后，若原有环境显式设置过旧模型，先执行
`export LLM_MODEL_NAME=qwen3.8-flash`，再运行对应的 `deploy.sh` 或 `deploy-dev.sh`；已有容器需
重新创建才能更新环境变量，仅重启不会重新读取部署配置。若另有 `.env` 或 Compose 配置，
也应将其中的 `LLM_MODEL_NAME` 同步更新。该配置与 Agent 使用的 `MODEL_NAME` 分别生效。

百炼原生 API 与生成回答使用的 OpenAI 兼容 API 路径不同，应分别配置。MySQL 端口在 Flask 的 `app/config.py` 中固定为 `3306`。更多研究模块与检索说明见 [Research Agent README](research-agent/README.md)。

AICP 通过 OpenAI SDK 调用，`FOREIGN_OCR_BASE_URL` 填写 `/v1` 基础地址，不追加 `/chat/completions`。它不复用 Embedding 或研究模型的密钥。输出上限为 8192 token；纯文字结果标记为页级定位，截断时分区域重试并标记区域级定位，不生成文字行框。AICP、百炼和 MinerU 均已移除 OCR 缓存，每次需要 OCR 时重新识别；历史索引不会因切换配置自动重建。

切回百炼时同时设置 `FOREIGN_OCR_PROVIDER=qwen` 和 `FOREIGN_OCR_MODEL=qwen-vl-ocr`；MinerU 模式使用 `FOREIGN_OCR_PROVIDER=mineru` 及有效的 `MINERU_API_TOKEN`。

## API 与认证

前端 `/api/*` 代理到 Flask 时会移除 `/api`，`/research-api/*` 则重写为 Agent 的 `/api/v1/*`。

| Flask 路由前缀 | 主要用途 |
| --- | --- |
| `/user` | 验证码、登录、注册、账号信息与密码 |
| `/collector` / `/corpus` / `/file` | 采集、文献管理、附件操作 |
| `/admin` / `/leader` | 管理、审核与负责人相关接口 |
| `/public` | 公开文献搜索、附件与统计 |
| `/topic` | 主题相关接口 |
| `/knowledge-bases` | 知识库、文献关联与索引重建 |
| `/ai_search` | AI 搜索、文献问答与索引状态 |

受保护接口使用 `Authorization: Bearer <token>`。角色定义为管理员 `0`、负责人 `1`、数据收集者 `2`、公众 `3`；具体访问规则以接口为准。研究入口面向数据收集者，Agent 会再次验证 JWT、有效账号及资源权限。

## 索引与维护

审核通过后登记任务，Worker 从原文提取文字，按需 OCR，再分块并生成向量与 BM25 文档。只有两路索引写入成功且附件、权限与构建版本复核通过，任务才发布为 `SUCCESS`。旧的 `is_vectorized=True` 不能单独代表研究检索就绪。

当前分块基线为目标 512 token、硬上限 768、重叠约 64；随项目发布的 tokenizer 仅估算预算，实际模型用量以百炼响应为准。Worker 默认八进程，每篇任务失败最多尝试三次，遗留运行任务按超时及文献锁规则回收。

历史回填在 `Backend/` 目录执行，默认只预览：

```bash
python -m scripts.backfill_hybrid_index --limit 20
# 将指定文献登记为重建任务，随后由 Worker 执行
python -m scripts.backfill_hybrid_index --apply --corpus-id 42
```

重建会重新解析和调用模型，并使目标文献在重建期间暂不可检索。部署本身不会自动回填所有历史文献；模型、分块或索引结构变化后，需按目标范围安排重建。

详细说明：

- [索引 Worker、OCR、回填与排障](docs/corpus_index_worker.md)
- [分块策略与测试库重建工具](docs/chunking.md)
- [文献问答参数与 SSE 事件](docs/corpus_qa_frontend.md)

## 开发验证

在 `Backend/` 的虚拟环境中运行离线回归：

```bash
python -m unittest discover -s tests -v
```

在线 OpenSearch 测试默认跳过；显式设置 `FOREIGN_TEST_OPENSEARCH_URL` 后会创建并清理临时测试索引。Agent 检查在其独立目录和环境执行：

```bash
python -m pytest tests
ruff check .
```

## 常见问题

| 现象 | 排查方向 |
| --- | --- |
| 数据库连接失败或迁移报错 | 检查容器到 MySQL 的网络、账号权限、库名、3306 端口及既有业务表；迁移任务表需要建表 / 加字段权限 |
| 页面显示排队且长期未处理 | 检查 Worker 日志，以及 API/Worker 的数据库和 `FOREIGN_INDEX_QUEUE_STATUS` 是否一致 |
| 索引失败或研究没有证据 | 查看任务错误、文件服务可达性、附件是否有文字，以及 Qdrant / OpenSearch 的当前模型和构建版本 |
| OCR 或 Embedding 调用失败 | OCR 按 provider 检查独立密钥、模型、地址、额度和网络；Embedding 检查百炼配置。AICP 401/403 不在请求层重试，MinerU 需有效 Token |
| 研究接口返回 502 / 503 | 查看 Agent 日志和 `/api/v1/health/ready`，确认研究表迁移、共享配置及前端代理目标 |
| 研究接口返回 401 / 403 | 检查 JWT 密钥一致性、令牌有效期、账号状态、角色和知识库权限 |
| 开发验证时导入错误 | 确认当前目录和虚拟环境，避免 Flask 与 Agent 的同名 `app` 包混用 |
