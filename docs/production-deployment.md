# dev 合入 master 后的正式部署

正式入口为 `sh deploy.sh`，部署 Flask（宿主端口 9502）、Research Agent
（宿主端口默认 18001，容器端口 18000）和八进程索引 Worker（不发布宿主端口）。
dev 的 9503/18000 及容器不受影响，旧独立生产 Worker 已由运维停止。
正式容器使用 `ForeignBackend` / `ForeignResearchAgent` / `ForeignIndexWorker`
与 `foreign-production` 网络。
部署前同时检查宿主监听和 Docker 端口映射，9502/18001 被其他服务占用时停止部署；
研究服务端口禁止配置为 dev 的 18000/9503 或 Backend 的 9502。

## 生产目标

Backend、Worker 和 Agent 使用同一组生产配置：

| 环境变量 | 默认生产值 |
| --- | --- |
| `FOREIGN_DBNAME` | `foreign_language` |
| `FOREIGN_FILE_URL` | `http://10.222.9.151:9100` |
| `QDRANT_HOST` / `QDRANT_PORT` | `10.222.9.151` / `6333` |
| `FOREIGN_QDRANT_URL` | `http://10.222.9.151:6333` |
| `FOREIGN_QDRANT_COLLECTION` | `corpus_chunks_prod_v1` |
| `FOREIGN_OPENSEARCH_URL` | `http://10.222.9.151:19200` |
| `FOREIGN_OPENSEARCH_INDEX` | `foreign_chunks_prod_v1` |
| `FOREIGN_INDEX_QUEUE_STATUS` | `PENDING` |
| `FOREIGN_EMBEDDING_MODEL` / `FOREIGN_EMBEDDING_DIMENSION` | `qwen3.7-text-embedding-flash` / `1024` |
| `MODEL_NAME` | Research Agent 使用 `qwen3.8-max` |
| `RESEARCH_MODEL_TIMEOUT_SECONDS` | 单次模型请求等待 `300` 秒 |

`start.sh` 保留正式环境原有凭据默认值，并允许环境变量覆盖。部署前确认数据库、JWT、
模型和 OCR 凭据属于当前正式环境；Backend 与 Agent 使用相同 JWT 密钥。
部署脚本会把配置传入新容器，单纯 `docker restart` 不会更新容器环境变量。

正式 Research 的模型默认值在 `start.sh` 中设置，先于共享配置加载；dev 仍默认使用
`qwen3.8-flash`。如果当前 shell 已设置旧的 `MODEL_NAME`，部署时显式运行
`MODEL_NAME=qwen3.8-max sh deploy.sh`。Agent 启动迁移会更新新研究任务的默认模型，
历史任务保留原模型配置。全局搜索的 `LLM_MODEL_NAME` 仍独立配置，默认 `qwen3.8-flash`。

Research 请求时限适用于溯知等直接入口，也适用于启明/AI 委派的专业子图；文献阅读
保留至少 180 秒的既有预算。该值限制单次请求等待，不限制整轮多节点分析总时长，
也不增加重试次数；需要调整时在部署前显式设置 `RESEARCH_MODEL_TIMEOUT_SECONDS`。

## 上线顺序与验证

1. 保存当前应用镜像 ID、数据库备份及 Qdrant/OpenSearch 快照，确认生产服务地址可达。
2. 运行 `sh deploy.sh`。脚本先构建两个镜像，在保留旧容器的情况下幂等补齐
   `corpus.owner_id`、`corpus_index_job`、`knowledge_base`、`knowledge_base_document`，
   再运行 Agent 的研究表迁移。迁移失败时停止部署，不删除现有容器。
3. 两套迁移成功后替换正式 Backend、Agent 和 `ForeignIndexWorker`。
   Worker 默认八进程，消费生产 `PENDING` 队列；停止时最多等待 60 秒完成在途任务。
   脚本不启动、停止或删除旧独立 Worker，也不操作 dev 服务。
4. 将正式前端的 `/research-api/` 代理配置为正式 Agent 的宿主端口 18001，并将路径
   重写到 `/api/v1/`；保留流式响应代理设置。修改 `FOREIGN_RESEARCH_PORT` 时同步调整代理。
   此 Backend 合并不包含正式前端和反向代理的发布。
5. 核验登录、术语多条目附件、知识库权限和研究检索；先确认已有生产文献可检索，
   再用明确的小范围文献验证新任务。不要把全量回填作为部署检查。

迁移只补齐所需结构，不重置文献或任务、不修改现有索引、不自动登记回填。
Agent 会校验任务双路成功、`lexical_index`、构建版本、附件指纹和正文哈希；
生产数据缺少这些条件时应先调查，不要通过修改成功标记绕过校验。

## Worker 与回退

默认 `FOREIGN_DEPLOY_INDEX_WORKER=1`，使用 Backend 镜像启动 `ForeignIndexWorker`，
执行 `python -m app.workers.corpus_index_worker --concurrency 8`。
它与正式 Backend 共用数据库、队列、附件服务、Qdrant 集合和 OpenSearch 索引。
该进程使用 Backend 的 Worker 运行方式，不等同于旧独立生产镜像的资源保护策略。

如仅需更新 Backend 与 Agent，可显式设置 `FOREIGN_DEPLOY_INDEX_WORKER=0`；
此模式不启动、停止或替换 Worker。旧独立镜像与 dev Worker 均不由正式脚本管理。

回退时停止新容器，使用保存的旧镜像 ID 和原运行参数恢复应用。
新增表和可空列保留，不在回退脚本中删除；现有生产索引名称保持原值。
