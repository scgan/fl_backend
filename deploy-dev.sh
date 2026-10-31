#!/bin/sh
# 在服务器构建并部署 dev 后端、八进程索引 Worker 和 Research Agent。
set -eu
cd "$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"

# 三个容器共用配置，按变量名传给 Docker。
START_DEV_CONFIG_ONLY=1
. ./start-dev.sh
unset START_DEV_CONFIG_ONLY
if [ "$FOREIGN_OCR_PROVIDER" = "aicp" ] && [ -z "$FOREIGN_OCR_API_KEY" ]; then
  echo "请先设置 FOREIGN_OCR_API_KEY，再执行部署。" >&2
  exit 1
fi

docker build -t foreign-backend-dev -f Dockerfile.dev .
docker build -t foreign-research-agent-dev -f research-agent/Dockerfile.foreign ./research-agent
docker network inspect foreign-dev >/dev/null 2>&1 || docker network create foreign-dev

set -- --network foreign-dev \
  -e FOREIGN_SECRET_KEY -e FOREIGN_USERNAME -e FOREIGN_PASSWORD -e FOREIGN_HOST \
  -e FOREIGN_DBNAME -e FOREIGN_FILE_URL -e QWEN_API_KEY -e QDRANT_HOST -e QDRANT_PORT \
  -e QWEN_BASE_URL -e LLM_MODEL_NAME -e LLM_TEMPERATURE -e QA_MODEL_NAME -e QA_TEMPERATURE \
  -e FOREIGN_OPENSEARCH_URL -e FOREIGN_OPENSEARCH_INDEX -e INDEX_FILE_BASE_DIR \
  -e MINERU_API_TOKEN -e MINERU_BASE_URL -e MINERU_MODEL_VERSION -e MINERU_LANGUAGE \
  -e MINERU_TIMEOUT -e MINERU_MAX_PAGES \
  -e FOREIGN_OCR_PROVIDER -e FOREIGN_OCR_BASE_URL -e FOREIGN_OCR_API_KEY -e FOREIGN_OCR_MODEL \
  -e FOREIGN_OCR_DPI -e FOREIGN_OCR_TIMEOUT -e FOREIGN_OCR_CONCURRENCY \
  -e FOREIGN_QDRANT_URL -e FOREIGN_QDRANT_COLLECTION -e FOREIGN_PUBLIC_API_BASE \
  -e FOREIGN_LITERATURE_ANALYSIS_CONCURRENCY \
  -e MODEL_NAME -e OPENAI_BASE_URL -e OPENAI_API_KEY

# 迁移失败时保留原服务，不执行删除。
docker run --rm "$@" foreign-backend-dev python -m app.workers.corpus_index_worker --migrate-only

# Worker 留 60 秒收尾，其余服务最多等待 15 秒。
for name in ForeignIndexWorker-dev ForeignResearchAgent-dev ForeignBackend-dev; do
  if docker container inspect "$name" >/dev/null 2>&1; then
    stop_seconds=15
    if [ "$name" = "ForeignIndexWorker-dev" ]; then
      stop_seconds=60
    fi
    echo "Stopping $name (timeout: ${stop_seconds}s)..."
    docker stop -t "$stop_seconds" "$name"
    docker rm "$name"
  fi
done

docker run -d --restart=always --stop-timeout 15 "$@" -p 9503:5000 \
  -v /home/ccnu/Foreign/FileDev:/foreign-files \
  --name ForeignBackend-dev foreign-backend-dev
docker run -d --init --restart=always --stop-timeout 60 "$@" \
  -v /home/ccnu/Foreign/FileDev:/foreign-files \
  --name ForeignIndexWorker-dev foreign-backend-dev \
  python -m app.workers.corpus_index_worker --concurrency 8
docker run -d --init --restart=always --stop-timeout 15 "$@" -p 18000:18000 \
  --name ForeignResearchAgent-dev foreign-research-agent-dev

docker ps --filter network=foreign-dev
