#!/bin/sh
# 部署正式 Backend、Research Agent 与索引 Worker，三者共用生产配置和索引。
set -eu
cd "$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"

START_PRODUCTION_CONFIG_ONLY=1
. ./start.sh
unset START_PRODUCTION_CONFIG_ONLY

production_network="${FOREIGN_PRODUCTION_NETWORK:-foreign-production}"
research_port="${FOREIGN_RESEARCH_PORT:-18001}"
deploy_worker="${FOREIGN_DEPLOY_INDEX_WORKER:-1}"
case "$research_port" in
    ''|*[!0-9]*|9502|9503|18000)
        echo 'FOREIGN_RESEARCH_PORT must be a free port distinct from production Backend and dev ports' >&2
        exit 1 ;;
esac
if [ "$research_port" -lt 1 ] || [ "$research_port" -gt 65535 ]; then
    echo 'FOREIGN_RESEARCH_PORT must be between 1 and 65535' >&2
    exit 1
fi
case "$deploy_worker" in
    0|1) ;;
    *) echo 'FOREIGN_DEPLOY_INDEX_WORKER must be 0 or 1' >&2; exit 1 ;;
esac
if [ "$FOREIGN_OCR_PROVIDER" = "aicp" ] && [ -z "$FOREIGN_OCR_API_KEY" ]; then
    echo 'Missing FOREIGN_OCR_API_KEY; existing services have not been replaced' >&2
    exit 1
fi

# 1. 先构建并完成两套迁移；任一步失败均保留当前容器，不消费或回填索引任务。
docker build -t foreign-backend -f Dockerfile .
docker build -t foreign-research-agent -f research-agent/Dockerfile.foreign ./research-agent

# 同机部署时同时检查 Docker 端口映射与宿主监听；仅允许被待替换的同名正式容器占用。
check_production_port() {
    target_port="$1"
    target_container="$2"
    own_binding=0
    for container in $(docker ps --format '{{.Names}}'); do
        if docker port "$container" | awk -v wanted="$target_port" \
            '{ n=split($NF, parts, ":"); if (parts[n] == wanted) found=1 } END { exit !found }'; then
            if [ "$container" != "$target_container" ]; then
                echo "Port $target_port is already published by $container; existing services have not been replaced" >&2
                exit 1
            fi
            own_binding=1
        fi
    done
    if [ "$own_binding" = '0' ]; then
        docker run --rm --network host --entrypoint python foreign-backend -c \
            'import socket, sys; sock = socket.socket(); sock.bind(("0.0.0.0", int(sys.argv[1]))); sock.close()' \
            "$target_port" || {
                echo "Host port $target_port is in use; existing services have not been replaced" >&2
                exit 1
            }
    fi
}
check_production_port 9502 ForeignBackend
check_production_port "$research_port" ForeignResearchAgent
docker network inspect "$production_network" >/dev/null 2>&1 || docker network create "$production_network"
set -- --network "$production_network" \
  -e FOREIGN_SECRET_KEY -e FOREIGN_USERNAME -e FOREIGN_PASSWORD -e FOREIGN_HOST \
  -e FOREIGN_DBNAME -e FOREIGN_FILE_URL -e QWEN_API_KEY -e QDRANT_HOST -e QDRANT_PORT \
  -e QWEN_BASE_URL -e LLM_MODEL_NAME -e LLM_TEMPERATURE -e EXTRACT_MODEL_NAME \
  -e QA_MODEL_NAME -e QA_TEMPERATURE -e FOREIGN_OPENSEARCH_URL -e FOREIGN_OPENSEARCH_INDEX \
  -e FOREIGN_INDEX_QUEUE_STATUS -e INDEX_FILE_BASE_DIR \
  -e MINERU_API_TOKEN -e MINERU_BASE_URL -e MINERU_MODEL_VERSION -e MINERU_LANGUAGE \
  -e MINERU_TIMEOUT -e MINERU_MAX_PAGES \
  -e FOREIGN_OCR_PROVIDER -e FOREIGN_OCR_BASE_URL -e FOREIGN_OCR_API_KEY -e FOREIGN_OCR_MODEL \
  -e FOREIGN_OCR_DPI -e FOREIGN_OCR_TIMEOUT -e FOREIGN_OCR_CONCURRENCY \
  -e FOREIGN_QDRANT_URL -e FOREIGN_QDRANT_COLLECTION -e FOREIGN_PUBLIC_API_BASE \
  -e FOREIGN_EMBEDDING_MODEL -e FOREIGN_EMBEDDING_DIMENSION \
  -e FOREIGN_LITERATURE_ANALYSIS_CONCURRENCY -e MODEL_NAME -e OPENAI_BASE_URL -e OPENAI_API_KEY \
  -e RESEARCH_MODEL_TIMEOUT_SECONDS
docker run --rm "$@" foreign-backend sh start.sh python -m scripts.migrate_application_schema
docker run --rm "$@" foreign-research-agent python -m app.foreign_integration.migrate

# 2. 只更新本脚本管理的正式服务，不操作旧独立 Worker 或 dev 容器。
if [ "$deploy_worker" = "1" ]; then
    managed_containers='ForeignIndexWorker ForeignResearchAgent ForeignBackend'
else
    managed_containers='ForeignResearchAgent ForeignBackend'
fi
for name in $managed_containers; do
    if docker container inspect "$name" >/dev/null 2>&1; then
        stop_seconds=15
        if [ "$name" = 'ForeignIndexWorker' ]; then stop_seconds=60; fi
        echo "Stopping $name (timeout: ${stop_seconds}s)..."
        docker stop -t "$stop_seconds" "$name"
        docker rm "$name"
    fi
done
docker run -d --restart=always --stop-timeout 15 "$@" -p 9502:5000 \
  --name ForeignBackend foreign-backend
docker run -d --init --restart=always --stop-timeout 15 "$@" -p "${research_port}:18000" \
  --name ForeignResearchAgent foreign-research-agent
if [ "$deploy_worker" = '1' ]; then
    docker run -d --init --restart=always --stop-timeout 60 "$@" \
      --name ForeignIndexWorker foreign-backend sh start.sh \
      python -m app.workers.corpus_index_worker --concurrency 8
fi
docker ps --filter "network=$production_network"
