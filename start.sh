#!/bin/sh
# 加载生产配置；默认沿用已有生产索引，允许通过部署环境覆盖。
set -eu
cd "$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"

export FOREIGN_SECRET_KEY="${FOREIGN_SECRET_KEY:-THIS IS USED TO SIGN AND VERIFY JWT TOKENS, REPLACE IT WITH YOUR OWN SECRET, IT CAN BE ANY STRING}"
export FOREIGN_USERNAME="${FOREIGN_USERNAME:-root}"
export FOREIGN_PASSWORD="${FOREIGN_PASSWORD:-ccnu#foreignlanguage}"
export FOREIGN_HOST="${FOREIGN_HOST:-10.222.9.151}"
export FOREIGN_DBNAME="${FOREIGN_DBNAME:-foreign_language}"
export FOREIGN_FILE_URL="${FOREIGN_FILE_URL:-http://10.222.9.151:9100}"
export QWEN_API_KEY="${QWEN_API_KEY:-sk-2fb3bc7207c0413fb9dc6dacd2844ed8}"
export QDRANT_HOST="${QDRANT_HOST:-10.222.9.151}"
export QDRANT_PORT="${QDRANT_PORT:-6333}"
export FOREIGN_QDRANT_COLLECTION="${FOREIGN_QDRANT_COLLECTION:-corpus_chunks_prod_v1}"
export FOREIGN_OPENSEARCH_INDEX="${FOREIGN_OPENSEARCH_INDEX:-foreign_chunks_prod_v1}"
export FOREIGN_INDEX_QUEUE_STATUS="${FOREIGN_INDEX_QUEUE_STATUS:-PENDING}"
export FOREIGN_EMBEDDING_MODEL="${FOREIGN_EMBEDDING_MODEL:-qwen3.7-text-embedding-flash}"
export FOREIGN_EMBEDDING_DIMENSION="${FOREIGN_EMBEDDING_DIMENSION:-1024}"

# 正式 Research 默认使用 Max；在复用 dev 配置前设置，仍允许显式环境变量覆盖。
export MODEL_NAME="${MODEL_NAME:-qwen3.8-max}"
# Max 的长文本结构化生成可能超过一分钟；这是单次请求等待时间，不是整轮总时长。
export RESEARCH_MODEL_TIMEOUT_SECONDS="${RESEARCH_MODEL_TIMEOUT_SECONDS:-300}"

# 复用其余模型、OCR 与客户端默认值；生产数据库及索引名已在上方显式设置。
START_DEV_CONFIG_ONLY=1
. ./start-dev.sh
unset START_DEV_CONFIG_ONLY

if [ "${START_PRODUCTION_CONFIG_ONLY:-0}" != "1" ]; then
    if [ "$#" -gt 0 ]; then
        exec "$@"
    fi
    python -m scripts.migrate_application_schema
    exec flask run --host=0.0.0.0 --port=5000
fi
