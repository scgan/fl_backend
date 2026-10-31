#!/bin/sh
# dev 共用配置；保留原默认值，允许部署环境覆盖。
set -eu

export FOREIGN_SECRET_KEY="${FOREIGN_SECRET_KEY:-THIS IS USED TO SIGN AND VERIFY JWT TOKENS, REPLACE IT WITH YOUR OWN SECRET, IT CAN BE ANY STRING}"
export FOREIGN_USERNAME="${FOREIGN_USERNAME:-root}"
export FOREIGN_PASSWORD="${FOREIGN_PASSWORD:-ccnu#foreignlanguage}"
export FOREIGN_HOST="${FOREIGN_HOST:-10.222.9.151}"
export FOREIGN_DBNAME="${FOREIGN_DBNAME:-foreign_language_test}"
export FOREIGN_FILE_URL="${FOREIGN_FILE_URL:-http://10.222.9.151:9101}"
export QWEN_API_KEY="${QWEN_API_KEY:-sk-2fb3bc7207c0413fb9dc6dacd2844ed8}"
export QDRANT_HOST="${QDRANT_HOST:-10.222.9.151}"
export QDRANT_PORT="${QDRANT_PORT:-6333}"
export QWEN_BASE_URL="${QWEN_BASE_URL:-https://dashscope.aliyuncs.com/compatible-mode/v1}"
export LLM_MODEL_NAME="${LLM_MODEL_NAME:-qwen3.8-flash}"
export LLM_TEMPERATURE="${LLM_TEMPERATURE:-0.7}"
# 文献信息提取（/collector/extract_pdf_info）使用的模型；qwen-turbo 并发额度低易触发429，默认用 qwen-plus
export EXTRACT_MODEL_NAME="${EXTRACT_MODEL_NAME:-qwen-plus}"
export QA_MODEL_NAME="${QA_MODEL_NAME:-qwen-long}"
export QA_TEMPERATURE="${QA_TEMPERATURE:-0.3}"
export FOREIGN_OPENSEARCH_URL="${FOREIGN_OPENSEARCH_URL:-http://10.222.9.151:19200}"
export FOREIGN_OPENSEARCH_INDEX="${FOREIGN_OPENSEARCH_INDEX:-foreign_chunks_v1}"
export INDEX_FILE_BASE_DIR="${INDEX_FILE_BASE_DIR:-/foreign-files}"
export MINERU_API_TOKEN="${MINERU_API_TOKEN:-sk-Hro0T8W22ijj2W55L7oa9XCCDERdL5OmMzAUybqIMvCX9trQ}"
export MINERU_BASE_URL="${MINERU_BASE_URL:-https://mineru.net/api/v4}"
export MINERU_MODEL_VERSION="${MINERU_MODEL_VERSION:-vlm}"
export MINERU_LANGUAGE="${MINERU_LANGUAGE:-ch}"
export MINERU_TIMEOUT="${MINERU_TIMEOUT:-1200}"
export MINERU_MAX_PAGES="${MINERU_MAX_PAGES:-200}"
export FOREIGN_OCR_PROVIDER="${FOREIGN_OCR_PROVIDER:-aicp}"
export FOREIGN_OCR_BASE_URL="${FOREIGN_OCR_BASE_URL:-https://aicp.ccnu.edu.cn/v1}"
export FOREIGN_OCR_API_KEY="${FOREIGN_OCR_API_KEY:-sk-yu2LqdcxJ30tPRGlfvyt0JrAaddR98xRD0DaKCSVubuPzCNq}"
if [ "$FOREIGN_OCR_PROVIDER" = "qwen" ]; then
    export FOREIGN_OCR_MODEL="${FOREIGN_OCR_MODEL:-qwen-vl-ocr}"
else
    export FOREIGN_OCR_MODEL="${FOREIGN_OCR_MODEL:-qwen/qwen-vl-ocr-2025-11-20}"
fi
export FOREIGN_OCR_DPI="${FOREIGN_OCR_DPI:-150}"
export FOREIGN_OCR_TIMEOUT="${FOREIGN_OCR_TIMEOUT:-120}"
export FOREIGN_OCR_CONCURRENCY="${FOREIGN_OCR_CONCURRENCY:-4}"
export FOREIGN_QDRANT_URL="${FOREIGN_QDRANT_URL:-http://${QDRANT_HOST}:${QDRANT_PORT}}"
export FOREIGN_QDRANT_COLLECTION="${FOREIGN_QDRANT_COLLECTION:-corpus_chunks}"
export FOREIGN_PUBLIC_API_BASE="${FOREIGN_PUBLIC_API_BASE:-/api}"
export FOREIGN_LITERATURE_ANALYSIS_CONCURRENCY="${FOREIGN_LITERATURE_ANALYSIS_CONCURRENCY:-4}"
export MODEL_NAME="${MODEL_NAME:-qwen3.8-flash}"
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-$QWEN_BASE_URL}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-$QWEN_API_KEY}"

# 部署脚本只加载配置；容器可通过参数启动 Worker 或一次性迁移。
if [ "${START_DEV_CONFIG_ONLY:-0}" != "1" ]; then
    if [ "$#" -gt 0 ]; then
        exec "$@"
    fi
    python -m app.workers.corpus_index_worker --migrate-only
    exec flask run --host=0.0.0.0 --port=5000
fi
