#!/usr/bin/env bash
set -euo pipefail

# Start retrieval services for prepared per-client context files.
# Run prepare_rag_contexts.py first to create build/rag_contexts/client_N_contexts.jsonl.

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

BASE_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON_CMD="${PYTHON_CMD:-python -u}"

CONTEXT_DIR="${CONTEXT_DIR:-${BASE_DIR}/build/rag_contexts}"
BGE_PATH="${BGE_M3_PATH:-${BASE_DIR}/../bge-m3}"
RERANKER_PATH="${BGE_RERANKER_PATH:-${BASE_DIR}/../bge-reranker-v2-m3}"
HOST="${HOST:-0.0.0.0}"
PORT_BASE="${PORT_BASE:-8000}"
DEVICES="${DEVICES:-cuda:0}"
mkdir -p "${BASE_DIR}/logs"

# Clean up any existing processes and files
echo "Cleaning up existing processes and files..."
pgrep -f "${BASE_DIR}/retrieval_api_service.py" | xargs -r kill -9 2>/dev/null || true
# Clean up temp files
rm -f ${BASE_DIR}/logs/client_*.log ${BASE_DIR}/logs/client_*.pid 2>/dev/null || true
sleep 2

for CONTEXT_FILE in "${CONTEXT_DIR}"/client_*_contexts.jsonl; do
  [ -e "${CONTEXT_FILE}" ] || { echo "No context files found in ${CONTEXT_DIR}"; exit 1; }
  CLIENT_BASENAME="$(basename "${CONTEXT_FILE}" _contexts.jsonl)"
  CLIENT_NUM="${CLIENT_BASENAME#client_}"
  PORT=$((PORT_BASE + CLIENT_NUM))
  CLIENT="Client_${CLIENT_NUM}"
  LOG="${BASE_DIR}/logs/client_${CLIENT_NUM}.log"
  PIDFILE="${BASE_DIR}/logs/client_${CLIENT_NUM}.pid"

  echo "Starting ${CLIENT} on port ${PORT}, logs -> ${LOG}"
  nohup ${PYTHON_CMD} ${BASE_DIR}/retrieval_api_service.py \
    --party-id "${CLIENT}" \
    --contexts-path "${CONTEXT_FILE}" \
    --port "${PORT}" \
    --host "${HOST}" \
    --bge-m3-path "${BGE_PATH}" \
    --reranker-path "${RERANKER_PATH}" \
    --devices "${DEVICES}" \
    >> "${LOG}" 2>&1 &
  PID=$!
  echo "${PID}" > "${PIDFILE}"
  echo "Started ${CLIENT} (PID ${PID})"

  echo "  Waiting for ${CLIENT} to initialize models..."
  for attempt in {1..30}; do
    if curl -s "http://localhost:${PORT}/health" > /dev/null 2>&1; then
      echo "  ${CLIENT} service is ready (models loaded)"
      break
    fi
    if [ $attempt -eq 30 ]; then
      echo "  Warning: ${CLIENT} service may not be ready yet"
    fi
    sleep 2
  done

  # brief pause to stagger startups
  sleep 0.5
done

echo ""
echo "=== Final GPU Status (All Models Loaded) ==="
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,memory.free,utilization.gpu,utilization.memory,temperature.gpu --format=csv,noheader,nounits || echo "nvidia-smi not available"
echo ""
echo "All requested clients started with models pre-loaded!"
echo "Use 'tail -f ${BASE_DIR}/logs/client_N.log' to follow logs."
