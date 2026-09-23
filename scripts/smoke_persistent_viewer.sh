#!/usr/bin/env bash
set -euo pipefail

# Manual GUI smoke test; intentionally not part of CI.  Run from a local
# desktop session and close the chat with /exit after checking each turn.
exec robot-agent chat \
  --robot ur5e \
  --provider "${ROBOT_AGENT_PROVIDER:-qwen}" \
  --planner "${ROBOT_AGENT_PLANNER:-qwen}" \
  --qwen-base-url "${QWEN_BASE_URL:-http://127.0.0.1:8080/v1}" \
  --qwen-model "${QWEN_MODEL:-Qwen3.8-27B}" \
  --structured-output json_schema \
  --output-dir "${ROBOT_AGENT_OUTPUT_DIR:-var/viewer-smoke}" \
  --viewer-mode auto
