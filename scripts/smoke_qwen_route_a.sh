#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

QWEN_BASE_URL="${QWEN_BASE_URL:-http://127.0.0.1:8080/v1}"
QWEN_MODEL="${QWEN_MODEL:-Qwen3.8-27B}"
COMMON=(
  --robot ur5e
  --provider qwen
  --planner qwen
  --qwen-base-url "$QWEN_BASE_URL"
  --qwen-model "$QWEN_MODEL"
  --structured-output json_schema
  --viewer-mode headless
)

env -u ALL_PROXY -u all_proxy -u HTTP_PROXY -u http_proxy -u HTTPS_PROXY -u https_proxy \
  NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost \
  robot-agent run "抓取棒球" "${COMMON[@]}" --output-dir var/smoke/route-a-grasp

env -u ALL_PROXY -u all_proxy -u HTTP_PROXY -u http_proxy -u HTTPS_PROXY -u https_proxy \
  NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost \
  robot-agent run "把棒球放到篮子里" "${COMMON[@]}" --output-dir var/smoke/route-a-pick-place

env -u ALL_PROXY -u all_proxy -u HTTP_PROXY -u http_proxy -u HTTPS_PROXY -u https_proxy \
  NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost \
  robot-agent run "把右上角的棒球放到左下角的篮子里" "${COMMON[@]}" --output-dir var/smoke/route-a-spatial
