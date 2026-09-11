#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python -m pip install -e "$ROOT/packages/robot_agent_protocol"
python -m pip install -e "$ROOT/packages/robot_agent_control"
python -m pip install -e "$ROOT/packages/robot_agent_sim[dev]"
