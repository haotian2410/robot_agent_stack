#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
python -m pytest -q packages/robot_agent_protocol/tests packages/robot_agent_control/tests packages/robot_agent_sim/tests tests/integration tests/e2e
python -m robot_agent_control.cli --commands packages/robot_agent_control/demo/skill_command/config_001.commands.json --viewer-mode headless >/dev/null
