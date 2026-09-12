# Deployment and runbook

## Requirements

Ubuntu/Linux, Python 3.11 or 3.12, MuJoCo 3.x, NumPy 2.x and a Conda
environment are recommended. GUI execution needs GLFW/OpenGL; CI/headless
execution uses OSMesa/EGL.

## Fresh Conda installation

```bash
git clone git@github.com:haotian2410/robot_agent_stack.git
cd robot_agent_stack
conda create -n robot_agent_integ python=3.11 -y
conda activate robot_agent_integ
python -m pip install -U pip
python -m pip install -e packages/robot_agent_protocol
python -m pip install -e packages/robot_agent_control
python -m pip install -e 'packages/robot_agent_sim[dev]'
```

If an existing verified `robot_agent` environment is available, use
`conda create -n robot_agent_integ --clone robot_agent` instead. For package
deployment, build/install the three wheels; package resources mean the
monorepo root is not required at runtime.

## Qwen and graphics

```bash
export QWEN_BASE_URL=http://127.0.0.1:8080/v1
export QWEN_MODEL=<model-name>
export QWEN_API_KEY=<optional>
```

Only `plan` and `run` may call Qwen. `compile` and `execute` are deterministic.
For headless Linux/CI set `MUJOCO_GL=osmesa` and `PYOPENGL_PLATFORM=osmesa`;
GUI viewer uses GLFW and should run in a clean process.

## Smoke tests

```bash
robot-agent run "把红色方块放进蓝色盒子" --robot ur5e --provider fake --planner recipe \
  --output-dir var/route-a-demo --viewer-mode headless

robot-agent run "打开柜门，把红球放到柜子上层，然后关闭柜门" --robot ur5e \
  --scene packages/robot_agent_control/world_model/robotsim/scene_001.xml \
  --interaction-registry packages/robot_agent_control/demo/common/scenes/scene_001.interactions.json \
  --provider fake --planner recipe --viewer-mode headless

python -m robot_agent_control.cli --commands packages/robot_agent_control/demo/skill_command/config_001.commands.json --viewer-mode headless
```

The output directory contains plan artifacts, `commands.json`,
`execution_bundle.json`, `execution_report.json` and `skill_trace.jsonl`.

## Overrides and troubleshooting

Set `ROBOT_AGENT_CONFIG_ROOT=/path/to/configs` with
`robots/ur5e.json` and/or `execution_profiles/default.json` to override package
resources. Common structured errors are:

- `ROBOT_MODEL_INCOMPATIBLE`: wrong joints, actuators, EE site or home keyframe.
- `ANCHOR_NOT_FOUND` / `EXECUTION_METADATA_MISSING`: author the required interaction metadata.
- `SCENE_FINGERPRINT_MISMATCH`: regenerate commands for the exact scene.
- Qwen connection refused: verify the local `/v1` endpoint and model name.
- `MUJOCO_GL`/GLFW initialization errors: use OSMesa for headless or a clean GUI process.
- wheel resource missing: rebuild all three wheels and verify package data before running.

Run all tests with `./scripts/test_all.sh`; CI additionally performs wheel
installation from a clean venv, runs from `/tmp`, and executes the legacy
`config_001` fixture.
