# Deployment

本文档说明如何在 Linux 上部署、配置和验证 `robot_agent_stack`。

当前推荐部署模式：

```text
one Git clone
+
one Python/Conda environment
+
three Python packages
+
optional Qwen service
```

不需要再分别 clone/deploy 旧的 `robot_agent_sim` 与 `robot_agent_control` 仓库。

---

## 1. Supported Environment

当前 Python package 声明：

```text
Python >=3.11,<3.13
MuJoCo >=3.3,<4
NumPy >=2,<3
Pydantic 2.x
```

CI 当前覆盖：

```text
Python 3.11
Python 3.12
Ubuntu
Mesa/OSMesa headless
```

推荐：

```text
Ubuntu/Linux
Conda
Python 3.11
```

---

## 2. Clone

```bash
git clone git@github.com:haotian2410/robot_agent_stack.git
cd robot_agent_stack
```

HTTPS 也可以：

```bash
git clone https://github.com/haotian2410/robot_agent_stack.git
cd robot_agent_stack
```

---

## 3. Create Environment

### 3.1 Fresh environment

推荐新机器使用：

```bash
conda create -n robot_agent_integ python=3.11 -y
conda activate robot_agent_integ

python -m pip install --upgrade pip
```

### 3.2 Clone an existing validated environment

如果本机已经有经过验证的：

```text
robot_agent
```

环境：

```bash
conda create -n robot_agent_integ --clone robot_agent
conda activate robot_agent_integ
```

这只是本地迁移便利方式，不是项目对新机器的必要前提。

---

## 4. Install Packages

安装顺序：

```text
protocol
→ control
→ sim
```

开发模式：

```bash
python -m pip install -e packages/robot_agent_protocol
python -m pip install -e packages/robot_agent_control
python -m pip install -e 'packages/robot_agent_sim[dev]'
```

验证：

```bash
python -c "import robot_agent_protocol"
python -c "import robot_agent_control"
python -c "import robot_agent_sim"

robot-agent --help
robot-agent-control --help
```

---

## 5. Headless Linux Dependencies

GitHub CI 当前安装：

```bash
sudo apt-get update
sudo apt-get install -y \
  libosmesa6-dev \
  libgl1-mesa-dri \
  libglu1-mesa
```

对于无显示器/SSH/CI 环境，建议先安装这些 Mesa/OSMesa 组件。

headless 测试可手工设置：

```bash
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
```

但通过 `robot-agent execute --viewer-mode headless` 运行时，execution 子进程会自行配置适合的 headless GL 环境。

---

## 6. GUI Viewer

Viewer 模式：

```text
auto
step
```

需要：

- 图形桌面或可用 DISPLAY；
- GLFW 能正常创建窗口；
- 不要把 GUI execution 进程固定到 EGL backend。

`robot-agent execute` 会使用独立 execution 子进程，并在 GUI 模式下移除：

```text
MUJOCO_GL
PYOPENGL_PLATFORM
```

避免 planning EGL 初始化影响 GLFW viewer。

如果从 SSH 使用 GUI，需要自行配置 X11 forwarding/远程桌面等显示环境；否则使用：

```bash
--viewer-mode headless
```

---

## 7. Qwen Service

Qwen 不是 compile/execute 的依赖。

只有使用：

```text
--provider qwen
```

时才需要模型服务。

当前 provider 期望 OpenAI-compatible HTTP API。

例如：

```bash
export QWEN_BASE_URL=http://127.0.0.1:8080/v1
export QWEN_MODEL=<your-model-name>
export QWEN_API_KEY=<optional-key>
```

也可以：

```bash
robot-agent plan \
  "..." \
  --provider qwen \
  --qwen-base-url http://127.0.0.1:8080/v1 \
  --qwen-model <model-name>
```

`QWEN_API_KEY` 当前通过环境变量读取。

---

## 8. Offline Mode

完全不启动 Qwen 时，可以使用：

```text
provider=fake
planner=recipe
```

例如：

```bash
robot-agent run \
  "把红色方块放进蓝色盒子" \
  --robot ur5e \
  --provider fake \
  --planner recipe \
  --output-dir /tmp/robot-agent-route-a \
  --viewer-mode headless
```

这适合：

- CI
- regression
- compiler/debug
- control integration

---

## 9. CLI Deployment Model

统一入口：

```bash
robot-agent
```

兼容入口：

```bash
robot-agent-sim
```

Control 独立入口：

```bash
robot-agent-control
robot-agent-control-demo
```

建议普通使用者只操作：

```text
robot-agent plan
robot-agent compile
robot-agent execute
robot-agent run
```

---

## 10. Command Lifecycle

### Planning

```bash
robot-agent plan "..." --robot ur5e --output-dir var/task01
```

生成语义任务、场景、grounding、SkillPlan。

### Compile

```bash
robot-agent compile var/task01
```

不需要 Qwen。

生成：

```text
interaction_registry.json
commands.json
execution_bundle.json
```

### Execute

```bash
robot-agent execute var/task01 --viewer-mode headless
```

不需要 Qwen，也不重新规划。

### End-to-end

```bash
robot-agent run \
  "..." \
  --robot ur5e \
  --output-dir var/task01 \
  --viewer-mode headless
```

---

## 11. Route A Smoke Test

```bash
rm -rf /tmp/robot-agent-route-a

robot-agent run \
  "把红色方块放进蓝色盒子" \
  --robot ur5e \
  --provider fake \
  --planner recipe \
  --output-dir /tmp/robot-agent-route-a \
  --viewer-mode headless
```

检查：

```bash
ls -lah /tmp/robot-agent-route-a
cat /tmp/robot-agent-route-a/execution_report.json
```

期望：

```text
execution_report.success = true
```

---

## 12. Route B Smoke Test

Route B execution 推荐准备：

```text
scene.xml
scene.interactions.json
```

例如：

```bash
robot-agent run \
  "打开柜门，把红球放到上层，然后关闭柜门" \
  --robot ur5e \
  --scene /path/to/scene_001.xml \
  --interaction-registry /path/to/scene_001.interactions.json \
  --provider fake \
  --planner recipe \
  --output-dir /tmp/robot-agent-route-b \
  --viewer-mode headless
```

如果 sidecar 与 scene 同目录并命名为：

```text
scene_001.interactions.json
```

可以省略 `--interaction-registry`。

注意：

> Route B 没有 authored execution metadata 时可以完成视觉 grounding/规划，但复杂 mechanism 不应假设可以直接 compile/execute。

---

## 13. Legacy Control Smoke Test

仓库保留原控制 fixture：

```bash
python -m robot_agent_control.cli \
  --commands packages/robot_agent_control/demo/skill_command/config_001.commands.json \
  --viewer-mode headless
```

该测试也在 `scripts/test_all.sh` 和 GitHub CI 中运行。

---

## 14. Full Local Validation

```bash
./scripts/test_all.sh
```

等价核心内容：

```bash
python -m pytest -q \
  packages/robot_agent_protocol/tests \
  packages/robot_agent_control/tests \
  packages/robot_agent_sim/tests \
  tests/integration \
  tests/e2e

python -m robot_agent_control.cli \
  --commands packages/robot_agent_control/demo/skill_command/config_001.commands.json \
  --viewer-mode headless
```

---

## 15. Wheel Deployment

CI 会验证三个 package 能以 wheel 独立部署。

手工构建：

```bash
python -m pip install --upgrade build

python -m build packages/robot_agent_protocol
python -m build packages/robot_agent_control
python -m build packages/robot_agent_sim
```

创建干净 venv：

```bash
python -m venv /tmp/robot-agent-wheel-test

/tmp/robot-agent-wheel-test/bin/pip install \
  packages/robot_agent_protocol/dist/*.whl \
  packages/robot_agent_control/dist/*.whl \
  packages/robot_agent_sim/dist/*.whl
```

切离源码目录：

```bash
cd /tmp

/tmp/robot-agent-wheel-test/bin/robot-agent --help
/tmp/robot-agent-wheel-test/bin/robot-agent-control --help
```

验证 package resources：

```bash
/tmp/robot-agent-wheel-test/bin/python - <<'PY'
from robot_agent_control.robot_profile import RobotProfile

print(RobotProfile.load_for_robot("ur5e"))
PY
```

wheel 部署不要求保留 monorepo 根目录 `configs/`。

---

## 16. Built-in Configuration

默认 RobotProfile 作为：

```text
robot_agent_control package resource
```

安装。

因此：

```text
site-packages
```

安装后不依赖原 checkout 目录。

---

## 17. Configuration Override

需要覆盖内置配置时：

```bash
export ROBOT_AGENT_CONFIG_ROOT=/path/to/my-configs
```

目录：

```text
/path/to/my-configs/
└── robots/
    └── ur5e.json
```

没有对应 override 文件时继续使用 package 内置资源。

根目录 [configs/README.md](../configs/README.md) 也说明了该机制。

---

## 18. RobotProfile Deployment Notes

当前 control backend 为 UR5e。

RobotProfile 用于验证：

```text
joints
actuators
end-effector site
home keyframe
IK backend
execution backend
```

如果 scene XML 缺少要求的 MuJoCo name，执行会在 preflight 阶段失败，而不是进入动作后才失败。

---

## 19. Compiler Deployment Notes

Compiler 保持 semantic Skill 与 command 一对一，不使用单独的 execution
profile，也不会在 `grasp` / `release` 后隐式追加 lift、retreat 或 home。
运动 command 在 Control 中以当前 MuJoCo 运行时状态为规划起点。

---

## 20. Output and Persistence

如果：

```bash
--output-dir var/task01
```

则这个目录同时是：

```text
planning artifact directory
+
compile artifact directory
+
execution report directory
```

因此推荐每次任务使用独立目录：

```text
var/task-001
var/task-002
...
```

不要让多个不相关 scene/task 共用同一个目录后手工拼接 bundle。

---

## 21. Replaying an Existing Task

已有：

```text
execution_bundle.json
```

后，可以关闭 Qwen 服务，直接：

```bash
robot-agent execute \
  /path/to/task/execution_bundle.json \
  --viewer-mode headless
```

或：

```bash
robot-agent execute /path/to/task --viewer-mode headless
```

这是调试 Control 最推荐的方式。

---

## 22. Headless vs Viewer Troubleshooting

### `MUJOCO_GL` / OpenGL initialization error

先确认模式。

CI/SSH：

```bash
--viewer-mode headless
```

本地 GUI：

```bash
--viewer-mode auto
```

不要在需要 GLFW viewer 的 shell 中长期：

```bash
export MUJOCO_GL=egl
```

如果之前设置过，可以：

```bash
unset MUJOCO_GL
unset PYOPENGL_PLATFORM
```

再启动 GUI execution。

### GLFW initialization error

确认：

```bash
echo "$DISPLAY"
```

本地桌面应有有效 DISPLAY。

SSH 无 GUI 时使用 headless。

---

## 23. Qwen Troubleshooting

### `--provider qwen` 提示缺少 URL/model

检查：

```bash
echo "$QWEN_BASE_URL"
echo "$QWEN_MODEL"
```

或者在 CLI 显式传入：

```bash
--qwen-base-url ...
--qwen-model ...
```

### connection refused

确认本地 Qwen/llama.cpp/OpenAI-compatible server 已启动，并确认：

```text
QWEN_BASE_URL
```

指向 API root，例如：

```text
http://127.0.0.1:8080/v1
```

### compile/execute 是否应该访问 Qwen？

不应该。

如果你只想调执行：

```bash
robot-agent compile ...
robot-agent execute ...
```

无需模型服务。

---

## 24. Planning Troubleshooting

### `unsupported_recipe`

当前 `planner=recipe` 不支持本次 task operation。

可以：

- 修改任务为支持的 recipe；
- 或在启用 Qwen 时使用 `--planner qwen/auto`。

### `grounding_failed`

Route B VLM detection 与 MuJoCo instance bbox 没有达到匹配条件。

检查：

```text
rgb.png
segmentation.png
visual_grounding.json
instances.json
```

### `grounding_ambiguous`

存在多个候选 object 无法稳定区分。

检查 `visual_grounding.json` 中：

```text
detections
matches
ambiguous
IoU
```

---

## 25. Compile Troubleshooting

### `ANCHOR_NOT_FOUND`

任务明确要求一个 semantic region，但 interaction registry 没有对应 anchor。

例如：

```text
container_interior → interior
grasp_region       → grasp
button_surface     → button_surface
```

不要通过放宽 fallback 来绕过，应补正确 metadata。

### `PERCEPTION_REQUIRED`

`search` 还没有在 perception 阶段被解决，不能进入当前 control runtime。

### execution metadata missing

Route B 可以完成对象识别，但没有用于 execution 的 authored metadata。

提供：

```bash
--interaction-registry ...
```

然后重新 plan/compile。

---

## 26. Execution Troubleshooting

### `SCENE_FINGERPRINT_MISMATCH`

compile 后 scene.xml 被修改。

解决：

```bash
robot-agent compile <task-dir>
```

重新生成 commands/bundle。

### `EXECUTION_BUNDLE_INCONSISTENT`

`execution_bundle.json` 引用的 scene/registry 与 `commands.json` 不一致。

不要从不同 task directory 手工组合文件。

### `ROBOT_MODEL_INCOMPATIBLE`

当前 XML 与 UR5e RobotProfile 不匹配，可能缺：

```text
joint
actuator
EE site
home keyframe
```

先修 scene/robot model，不要绕过 preflight。

### runtime timeout / IK / collision

查看：

```text
execution_report.json
skill_trace.jsonl
```

通过：

```text
source_skill_step_id
command_id
runtime_step_id
```

定位失败层。

---

## 27. Viewer Step Mode

调试动作时：

```bash
robot-agent execute var/task01 --viewer-mode step
```

运行时每个 step 后等待：

```text
Space / Enter
```

适合检查：

- grasp approach
- release
- press
- pull/push
- trajectory/pose 是否合理

关闭 Viewer 会被视为用户取消执行。

---

## 28. CI

GitHub Actions 当前有两个主要验证 job：

### test

Python：

```text
3.11
3.12
```

流程：

```text
clean install
→ all pytest
→ CLI from /tmp
→ legacy config_001 headless
```

### wheel-install

Python 3.11：

```text
build wheels
→ clean venv
→ install wheels
→ cd /tmp
→ package resource check
→ CLI check
```

因此本地改动如果涉及：

```text
pyproject
resources
configs
CLI
control fixture
```

应确保两个 job 都能通过。

---

## 29. Recommended Development Workflow

开发 Planner/Compiler：

```text
plan once
→ repeatedly compile
```

开发 Control：

```text
reuse existing ExecutionBundle
→ repeatedly execute
```

开发 interaction metadata：

```text
keep scene fixed
→ edit registry
→ compile
→ execute
```

这样避免每轮都请求 Qwen，减少变量数量。

---

## 30. Repository Source of Truth

新的 source of truth 是：

```text
robot_agent_stack
```

旧：

```text
robot_agent_sim
robot_agent_control
```

仓库只作为历史/迁移来源保留。

后续功能、bugfix、CI、文档更新都应提交到 `robot_agent_stack`。

---

## 31. Production Boundary

当前项目是 MuJoCo 研究/开发执行栈，不应直接把模拟安全假设当成真实机器人安全保证。

当前未包含：

- hardware E-stop integration
- real robot joint limit/safety controller integration
- human safety zone
- force-torque safety policy
- sim-to-real calibration

在加入真实机器人 backend 前需要单独设计 hardware safety boundary。
