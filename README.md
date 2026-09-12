# robot_agent_stack

`robot_agent_stack` 是面向 MuJoCo 机器人任务的统一 monorepo，覆盖自然语言任务理解、
场景 grounding、Skill planning、确定性命令编译和 UR5e 控制执行。

```text
User Instruction
      │
      ▼
robot_agent_sim ── TaskIntent → GroundedTask → SkillPlan → Compiler
      │                                      │
      └────────────── ExecutionBundle ──────┘
                         │
                 robot_agent_protocol
                         │
                         ▼
robot_agent_control ── IK / collision / trajectory / gripper
                         │
                         ▼
                 persistent MuJoCo MjModel + MjData
```

统一的 Robot Agent monorepo，保留三个独立 Python package：

```text
packages/robot_agent_protocol  # planner/control 唯一共享契约
packages/robot_agent_sim       # TaskIntent、Route A/B、场景/模型检索、Compiler
packages/robot_agent_control   # ControlExecutor、IK、夹爪、MuJoCo runtime
```

不使用 Git submodule；一个 clone、一个 Conda 环境即可安装完整执行链。`sim` 和 `control`
仍然保持独立 import 名称，跨包的 Command/Execution/Interaction schema 由
`robot_agent_protocol` 唯一维护。

## 能力边界

| Capability | Panda | UR5e |
|---|---:|---:|
| Route A/B planning、SkillPlan | ✅ | ✅ |
| Compile execution commands | ❌ | ✅ |
| MuJoCo control execution | ❌ | ✅ |

| Execution feature | Status |
|---|---|
| generic grasp（MVP，标记 `generic_default`） | ✅ |
| authored cabinet/door metadata | ✅ |
| arbitrary grasp pose / 自动推断任意机构 | ❌ |
| headless / interactive viewer | ✅ |
| real robot hardware | ❌ |

Route A 使用系统生成场景和确定性对象身份；Route B 使用已有 XML，优先读取 authored
interaction sidecar，否则走 RGB/segmentation/semantic grounding。任意 XML 不会自动
获得柜门、抽屉等机构的操作元数据。

## Package 职责与边界

- `robot_agent_protocol`：跨包 Pydantic contract、版本、fingerprint、错误码；不依赖 MuJoCo、Qwen 或 control。
- `robot_agent_sim`：任务理解、Route A/B、grounding、SkillPlan、命令 compiler；compiler 不调用 LLM。
- `robot_agent_control`：IK、碰撞、轨迹、夹爪、Skill 执行、viewer 和持久 MuJoCo runtime。

LLM 只决定语义；InteractionRegistry 决定物体 affordance/anchor；Control 决定 IK、碰撞和轨迹。
`locate` 是 no-op 查询，`search` 必须在 execution 前完成；`open/close` 是 task-level operation，
由 move/grasp/pull(or push)/release 组合而成。

## Fresh install

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

已有经过验证的 `robot_agent` 环境也可以使用 `conda create -n robot_agent_integ --clone robot_agent`。
内置 profile 随 wheel 安装；自定义配置通过 `ROBOT_AGENT_CONFIG_ROOT=/path/to/configs` 覆盖。

## Quick start

```bash
robot-agent run "把红色方块放进蓝色盒子" --robot ur5e --provider fake --planner recipe \
  --output-dir var/route-a-demo --viewer-mode headless
```

真实 Qwen：

```bash
export QWEN_BASE_URL=http://127.0.0.1:8080/v1
export QWEN_MODEL=<model-name>
export QWEN_API_KEY=<optional>
robot-agent plan "把红色方块放进蓝色盒子" --robot ur5e --provider qwen --planner qwen
```

`plan` = Instruction→SkillPlan，`compile` = SkillPlan→ExecutionBundle，`execute` = Bundle→ControlExecutor，
`run` = 三者串联。`compile`/`execute` 不调用 Qwen。

Route B 示例：

```bash
robot-agent run "打开柜门，把红球放到柜子上层，然后关闭柜门" --robot ur5e \
  --scene packages/robot_agent_control/world_model/robotsim/scene_001.xml \
  --interaction-registry packages/robot_agent_control/demo/common/scenes/scene_001.interactions.json \
  --provider fake --planner recipe --viewer-mode headless
```

Viewer 模式为 `auto`、`step`、`headless`；headless 使用 EGL/OSMesa，GUI 使用 GLFW。输出目录会保存
`task_intent.json`、`grounded_task.json`、`skill_plan.json`、`interaction_registry.json`、
`commands.json`、`execution_bundle.json`、`execution_report.json` 和 `skill_trace.jsonl`。

## Testing and limitations

```bash
./scripts/test_all.sh
pytest -q packages/robot_agent_protocol/tests
pytest -q packages/robot_agent_control/tests
pytest -q packages/robot_agent_sim/tests
pytest -q tests/integration tests/e2e
```

CI 还验证 wheel install、`cwd=/tmp`、`config_001` headless execution。当前 control 仅支持 UR5e；Panda
为 planning-only；generic grasp 是简单 primitive 的 MVP；search 尚不是完整 active perception loop，
也不支持真实机器人 hardware。

## 安装

```bash
conda create -n robot_agent_integ --clone robot_agent
conda activate robot_agent_integ
python -m pip install -e packages/robot_agent_protocol
python -m pip install -e packages/robot_agent_control
python -m pip install -e 'packages/robot_agent_sim[dev]'
```

## 统一命令

安装 sim 后，统一入口为 `robot-agent`，兼容入口 `robot-agent-sim` 仍保留：

```bash
robot-agent plan "把红色方块放进蓝色盒子" --robot ur5e
robot-agent compile var
robot-agent execute var --viewer-mode headless
robot-agent run "把红色方块放进蓝色盒子" --robot ur5e --viewer-mode headless
```

真实 Qwen 只参与 `plan`/`run`：先启动本机 OpenAI-compatible Qwen 服务，再设置
`QWEN_BASE_URL` 和 `QWEN_MODEL`。`compile`/`execute` 不依赖 Qwen。

## 验收

```bash
./scripts/test_all.sh
```

架构、Route A/B、MuJoCo/GL 边界和迁移说明见 [`docs/architecture.md`](docs/architecture.md)
与 [`docs/deployment.md`](docs/deployment.md)。
