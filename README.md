# robot_agent_stack

`robot_agent_stack` 是一个面向 MuJoCo 机器人任务的统一 monorepo，覆盖从自然语言任务理解、场景构建/读取、视觉与语义 grounding、Atomic Skill 规划，到确定性控制命令编译和 UR5e MuJoCo 执行的完整链路。

仓库由三个独立 Python package 组成：

```text
robot_agent_stack/
├── packages/
│   ├── robot_agent_protocol/   # sim/control 唯一共享协议
│   ├── robot_agent_sim/        # TaskIntent、Route A/B、grounding、SkillPlan、Compiler
│   └── robot_agent_control/    # ControlExecutor、IK、碰撞、轨迹、夹爪、MuJoCo runtime
├── configs/                    # 可选的用户配置 override
├── docs/                       # 架构与部署说明
├── tests/                      # 跨 package integration/e2e 测试
├── scripts/
└── .github/workflows/
```

不使用 Git submodule。完整系统只需要 clone 一个仓库，并在一个 Python/Conda 环境中安装三个 package。

> 当前执行后端以 UR5e + Robotiq 2F-85 + MuJoCo 为主。Panda 当前可参与规划，但不支持 control execution。

---

## 1. 系统目标

本项目将机器人任务拆成明确分层：

```text
自然语言
    ↓
TaskIntent
    ↓
Scene / Grounding
    ↓
GroundedTask
    ↓
SkillPlan
    ↓
deterministic Compiler
    ↓
ExecutionBundle
    ↓
ControlExecutor
    ↓
IK / Collision / Trajectory / Gripper
    ↓
Persistent MuJoCo Runtime
    ↓
ExecutionReport
```

核心原则：

- LLM 负责理解任务语义，不直接生成关节角、轨迹或 actuator 指令。
- Planner 负责“做什么”，Control 负责“如何运动”。
- `Compiler` 不调用 LLM；相同 `SkillPlan + scene + interaction metadata + profile` 应得到确定性的 control commands。
- `InteractionRegistry` 保存物体的执行语义，例如 grasp anchor、container interior、button surface、柜门/抽屉机构信息等。
- 执行阶段使用单一持久的 `MjModel/MjData`，确保 grasp、门关节、机器人姿态等状态在连续 Skill 之间不会丢失。
- `robot_agent_protocol` 是跨 planner/control 的唯一正式协议来源。

更详细的设计说明见 [docs/architecture.md](docs/architecture.md)。

---

## 2. Package 职责

### `robot_agent_protocol`

只包含跨 package 的轻量协议和共享定义，例如：

- `CommandDocument`
- `SkillCommand`
- `ExecutionOptions`
- `ExecutionBundle`
- `ExecutionReport`
- `ExecutionFailure`
- `ErrorCode`
- scene fingerprint

它不应依赖：

- MuJoCo
- Qwen
- OpenGL
- IK / trajectory library
- `robot_agent_sim`
- `robot_agent_control`

### `robot_agent_sim`

负责：

- 自然语言任务理解
- Route A 自动场景
- Route B 已有 MuJoCo XML
- asset / scene registry
- RGB / segmentation 渲染
- 视觉和语义 grounding
- `GroundedTask`
- deterministic recipe planner / Qwen planner
- `SkillPlan`
- `SkillPlan → CommandDocument / ExecutionBundle`
- execution orchestration

它不负责底层 IK、碰撞路径、夹爪 actuator 控制。

### `robot_agent_control`

负责：

- `CommandDocument` 校验
- interaction registry 解析
- RobotProfile
- execution preflight
- Skill command conversion
- IK
- collision checking
- trajectory
- gripper
- press / pull / push 等 control runtime
- persistent MuJoCo state
- headless / GUI viewer
- `ExecutionReport` 和 `skill_trace.jsonl`

Control 不理解自然语言，也不依赖 `TaskIntent`、Qwen 或 VLM。

---

## 3. Route A 与 Route B

### Route A：系统自动生成场景

当 `plan/run` 不提供 `--scene` 时走 Route A：

```text
Instruction
  ↓
TaskIntent
  ↓
AssetResolver
  ↓
SceneComposer
  ↓
scene.xml
  ↓
deterministic SceneRegistry
  ↓
InteractionRegistryBuilder
  ↓
GroundedTask
  ↓
SkillPlan
```

Route A 的对象 identity 由系统自己生成，因此不需要再通过 VLM 猜测 object identity。

当前自动生成的 execution metadata 主要支持：

- 普通简单 graspable primitive
- `open_box`
- `button_basic`

普通 graspable 使用 `grasp_source = generic_default` 的 MVP fallback。它是用于验证执行链的通用 top-down 抓取近似，不等价于任意物体 6-DoF grasp pose estimation。

复杂关节机构（任意柜门、抽屉、旋钮等）目前不通过几何自动猜测 hinge/axis/travel，需要 authored interaction metadata。

### Route B：使用已有 MuJoCo XML

提供 `--scene path/to/scene.xml` 时走 Route B。

如果提供 interaction registry：

```text
scene.xml
  +
scene.interactions.json
  ↓
semantic/object binding
  ↓
GroundedTask
  ↓
SkillPlan
  ↓
compile / execute
```

如果没有提供 interaction registry，系统会进行：

```text
scene.xml
  ↓
MuJoCo render
  ↓
RGB
  ↓
VLM detection
  ↓
segmentation truth
  ↓
IoU matching
  ↓
semantic grounding
```

这可以完成对象 grounding 和规划，但**不代表任意 XML 都自动具有可执行的 grasp/door/drawer metadata**。

对于需要编译执行的 Route B 任务，应提供与当前 scene 精确匹配的 interaction registry。CLI 会优先自动查找与 scene 同目录的：

```text
<scene-stem>.interactions.json
```

也可以使用 `--interaction-registry` 显式指定。

---

## 4. Atomic Skill Catalog

当前 planner Skill Catalog：

| Skill | Planner | Control | 含义 |
|---|---:|---:|---|
| `locate` | ✅ | no-op | 定位/引用已知目标 |
| `move` | ✅ | ✅ | 移动末端到目标/语义区域 |
| `grasp` | ✅ | ✅ | 抓取目标 |
| `release` | ✅ | ✅ | 释放目标 |
| `press` | ✅ | ✅ | 按压目标 |
| `pull` | ✅ | ✅ | 拉动已定义的机构 |
| `push` | ✅ | ✅ | 推动已定义的机构 |

| Skill    | Planner | Execution owner       | Status                    |
| -------- | ------: | --------------------- | ------------------------- |
| `search` |       ✅ | Sim / perception loop | ⚠️ active search loop 待实现 |

`open` 和 `close` 是 task-level operation，不是新的底层 atomic skill。Recipe planner 会展开为：

```text
open:
locate(handle)
→ move(handle, grasp_region)
→ grasp(handle)
→ pull(door)
→ release(handle)

close:
locate(handle)
→ move(handle, grasp_region)
→ grasp(handle)
→ push(door)
→ release(handle)
```

Pick-and-place 的语义 recipe 为：

```text
locate(source)
→ move(source, grasp_region)
→ grasp(source)
→ locate(destination)
→ move(destination, container_interior)
→ release(source)
```

Compiler 还会根据 ExecutionProfile 添加确定性的 execution micro-steps，例如 post-grasp lift、home、post-release retreat。数值不由 LLM 输出。

---

## 5. 当前能力边界

### Robot 支持

| Capability | Panda | UR5e |
|---|---:|---:|
| Route A planning | ✅ | ✅ |
| Route B planning | ✅ | ✅ |
| Grounding / SkillPlan | ✅ | ✅ |
| Compile execution commands | ❌ | ✅ |
| MuJoCo control execution | ❌ | ✅ |

### Execution feature

| Feature | Status |
|---|---|
| simple primitive `generic_default` grasp | ✅ MVP |
| generated open box placement target | ✅ |
| generated basic button press metadata | ✅ |
| authored cabinet/door interaction metadata | ✅ |
| arbitrary object grasp pose estimation | ❌ |
| automatically inferred arbitrary articulated mechanism | ❌ |
| headless execution | ✅ |
| interactive MuJoCo viewer | ✅ |
| real robot hardware backend | ❌ |
| full active visual-search loop | ❌ |

---

## 6. 环境要求

当前 package 声明：

- Python `>=3.11,<3.13`
- MuJoCo `>=3.3,<4`
- NumPy `>=2,<3`
- Pydantic 2.x

推荐 Ubuntu/Linux + Conda。

---

## 7. 安装

### 7.1 全新环境

```bash
git clone git@github.com:haotian2410/robot_agent_stack.git
cd robot_agent_stack

conda create -n robot_agent_integ python=3.11 -y
conda activate robot_agent_integ

python -m pip install --upgrade pip
python -m pip install -e packages/robot_agent_protocol
python -m pip install -e packages/robot_agent_control
python -m pip install -e 'packages/robot_agent_sim[dev]'
```

安装完成后验证：

```bash
robot-agent --help
robot-agent-control --help
```

### 7.2 复用已有本地环境

如果本机已有经过验证的 `robot_agent` Conda 环境，可以 clone：

```bash
conda create -n robot_agent_integ --clone robot_agent
conda activate robot_agent_integ

python -m pip install -e packages/robot_agent_protocol
python -m pip install -e packages/robot_agent_control
python -m pip install -e 'packages/robot_agent_sim[dev]'
```

详细部署与 headless/GUI 环境说明见 [docs/deployment.md](docs/deployment.md)。

---

## 8. Qwen 配置

`provider=qwen` 使用 OpenAI-compatible HTTP API。

```bash
export QWEN_BASE_URL=http://127.0.0.1:8080/v1
export QWEN_MODEL=<model-name>
export QWEN_API_KEY=<optional-key>
```

也可以通过 CLI 指定：

```bash
--qwen-base-url ...
--qwen-model ...
```

当前边界：

```text
plan     → 可能调用 Qwen
run      → planning 阶段可能调用 Qwen
compile  → 不调用 Qwen
execute  → 不调用 Qwen
```

因此只要已经生成 `ExecutionBundle`，后续 compile/execute 调试可以与模型服务解耦。

### Provider

```text
fake  → 离线测试 provider
qwen  → OpenAI-compatible Qwen HTTP provider
```

### Planner

```text
recipe → deterministic recipe planner
qwen   → Qwen Skill planner
auto   → recipe 支持时优先 recipe，否则使用 Qwen planner
```

---

## 9. 统一 CLI

安装 `robot_agent_sim` 后提供：

```text
robot-agent
robot-agent-sim   # compatibility alias
```

主要使用 `robot-agent`。

### 9.1 只规划

```bash
robot-agent plan \
  "把红色方块放进蓝色盒子" \
  --robot ur5e \
  --output-dir var/task01
```

`plan` 只生成规划/场景/grounding 产物，不执行机器人动作。

注意：`plan` 当前默认 robot 是 `panda`；如果后续准备执行，请显式指定：

```bash
--robot ur5e
```

### 9.2 只编译

```bash
robot-agent compile var/task01
```

将已有：

```text
GroundedTask + SkillPlan + scene + interaction metadata
```

确定性编译为：

```text
commands.json + execution_bundle.json
```

不调用 Qwen。

### 9.3 只执行

```bash
robot-agent execute var/task01 --viewer-mode headless
```

也可直接传：

```bash
robot-agent execute var/task01/execution_bundle.json --viewer-mode headless
```

不调用 Qwen，不重新规划。

### 9.4 一次完成 plan → compile → execute

```bash
robot-agent run \
  "把红色方块放进蓝色盒子" \
  --robot ur5e \
  --output-dir var/task01 \
  --viewer-mode headless
```

`run` 当前默认 robot 为 `ur5e`。

---

## 10. Route A 示例

离线 deterministic smoke test：

```bash
robot-agent run \
  "把红色方块放进蓝色盒子" \
  --robot ur5e \
  --provider fake \
  --planner recipe \
  --output-dir var/route-a-demo \
  --viewer-mode headless
```

只规划：

```bash
robot-agent plan \
  "把红色方块放进蓝色盒子" \
  --robot ur5e \
  --provider fake \
  --planner recipe \
  --output-dir var/route-a-demo
```

之后可以反复：

```bash
robot-agent compile var/route-a-demo
robot-agent execute var/route-a-demo --viewer-mode headless
```

这样修改 Compiler/Control 时无需重复调用模型。

---

## 11. Route B 示例

使用已有 MuJoCo XML：

```bash
robot-agent plan \
  "打开柜门，把红球放到上层，然后关闭柜门" \
  --robot ur5e \
  --scene /path/to/scene_001.xml \
  --interaction-registry /path/to/scene_001.interactions.json \
  --provider fake \
  --planner recipe \
  --output-dir var/route-b-demo
```

编译：

```bash
robot-agent compile var/route-b-demo
```

执行：

```bash
robot-agent execute var/route-b-demo --viewer-mode headless
```

或者：

```bash
robot-agent run \
  "打开柜门，把红球放到上层，然后关闭柜门" \
  --robot ur5e \
  --scene /path/to/scene_001.xml \
  --interaction-registry /path/to/scene_001.interactions.json \
  --provider fake \
  --planner recipe \
  --output-dir var/route-b-demo \
  --viewer-mode auto
```

如果 `scene_001.interactions.json` 与 XML 同目录且命名为 `<scene-stem>.interactions.json`，CLI 会优先自动发现。

---

## 12. Viewer 模式

`execute/run` 支持：

```text
auto
step
headless
```

### `auto`

打开 MuJoCo viewer 并连续执行 runtime steps。

### `step`

打开 viewer，每个 runtime step 后等待：

```text
Space / Enter
```

### `headless`

不开 viewer，适合：

- CI
- SSH
- 自动回归测试
- execution debugging

Planning/offscreen render 使用 EGL；GUI execution 会在独立进程中清理 EGL 相关环境变量后启动 GLFW viewer，避免 OpenGL backend 在同一进程初始化后发生冲突。

---

## 13. 任务输出目录

一次成功的 plan/compile/execute 任务通常会在 `--output-dir` 下产生：

```text
var/task01/
├── task_intent.json
├── scene_registry.json
├── grounded_task.json
├── visual_grounding.json        # Route B/VLM 场景可能存在
├── skill_plan.json
├── model_usage.json
├── summary.json
├── asset_bindings.json
├── rgb.png
├── segmentation.npy
├── segmentation.png
├── instances.json
├── scene.xml                    # Route A 生成
├── interaction_registry.json
├── commands.json                # compile 生成
├── execution_bundle.json        # compile 生成
├── execution_report.json        # execute 生成
└── skill_trace.jsonl            # execute 生成
```

主要文件：

| 文件 | 含义 |
|---|---|
| `task_intent.json` | 从用户指令得到的任务语义 |
| `scene_registry.json` | 当前 scene 中的对象 identity / body 等信息 |
| `grounded_task.json` | TaskIntent 中实体与 scene object 的绑定结果 |
| `visual_grounding.json` | Route B 视觉检测、IoU、匹配结果 |
| `skill_plan.json` | Planner 输出的 atomic semantic steps |
| `interaction_registry.json` | 物体执行语义：anchor、action request、mechanism metadata |
| `commands.json` | Compiler 生成的正式 `CommandDocument` |
| `execution_bundle.json` | scene/registry/commands 的一致性封装 |
| `execution_report.json` | 控制执行成功/失败、command/step 统计与 failure |
| `skill_trace.jsonl` | runtime step 级执行追踪 |
| `summary.json` | route、planner、status、source scene、model usage 等摘要 |

`commands.json` 和 `execution_bundle.json` 中使用绝对路径，并通过 scene SHA-256 fingerprint 防止 scene 与 registry/command 漂移。

---

## 14. Interaction Registry

`SceneRegistry` 和 `InteractionRegistry` 的职责不同：

```text
SceneRegistry
= “这个对象是谁”

InteractionRegistry
= “这个对象如何被操作”
```

Interaction metadata 可以包含：

- aliases
- MuJoCo body/site/geom/joint source
- reference pose
- anchors
- local directions
- tool orientation
- `grasp` / `release` / `press` action requests
- affordances
- door/drawer acting target
- mechanism joint/axis/travel 等

对于复杂机制，不应让 LLM 猜这些物理参数。

---

## 15. RobotProfile 与 ExecutionProfile

### RobotProfile

UR5e 内置 profile 随 `robot_agent_control` package 安装，用于描述：

- robot name
- required joints
- actuators
- end-effector site
- home keyframe
- IK backend
- execution backend

Control preflight 会先在 raw MuJoCo model 上验证 RobotProfile，再初始化 SkillRuntime。

### ExecutionProfile

默认 execution profile 随 `robot_agent_sim` package 安装，用于确定性 execution macros，例如：

- post-grasp lift distance
- lift relation/frame
- post-release retreat distance
- retreat relation/frame

这些参数不由 Qwen 输出。

### 用户 override

可以设置：

```bash
export ROBOT_AGENT_CONFIG_ROOT=/path/to/configs
```

目录结构：

```text
/path/to/configs/
├── robots/
│   └── ur5e.json
└── execution_profiles/
    └── default.json
```

没有 override 时使用随 wheel/package 安装的内置资源。

---

## 16. Control 独立入口

Control 仍可以脱离 planner 单独测试：

```bash
robot-agent-control \
  --commands /path/to/commands.json \
  --viewer-mode headless
```

兼容入口：

```text
robot-agent-control-demo
```

项目保留 legacy `config_001.commands.json` 作为 regression fixture。

---

## 17. 测试

完整本地验收：

```bash
./scripts/test_all.sh
```

它会运行：

```text
protocol unit tests
control unit tests
sim unit tests
integration tests
e2e tests
legacy config_001 headless execution
```

也可以分别执行：

```bash
pytest -q packages/robot_agent_protocol/tests
pytest -q packages/robot_agent_control/tests
pytest -q packages/robot_agent_sim/tests
pytest -q tests/integration
pytest -q tests/e2e
```

GitHub Actions 当前还会：

- Python 3.11 / 3.12 clean install
- headless Mesa/OSMesa 环境
- 从 `/tmp` 验证 CLI 不依赖 checkout cwd
- 完整 legacy control fixture
- 构建 protocol/control/sim wheels
- 在 clean venv 中安装 wheels
- 从 `/tmp` 验证 package resources 与 CLI

---

## 18. 常见问题

### `control execution currently supports ur5e only`

当前 control backend 只支持 UR5e。

规划 Panda：

```bash
robot-agent plan "..." --robot panda
```

可以。

但 Panda execution 当前不支持。

### `ANCHOR_NOT_FOUND`

SkillPlan 指定了明确 semantic target，例如：

```text
container_interior
grasp_region
button_surface
```

但 interaction registry 中没有对应 anchor。

Compiler 会 fail-closed，不会静默改用其它 anchor。

### `EXECUTION_METADATA_MISSING`

scene/object 已经被识别，但缺少执行该操作所需的 interaction metadata。

典型场景：

- 任意 Route B XML 没有 authored mechanism metadata
- 门/抽屉没有 joint/handle/action request
- object 没有目标 anchor

### `SCENE_FINGERPRINT_MISMATCH`

`commands.json` / `ExecutionBundle` 所引用的 scene 与编译时 scene 不一致。

重新 compile，或者恢复原 scene。

### `EXECUTION_BUNDLE_INCONSISTENT`

`execution_bundle.json` 和它引用的 `commands.json` 中：

```text
scene
scene_fingerprint
interaction_registry
```

不一致。

不要手工拼接来自不同 task directory 的 bundle/commands。

---

## 19. 当前限制

当前项目仍明确不解决：

- Panda control backend
- 任意复杂 mesh 的通用 6-DoF grasp pose generation
- 无 metadata 的任意门/抽屉机构自动建模
- 完整 active search + camera re-observation loop
- 真实机器人 hardware backend
- sim-to-real calibration / safety layer

这些是后续能力扩展，不属于当前 monorepo 整合问题。

---

## 20. 文档

- [Architecture](docs/architecture.md)：模块边界、Route A/B、协议、Compiler、ControlRuntime、GL 边界和错误传播。
- [Deployment](docs/deployment.md)：从新机器安装、Qwen、headless、GUI、wheel、配置 override 到 smoke test。
- [Config overrides](configs/README.md)：RobotProfile / ExecutionProfile override 目录规则。

---

## 21. 开发原则

后续开发建议只在 `robot_agent_stack` 中进行。

旧的：

```text
robot_agent_sim
robot_agent_control
```

仓库仅保留为迁移来源/历史，不再作为新的 source of truth。

下一阶段推荐独立规划：

1. 通用 grasp pose generator
2. active visual search / perception loop
3. Panda execution backend
4. dynamic world-state feedback
5. real robot backend
