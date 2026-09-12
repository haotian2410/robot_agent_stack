# Architecture

本文档描述 `robot_agent_stack` 当前已经实现的架构、模块责任边界和数据流。它用于回答三个核心问题：

1. Planner、Compiler 和 Control 各自负责什么？
2. Route A / Route B 如何把自然语言任务落到具体 MuJoCo 对象？
3. 为什么执行阶段能够脱离 Qwen，并保持 scene、registry 和 commands 一致？

---

## 1. Design Goals

系统设计遵循以下目标：

- **语义与物理控制解耦**：LLM 负责理解和高层语义，底层控制保持确定性。
- **可重放**：一旦生成 `ExecutionBundle`，可以不依赖 Qwen 反复执行。
- **可审计**：从 `SkillPlan step → Command → runtime step` 可以追踪失败位置。
- **单一执行状态**：连续 Skill 共用一个 MuJoCo `MjModel/MjData`。
- **跨 package 契约稳定**：sim/control 不重复定义 Command/Execution schema。
- **部署可移植**：内置 profile 随 wheel 安装，不依赖 monorepo 根目录。
- **失败优先于猜测**：缺少明确 anchor/metadata 时 fail-closed，不让 LLM 补物理参数。

---

## 2. Monorepo Boundary

```text
robot_agent_stack/
├── packages/
│   ├── robot_agent_protocol/
│   ├── robot_agent_sim/
│   └── robot_agent_control/
├── tests/
│   ├── integration/
│   └── e2e/
├── docs/
├── configs/
└── scripts/
```

仓库统一，但 package 仍独立。

### Dependency direction

```text
                 ┌──────────────────────┐
                 │ robot_agent_protocol │
                 └──────────▲───────────┘
                            │
                ┌───────────┴───────────┐
                │                       │
      robot_agent_sim          robot_agent_control
                │                       ▲
                └── execution adapter ──┘
```

正式依赖原则：

```text
sim     → protocol
control → protocol
sim execution adapter → control public API
```

禁止：

```text
control → sim
protocol → sim
protocol → control
```

`protocol` 不包含 MuJoCo、Qwen、OpenGL、IK 等重依赖。

---

## 3. End-to-End Data Flow

```text
User Instruction
        │
        ▼
Task Understanding
        │
        ▼
TaskIntent
        │
        ▼
Scene Construction / Scene Loading
        │
        ▼
Grounding
        │
        ▼
GroundedTask
        │
        ▼
Skill Planning
        │
        ▼
SkillPlan
        │
        ▼
Deterministic Compiler
        │
        ▼
CommandDocument
        │
        ├── scene
        ├── scene_fingerprint
        ├── interaction registry
        ├── runtime options
        └── SkillCommands
        │
        ▼
ExecutionBundle
        │
        ▼
ControlExecutor
        │
        ▼
Preflight
        │
        ▼
SkillCommandConverter
        │
        ▼
SkillRuntime
        │
        ▼
Persistent MuJoCo MjModel/MjData
        │
        ▼
ExecutionReport + skill_trace.jsonl
```

---

## 4. LLM Boundary

LLM/Qwen 可以参与：

- Task understanding
- Route B visual semantic detection
- Qwen Skill planner

LLM 不负责：

- world XYZ
- robot joint values
- IK
- trajectory waypoints
- collision-free path
- gripper actuator command
- cabinet hinge axis
- push/pull physical trajectory
- post-grasp lift distance

这些内容来自：

```text
Scene
+
InteractionRegistry
+
RobotProfile
+
ExecutionProfile
+
Control algorithms
```

### 为什么 Compiler 不能调用 LLM

`SkillPlan` 是语义层：

```text
grasp(red_ball)
move(blue_box, container_interior)
```

`CommandDocument` 是正式执行协议。

如果 Compiler 再请求 LLM，就会导致：

- 相同计划无法稳定重放；
- 物理参数漂移；
- 失败难以归因；
- token/latency 增加；
- 测试无法 deterministic。

因此 Compiler 是纯确定性转换。

---

## 5. Route A

Route A 在没有 `--scene` 时使用。

```text
Instruction
   ↓
TaskIntent
   ↓
AssetResolver
   ↓
SceneComposer
   ↓
SceneRegistry
   ↓
MuJoCo scene.xml
   ↓
InteractionRegistryBuilder
   ↓
Render
   ↓
deterministic object binding
   ↓
GroundedTask
   ↓
SkillPlan
```

### Identity

Route A 是系统自己选择 asset 并创建 object：

```text
entity_id
→ selected model
→ object_id
→ body_name
```

因此 identity 是 deterministic 的，不需要 VLM 再通过 bbox 猜 object identity。

### Generated InteractionRegistry

Builder 会重新加载最终 `scene.xml`：

```python
MjModel.from_xml_path(...)
MjData(...)
mj_forward(...)
```

然后从最终模型读取 object world pose。

这样避免：

```text
scene composer 中间坐标
≠
最终 MuJoCo world 坐标
```

导致 execution registry 漂移。

### 当前自动 metadata 边界

自动 builder 主要覆盖：

```text
ordinary graspable primitive
open_box
button_basic
```

普通物体使用：

```text
interaction_metadata.grasp_source = generic_default
```

它是 fallback，不是完整 grasp pose estimation。

任意 door/drawer 等 articulated mechanism 不自动猜：

```text
joint
axis
handle
travel
arc center
```

这类信息需要 authored metadata。

---

## 6. Route B

Route B 在提供已有 XML/MJCF 时使用：

```bash
--scene path/to/scene.xml
```

系统先加载并 render 当前 scene。

### Route B with authored interaction registry

```text
scene.xml
+
scene.interactions.json
        ↓
InteractionRegistry semantic grounding
        ↓
GroundedTask
        ↓
SkillPlan
        ↓
Compiler
        ↓
Execution
```

这是复杂 mechanism 推荐路径。

CLI 如果没有显式提供 registry，会先尝试：

```text
<scene-directory>/<scene-stem>.interactions.json
```

### Route B without authored registry

```text
scene.xml
  ↓
RGB render
  ↓
VLM detection
  ↓
MuJoCo segmentation / instance truth
  ↓
bbox IoU matching
  ↓
WorldRelationResolver
  ↓
GroundedTask
```

当前匹配流程维护：

- detection bbox
- instance bbox
- IoU score
- unmatched
- ambiguous
- selection relations

但这只解决“哪个可见 object 对应用户语义”，不自动解决“这个 object 如何被物理操作”。

因此：

```text
Route B visual grounding success
≠
Route B execution metadata available
```

没有 interaction metadata 时，可以 plan；compile execution 会要求 execution metadata。

---

## 7. SceneRegistry vs InteractionRegistry

这是系统里两个容易混淆但职责完全不同的 registry。

### SceneRegistry

回答：

> 这个对象是谁？

典型内容：

```text
scene_id
robot
object_id
entity_id
semantic_name
model_id
model_name
body_name
dimensions
bindings
```

主要服务：

- asset/scene construction
- identity
- grounding
- render/segmentation

### InteractionRegistry

回答：

> 这个对象怎么被执行层操作？

典型内容：

```text
aliases
body/site/geom/joint source
reference_pose
anchors
directions_local
default_distances
tool_orientation
default_interactions
action_requests
affordances
mechanism metadata
```

例如一个容器可能有：

```text
anchors.interior
```

一个按钮有：

```text
anchors.button_surface
action_requests.press
```

柜门可能需要：

```text
handle
joint
axis
pull/push affordance
```

### 为什么不合并

Scene identity 与 execution affordance 生命周期不同。

一个 object 即使可以被视觉识别，也不代表系统知道：

- 从哪里抓；
- 怎么拉；
- 绕哪个 joint 转；
- 放置 interior 在哪里。

分离两个 registry 可以让 grounding 与 control failure 更清晰。

---

## 8. TaskIntent, GroundedTask, SkillPlan

### TaskIntent

来自自然语言任务理解，描述：

- task type
- semantic entities
- operations
- spatial relations
- dependency

此阶段不绑定 MuJoCo object id。

### GroundedTask

将语义实体绑定到当前 scene object：

```text
semantic entity
→ object_id
→ body_name
```

Grounding method 可能是：

```text
asset_scene_binding
interaction_registry
vlm_iou
```

### SkillPlan

只描述 semantic atomic steps，例如：

```text
step-1 locate red_ball
step-2 move red_ball / grasp_region
step-3 grasp red_ball
step-4 locate blue_box
step-5 move blue_box / container_interior
step-6 release red_ball
```

SkillPlan 不包含 joint angles 或 trajectory。

---

## 9. Atomic Skills and Recipes

当前 Atomic Skill：

```text
locate
search
move
grasp
release
press
pull
push
```

### Recipe examples

Grasp：

```text
locate
move(grasp_region)
grasp
```

Press：

```text
locate
move(button_surface)
press
```

Pick-and-place：

```text
locate(source)
move(source, grasp_region)
grasp(source)
locate(destination)
move(destination, container_interior)
release(source)
```

Open：

```text
locate(handle)
move(handle, grasp_region)
grasp(handle)
pull(mechanism)
release(handle)
```

Close：

```text
locate(handle)
move(handle, grasp_region)
grasp(handle)
push(mechanism)
release(handle)
```

`open/close` 是 operation recipe，不是底层 atomic skill。

---

## 10. Semantic Step vs Execution Micro-Step

一个 SkillPlan step 不一定等于一个 runtime step。

例如：

```text
SkillPlan:
grasp(red_ball)
```

Compiler 可以在 pick-and-place 中追加：

```text
move end_effector above by profile distance
move home
```

Control converter 还可能将一个 move command 展开成多个实际 runtime steps。

因此 trace 关系是：

```text
SkillPlan step
   ↓ source_skill_step_id
Command
   ↓ command_id
Runtime step(s)
   ↓ runtime_step_id
ExecutionStepReport
```

这也是 `ExecutionReport` 保留三种 ID 的原因。

---

## 11. Compiler Responsibilities

Compiler 输入：

```text
SkillPlan
GroundedTask
scene.xml
InteractionRegistry
ExecutionProfile
robot
```

输出：

```text
interaction_registry.json
commands.json
execution_bundle.json
```

Compiler 负责：

- semantic region → anchor 映射；
- `grasp_region → grasp`
- `container_interior → interior`
- `button_surface → button_surface`
- 生成 command id；
- 保留 `source_skill_step_id`；
- 添加 deterministic execution macros；
- 写 scene fingerprint；
- 生成 absolute paths。

Compiler 不负责：

- 调用 Qwen；
- IK；
- collision path；
- actuator control。

### Fail-closed anchor

如果 SkillPlan 明确要求：

```text
container_interior
```

而 registry 没有：

```text
anchors.interior
```

Compiler 直接失败。

它不会 fallback 到 `default_anchor`，避免把“放进盒子”误执行成“移动到盒子其它默认位置”。

### Search

`search` 在 Skill Catalog 中存在，但当前不能直接进入 Control。

未完成 perception resolution 的 `search` 会在 compile 阶段被拒绝。

---

## 12. ExecutionProfile

ExecutionProfile 是 sim-side deterministic execution macro 配置。

当前包括：

```text
post_grasp_lift_m
lift_relation
lift_frame
post_release_retreat_m
retreat_relation
retreat_frame
```

合法 frame 当前限定：

```text
world
tool
```

默认 profile 作为 package resource 随 wheel 安装。

用户可以通过：

```bash
ROBOT_AGENT_CONFIG_ROOT
```

override。

---

## 13. CommandDocument

正式 v1 command contract 由 `robot_agent_protocol` 定义。

核心字段：

```text
schema_version = 1.0
robot
scene
registry
scene_fingerprint
runtime
request_defaults
commands[]
```

要求：

- `scene` 是绝对路径；
- `registry` 是绝对路径；
- fingerprint 是 SHA-256；
- `commands` 至少一个；
- SkillCommand 必须有非空 `parameters.target`；
- unknown fields 默认被 Pydantic strict model 拒绝。

SkillCommand：

```text
command_id
source_skill_step_id
skill_name
parameters
```

当前 control command skill：

```text
locate
move
grasp
release
press
pull
push
```

---

## 14. ExecutionBundle

ExecutionBundle 是持久化的执行入口：

```text
schema_version
robot
route
task_dir
scene
scene_fingerprint
interaction_registry
commands
```

执行前会验证 bundle 与 `commands.json` 中：

```text
scene
scene_fingerprint
registry
```

完全一致。

这防止手工把：

```text
scene A
+
commands B
```

拼在一起执行。

---

## 15. RobotProfile

RobotProfile 属于 control。

当前 UR5e profile 描述：

- joint names
- actuator names
- end-effector site
- home keyframe
- IK backend
- execution backend

### Preflight order

ControlExecutor 的目标顺序是：

```text
load CommandDocument
   ↓
check scene/registry path
   ↓
verify scene fingerprint
   ↓
load raw MjModel
   ↓
RobotProfile.validate_model
   ↓
derive home from MuJoCo keyframe
   ↓
validate registry MuJoCo sources
   ↓
probe command conversion
   ↓
construct SkillRuntime
```

关键点：

> 先验证 robot model，再初始化 Runtime/Viewer。

这样错误 robot XML 不会先初始化 UR5e controller 后才失败。

### Home

Home 的单一物理真值来自：

```text
RobotProfile.home_keyframe
+
当前 scene.xml 中对应 MuJoCo keyframe
```

Control 在 preflight 时从 `model.key_qpos` 读取当前 scene 的 home joint positions，再注入 runtime registry。

sim 不维护另一份 UR5e home 数字。

---

## 16. ControlExecutor

ControlExecutor 是正式 Python execution API：

```python
ControlExecutor().execute(
    command_document,
    viewer_mode=...,
    output_dir=...,
)
```

它：

1. preflight；
2. 创建单一 `SkillRuntime`；
3. 创建 `SkillCommandConverter`；
4. 顺序执行所有 commands；
5. fail-fast；
6. 输出 `ExecutionReport`；
7. 输出 `skill_trace.jsonl`。

### Persistent Runtime

整个 command sequence 共用：

```text
MjModel
MjData
SceneRobotRuntime
SkillRuntime
```

不能每个 Skill reload XML。

否则：

- 抓住的物体状态丢失；
- 门 joint 状态丢失；
- robot current pose 丢失；
- gripper state 丢失。

---

## 17. Preflight

执行前检查：

- robot backend support
- scene 文件存在
- registry 文件存在
- scene fingerprint
- RobotProfile
- required MuJoCo joints
- required actuators
- end-effector site
- home keyframe
- interaction registry body/site/geom/joint source
- action request / affordance reference
- command conversion

目标是：

> 尽可能在打开 viewer、开始动作之前发现静态错误。

---

## 18. Runtime Failure Model

默认策略是 fail-fast。

例如：

```text
command-003 grasp failed
```

后续：

```text
move destination
release
```

不会继续执行。

`ExecutionFailure` 可以记录：

```text
command_id
source_skill_step_id
runtime_step_id
skill_name
target
error_code
error_message
recoverable
```

常见错误类别包括：

```text
ANCHOR_NOT_FOUND
CONTROL_BACKEND_UNSUPPORTED_ROBOT
EXECUTION_BUNDLE_INCONSISTENT
EXECUTION_CANCELLED_BY_USER
EXECUTION_FAILED
EXECUTION_METADATA_MISSING
INTERNAL_ERROR
INVALID_REQUEST
PERCEPTION_REQUIRED
REGISTRY_INVALID
ROBOT_MODEL_INCOMPATIBLE
RUNTIME_TIMEOUT
SCENE_FINGERPRINT_MISMATCH
SCENE_INVALID
UNSUPPORTED_SKILL
```

---

## 19. Scene Fingerprint

scene fingerprint 由 `robot_agent_protocol` 统一实现。

Compiler 将 fingerprint 写入：

```text
commands.json
execution_bundle.json
interaction_registry.json
```

Control 执行前重新计算实际 scene SHA-256。

如果 scene 在 compile 后被修改：

```text
SCENE_FINGERPRINT_MISMATCH
```

避免旧 registry/commands 在新 XML 上执行。

---

## 20. MuJoCo State Ownership

Planning 与 execution 都可能 load MuJoCo，但职责不同。

### Planning MuJoCo

用于：

- compose/load scene
- RGB render
- segmentation
- geometry/world pose extraction

不是最终 execution state。

### Execution MuJoCo

`ControlExecutor` 创建：

```text
one MjModel
one MjData
```

并在整个 command sequence 中持久存在。

这是最终执行状态。

---

## 21. EGL / GLFW Process Boundary

Planning 需要 offscreen rendering，因此 CLI 在 import MuJoCo planning/rendering module 前设置：

```text
MUJOCO_GL=egl
PYOPENGL_PLATFORM=egl
```

GUI viewer 使用 GLFW。

OpenGL backend 在一个 Python 进程中初始化后再切换不可靠，因此 `robot-agent execute` 使用独立子进程作为 execution boundary。

### headless

child process：

```text
MUJOCO_GL=egl
PYOPENGL_PLATFORM=egl
```

### auto / step viewer

child process会移除：

```text
MUJOCO_GL
PYOPENGL_PLATFORM
```

再启动 GUI viewer。

这不是为了“两个 MuJoCo 实例互相通信”，而是为了隔离 planning/offscreen GL 与 execution/viewer GL 初始化。

---

## 22. Viewer Modes

```text
auto
step
headless
```

`auto`：

- viewer 打开；
- runtime 连续执行。

`step`：

- viewer 打开；
- runtime step 后等待 Space/Enter。

`headless`：

- 无 viewer；
- 用于 CI/SSH/regression。

用户关闭 viewer 时执行应视为取消，而不是成功。

---

## 23. Interaction Metadata Strategy

物理参数来源按优先级区分。

### Generated simple object

由 Route A builder 生成：

```text
generic_default grasp
open_box interior
button surface
```

### Authored interaction metadata

用于：

- cabinet
- door
- drawer
- specific grasp
- mechanism trajectory

### Future external grasp generator

架构允许未来将：

```text
external_grasp_generator
```

作为新的 metadata 来源，但当前没有实现任意物体 grasp pose generator。

---

## 24. Why `SKILL.md` Is Not the Runtime Protocol

Skill 文档用于：

- 人类阅读；
- Agent/Planner 能力语义；
- 策略说明；
- 参数含义；
- Failure Handling。

正式 runtime protocol 是：

```text
SkillPlan
→ Compiler
→ CommandDocument
→ ControlExecutor
```

运行时不应每次解析长 Markdown 来决定物理控制参数。

这样可以：

- 减少 token；
- 避免 Markdown/schema 漂移直接影响控制；
- 保持执行确定性；
- 让 command fixture 可直接回归。

---

## 25. Testing Layers

### Package unit tests

```text
robot_agent_protocol/tests
robot_agent_sim/tests
robot_agent_control/tests
```

### Integration tests

```text
tests/integration
```

验证 package 接口，例如：

```text
SkillPlan
→ Compiler
→ CommandDocument
```

### E2E

```text
tests/e2e
```

验证完整 headless 链路。

### Legacy fixture

保留：

```text
config_001.commands.json
```

直接进入 ControlExecutor，确保原控制能力不会在上层整合时退化。

### Wheel install

CI 构建三个 wheel，并在 clean venv、`/tmp` cwd 中验证：

- RobotProfile package resource
- ExecutionProfile package resource
- `robot-agent`
- `robot-agent-control`

目的是保证项目不是只在源码 checkout + editable install 时可用。

---

## 26. Current Backend Boundary

当前明确支持：

```text
Planning:
Panda ✅
UR5e ✅

Execution:
UR5e ✅
Panda ❌
```

不要通过把 Panda command 标记成 UR5e 来绕过限制。

后续 Panda backend 应通过 RobotRuntime abstraction / RobotProfile 驱动，而不是继续在通用代码散落 `if robot == ...`。

---

## 27. Extension Points

后续功能应在现有边界内扩展：

### Grasp pose generation

推荐替换：

```text
generic_default
```

为：

```text
asset_authored
external_grasp_generator
```

而不改变 SkillPlan/Control 主协议。

### Active Search

当前 `search` 不能进入 Control。

未来可以实现：

```text
search
→ camera/view update
→ render
→ vision grounding
→ GroundedTask update
→ recompile
```

### Panda

增加 Panda RobotProfile/Runtime，不改变 protocol。

### Real Robot

未来 hardware backend 应复用：

```text
SkillPlan
CommandDocument
InteractionRegistry
ExecutionReport
```

替换 MuJoCo-specific runtime。

---

## 28. Architectural Invariants

后续修改必须尽量保持：

1. `robot_agent_protocol` 不依赖 sim/control。
2. Control 不依赖 Qwen/TaskIntent。
3. Compiler 不调用模型。
4. 明确 semantic anchor 缺失时 fail-closed。
5. scene/commands/registry 必须 fingerprint 一致。
6. 一个 command sequence 使用一个 persistent execution runtime。
7. execute 可以完全离线重放。
8. LLM 不输出 joint/trajectory/actuator。
9. 复杂 mechanism metadata 不由通用 VLM 凭空猜测。
10. package 内置配置必须可通过 wheel 独立部署。
