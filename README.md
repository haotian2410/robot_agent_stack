# robot_agent_stack

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
