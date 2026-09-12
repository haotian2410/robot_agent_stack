# Deployment

完整执行链只需要一个 clone 和一个 Conda 环境。安装顺序是 protocol → control → sim。
真实 Qwen/llama.cpp 是可选的 planner 服务；headless execute 不需要它。当前控制后端是
UR5e，Panda 仍可规划但执行时返回不支持错误。旧的 `robot_agent_sim` 和
`robot_agent_control` 仓库作为迁移源保留，新功能应只提交到本仓库。

开发安装：

```bash
conda activate robot_agent_integ
pip install -e packages/robot_agent_protocol
pip install -e packages/robot_agent_control
pip install -e 'packages/robot_agent_sim[dev]'
```

wheel 部署不需要保留 monorepo 根目录；内置 RobotProfile 和 ExecutionProfile 随
package 安装。需要覆盖时设置 `ROBOT_AGENT_CONFIG_ROOT`，并提供
`robots/<robot>.json` 或 `execution_profiles/<name>.json`。
