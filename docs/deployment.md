# Deployment

完整执行链只需要一个 clone 和一个 Conda 环境。安装顺序是 protocol → control → sim。
真实 Qwen/llama.cpp 是可选的 planner 服务；headless execute 不需要它。当前控制后端是
UR5e，Panda 仍可规划但执行时返回不支持错误。旧的 `robot_agent_sim` 和
`robot_agent_control` 仓库作为迁移源保留，新功能应只提交到本仓库。
