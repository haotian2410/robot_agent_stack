# Architecture

```text
Instruction → robot_agent_sim → TaskIntent → GroundedTask → SkillPlan
           → Compiler → ExecutionBundle → robot_agent_control
           → ControlExecutor → persistent MuJoCo runtime → ExecutionReport
```

`robot_agent_protocol` 只包含 Pydantic 数据契约、错误码和文件 fingerprint；它不依赖
MuJoCo、Qwen、OpenGL、IK 或轨迹库。正式代码位于 `src/robot_agent_*`，`demo/` 只保留
fixture、示例 CLI 和 sample registry。
