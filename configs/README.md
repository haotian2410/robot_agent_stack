# Optional configuration overrides

Built-in RobotProfile and ExecutionProfile resources are shipped inside the
`robot_agent_control` and `robot_agent_sim` packages. The repository root does
not need to be present at runtime. To override them, set:

```bash
export ROBOT_AGENT_CONFIG_ROOT=/path/to/configs
```

and provide `robots/<robot>.json` and/or `execution_profiles/<name>.json`.
