# Optional configuration overrides

The built-in RobotProfile resource is shipped inside the
`robot_agent_control` package. The repository root does not need to be present
at runtime. To override it, set:

```bash
export ROBOT_AGENT_CONFIG_ROOT=/path/to/configs
```

and provide `robots/<robot>.json`.
