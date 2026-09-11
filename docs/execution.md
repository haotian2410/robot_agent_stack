# Execution contract

`robot-agent compile` produces an absolute-path `ExecutionBundle` with a scene
SHA-256 fingerprint and interaction-registry reference. `robot-agent execute`
validates those three references before constructing `SkillRuntime`, then runs
the complete command sequence on one persistent `MjModel`/`MjData` pair.

Every failure is serialized as `execution_report.json` and includes the command,
source step, runtime step, skill, target, error code, message, and recoverability
when that context exists.
