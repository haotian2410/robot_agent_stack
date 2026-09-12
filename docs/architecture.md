# Architecture

## Design goals and package boundaries

`robot_agent_protocol` is the only source of cross-package schemas and depends
only on lightweight Python/Pydantic. `robot_agent_sim` owns understanding,
Route A/B, grounding, planning and deterministic compilation. `robot_agent_control`
owns IK, collision, trajectory, gripper skills, viewer and execution. The
dependency direction is `sim → protocol`, `control → protocol`; only the
execution adapter imports the control public API.

## End-to-end data flow

```text
Instruction → TaskIntent → GroundedTask → SkillPlan
           → InteractionRegistry + ExecutionProfile
           → CommandDocument → ExecutionBundle
           → ControlExecutor → one MjModel/MjData pair
           → ExecutionReport + skill_trace.jsonl
```

LLM output is semantic only. The compiler never calls an LLM and never invents
XYZ/joint trajectories. InteractionRegistry supplies anchors, affordances and
authored mechanism metadata; Control selects IK, collision checks and motion.

## Routes and registries

Route A creates a deterministic MuJoCo scene and reads object poses back from
the final model before generating interaction metadata. Simple graspable
objects use `generic_default` grasp provenance. Route B consumes an existing
XML, renders RGB/segmentation, performs grounding, and prefers an authored
interaction sidecar. Grounding does not infer arbitrary door/drawer mechanics;
those require explicit joint/axis/handle/travel metadata.

`SceneRegistry` describes scene identity and detections. `InteractionRegistry`
describes execution semantics: anchors, action requests and mechanism metadata.

## Contracts and execution

`CommandDocument` is the versioned command list. `ExecutionBundle` binds it to
an absolute scene, registry and SHA-256 fingerprint. Before runtime construction,
ControlExecutor validates the RobotProfile (joints, actuators, EE site and home
keyframe), registry sources, commands and bundle references. `ExecutionOptions`
is robot-neutral; an omitted EE site is filled from the selected RobotProfile.

Semantic steps may expand into runtime micro-steps (for example deterministic
lift/retreat macros from ExecutionProfile). All steps share a persistent
`SkillRuntime` owning one MuJoCo `MjModel` and `MjData`. Failures are serialized
as `ExecutionReport`; malformed process input uses the protocol `ProcessFailure`
envelope.

## Profiles, fingerprinting and graphics

RobotProfile and ExecutionProfile are package resources, with optional
`ROBOT_AGENT_CONFIG_ROOT` overrides. Home joint positions come only from the
configured MuJoCo keyframe in the actual scene. Scene fingerprinting has one
implementation in protocol. Headless planning/rendering and interactive viewer
use separate processes/backends (EGL/OSMesa versus GLFW) to avoid GL backend
switching in one process.

## Error propagation and limitations

Stable `ErrorCode` values cover invalid requests, missing anchors/metadata,
fingerprint mismatches, unsupported robots/skills, timeouts and runtime
failures. Panda remains planning-only; generic grasp is an MVP fallback;
search is not yet a complete active-perception loop; arbitrary articulated
mechanism inference and real hardware backends are not implemented.
