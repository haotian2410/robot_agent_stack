"""Formal execution boundary for versioned skill-command documents."""

from __future__ import annotations

import json
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import mujoco

from robot_agent_control.command.registry import SceneRegistry
from robot_agent_control.command.converter import CommandConversionError, SkillCommandConverter
from robot_agent_control.command.runtime import SkillRuntime
from robot_agent_control.robot_profile import RobotProfile
from robot_agent_protocol import ErrorCode

from .contracts import (
    CommandDocument,
    ExecutionFailure,
    ExecutionReport,
    RuntimeStepReport,
    ViewerMode,
    load_command_document,
    scene_sha256,
)


class ExecutionPreflightError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


class ControlExecutor:
    """Execute all commands on one persistent MuJoCo model and data pair."""

    def execute(
        self,
        command_document: CommandDocument | str | Path,
        *,
        viewer_mode: ViewerMode | str = ViewerMode.AUTO,
        output_dir: str | Path | None = None,
    ) -> ExecutionReport:
        started = datetime.now(UTC)
        document = (
            load_command_document(command_document)
            if isinstance(command_document, (str, Path))
            else command_document
        )
        mode = ViewerMode(viewer_mode)
        try:
            registry, session = self._preflight(document, headless=mode == ViewerMode.HEADLESS)
        except Exception as exc:
            code = getattr(exc, "code", "INTERNAL_ERROR")
            first = document.commands[0] if document.commands else None
            report = ExecutionReport(
                success=False,
                robot=document.robot,
                scene=document.scene,
                started_at=started,
                finished_at=datetime.now(UTC),
                commands_total=len(document.commands),
                commands_started=0,
                commands_completed=0,
                steps=[],
                failure=ExecutionFailure(
                    command_id=first.command_id if first else None,
                    source_skill_step_id=first.source_skill_step_id if first else None,
                    runtime_step_id=None,
                    skill_name=first.skill_name if first else None,
                    target=str(first.parameters.get("target", "")) if first else None,
                    error_code=str(code), error_message=str(exc), recoverable=False,
                ),
            )
            if output_dir:
                output = Path(output_dir).resolve()
                output.mkdir(parents=True, exist_ok=True)
                (output / "execution_report.json").write_text(report.model_dump_json(indent=2), encoding="utf-8")
            return report
        trace_path = Path(output_dir).resolve() / "skill_trace.jsonl" if output_dir else None
        if trace_path:
            trace_path.parent.mkdir(parents=True, exist_ok=True)
            trace_path.write_text("", encoding="utf-8")

        converter = SkillCommandConverter(
            registry.data,
            collision_checker=session.approach_checker,
            pose_provider=session.pose_provider,
        )
        converter.reset()
        reports: list[RuntimeStepReport] = []
        state = {"started": 0, "completed": 0, "failure": None}

        if mode == ViewerMode.HEADLESS:
            try:
                self._run(document, registry, session, converter, reports, state, None, mode, trace_path)
            except Exception as exc:
                state["failure"] = ExecutionFailure(
                    error_code=ErrorCode.INTERNAL_ERROR, error_message=str(exc), recoverable=False
                )
        else:
            import mujoco.viewer

            continue_event = threading.Event()

            def on_key(keycode: int) -> None:
                if keycode in {32, 257}:
                    continue_event.set()

            with mujoco.viewer.launch_passive(
                session.model, session.data, key_callback=on_key
            ) as viewer:
                self._configure_camera(viewer, session.runtime)
                session.runtime.attach_viewer(viewer)
                viewer.sync()
                try:
                    self._run(
                        document, registry, session, converter, reports, state,
                        (viewer, continue_event), mode, trace_path,
                    )
                except Exception as exc:
                    state["failure"] = ExecutionFailure(
                        error_code=ErrorCode.INTERNAL_ERROR, error_message=str(exc), recoverable=False
                    )
                while viewer.is_running():
                    viewer.sync()
                    time.sleep(1.0 / session.runtime.playback_fps)
                session.runtime.attach_viewer(None)

        report = ExecutionReport(
            success=state["failure"] is None,
            robot=document.robot,
            scene=document.scene,
            started_at=started,
            finished_at=datetime.now(UTC),
            commands_total=len(document.commands),
            commands_started=state["started"],
            commands_completed=state["completed"],
            steps=reports,
            failure=state["failure"],
        )
        if output_dir:
            output = Path(output_dir).resolve()
            output.mkdir(parents=True, exist_ok=True)
            (output / "execution_report.json").write_text(
                report.model_dump_json(indent=2), encoding="utf-8"
            )
        return report

    def _preflight(
        self, document: CommandDocument, *, headless: bool
    ) -> tuple[SceneRegistry, SkillRuntime]:
        if document.robot != "ur5e":
            raise ExecutionPreflightError(
                "CONTROL_BACKEND_UNSUPPORTED_ROBOT",
                "control execution currently supports ur5e only",
            )
        scene = Path(document.scene)
        registry_path = Path(document.registry)
        if not scene.is_file():
            raise ExecutionPreflightError("SCENE_INVALID", f"scene not found: {scene}")
        if not registry_path.is_file():
            raise ExecutionPreflightError(
                "REGISTRY_INVALID", f"interaction registry not found: {registry_path}"
            )
        actual_hash = scene_sha256(scene)
        if actual_hash != document.scene_fingerprint:
            raise ExecutionPreflightError(
                "SCENE_FINGERPRINT_MISMATCH",
                f"scene fingerprint mismatch: expected {document.scene_fingerprint}, got {actual_hash}",
            )
        registry = SceneRegistry(registry_path, scene_path=scene)
        # Preflight against a raw model before constructing SkillRuntime.  This
        # keeps an incompatible robot from initializing controllers or viewer
        # state as a side effect of validation.
        preflight_model = mujoco.MjModel.from_xml_path(str(scene))
        try:
            profile = RobotProfile.load_for_robot(document.robot)
            profile.validate_model(preflight_model, document.runtime.end_effector_site)
            registry.data.setdefault("named_targets", {}).setdefault("home", {
                "aliases": ["home", "初始位"], "request": {"target": {"type": "joint"}}
            })["request"]["target"]["joint_positions"] = profile.home_joint_positions(preflight_model)
        except Exception as exc:
            raise ExecutionPreflightError("ROBOT_MODEL_INCOMPATIBLE", str(exc)) from exc
        self._validate_registry_sources(preflight_model, registry)
        probe = SkillCommandConverter(registry.data)
        for index, command in enumerate(document.commands, 1):
            try:
                probe.convert_command(command.model_dump(), index)
            except CommandConversionError as exc:
                message = str(exc)
                if "unsupported skill" in message:
                    code = ErrorCode.UNSUPPORTED_SKILL
                elif "action_requests" in message or "anchor" in message:
                    code = ErrorCode.EXECUTION_METADATA_MISSING
                else:
                    code = ErrorCode.INVALID_REQUEST
                raise ExecutionPreflightError(code, message) from exc
        runtime = document.runtime.model_dump()
        runtime["realtime"] = not headless
        if headless:
            runtime["minimum_playback_duration"] = 0.0
        session = SkillRuntime(registry, runtime)
        return registry, session

    @staticmethod
    def _validate_registry_sources(model: mujoco.MjModel, registry: SceneRegistry) -> None:
        """Resolve every authored body/site/geom source before opening a viewer."""
        missing: list[str] = []
        valid_types = {
            "body": mujoco.mjtObj.mjOBJ_BODY,
            "site": mujoco.mjtObj.mjOBJ_SITE,
            "geom": mujoco.mjtObj.mjOBJ_GEOM,
            "joint": mujoco.mjtObj.mjOBJ_JOINT,
        }
        for object_key, obj in registry.objects.items():
            source = obj.get("spatial", {}).get("source", {})
            source_type = source.get("type")
            source_name = source.get("name")
            if source_type not in valid_types or not isinstance(source_name, str):
                missing.append(f"{object_key}:invalid_source")
                continue
            if mujoco.mj_name2id(model, valid_types[source_type], source_name) < 0:
                missing.append(f"{object_key}:{source_type}:{source_name}")
            body_name = obj.get("body_name")
            if isinstance(body_name, str) and mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_BODY, body_name
            ) < 0:
                missing.append(f"{object_key}:body:{body_name}")
            for action, spec in obj.get("affordances", {}).items():
                if not isinstance(spec, dict):
                    continue
                acting = str(spec.get("acting_target", object_key))
                action_request = registry.objects.get(acting, {}).get("action_requests", {}).get(action)
                if action_request is None:
                    missing.append(f"{object_key}:action_request:{acting}.{action}")
        if missing:
            raise ExecutionPreflightError(
                "REGISTRY_INVALID",
                "registry references missing MuJoCo names: " + ", ".join(missing),
            )

    @staticmethod
    def _validate_robot(model: mujoco.MjModel, end_effector_site: str) -> None:
        required_joints = (
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint",
        )
        missing = [
            name
            for name in required_joints
            if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) < 0
        ]
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, end_effector_site) < 0:
            missing.append(end_effector_site)
        if missing:
            raise ExecutionPreflightError(
                "ROBOT_MODEL_INCOMPATIBLE", f"missing UR5e model names: {missing}"
            )

    def _run(
        self,
        document: CommandDocument,
        registry: SceneRegistry,
        session: SkillRuntime,
        converter: SkillCommandConverter,
        reports: list[RuntimeStepReport],
        state: dict[str, Any],
        viewer_state: tuple[Any, threading.Event] | None,
        mode: ViewerMode,
        trace_path: Path | None,
    ) -> None:
        runtime_index = 0
        for command_index, command in enumerate(document.commands, 1):
            viewer = viewer_state[0] if viewer_state else None
            if viewer is not None and not viewer.is_running():
                state["failure"] = self._failure(
                    command, None, "EXECUTION_CANCELLED_BY_USER", "viewer closed by user"
                )
                return
            state["started"] += 1
            session.update()
            converted = converter.convert_command(command.model_dump(), command_index)
            steps = [] if converted is None else (converted if isinstance(converted, list) else [converted])
            if not steps:
                state["completed"] += 1
                now = datetime.now(UTC)
                self._trace(trace_path, command, None, True, 0.0, None, now, now)
                continue
            for step in steps:
                runtime_index += 1
                runtime_step_id = f"runtime-step-{runtime_index:03d}"
                before = datetime.now(UTC)
                try:
                    result = session.execute_step(step, document.request_defaults)
                except (ValueError, TimeoutError, RuntimeError, OSError) as exc:
                    after = datetime.now(UTC)
                    duration = (after - before).total_seconds()
                    state["failure"] = self._failure(
                        command, runtime_step_id, self._error_code(exc), str(exc), False
                    )
                    self._trace(trace_path, command, runtime_step_id, False, duration, {"error": {"error_code": state["failure"].error_code, "error_message": str(exc)}}, before, after)
                    return
                after = datetime.now(UTC)
                duration = (after - before).total_seconds()
                success = bool(result.get("success"))
                report = RuntimeStepReport(
                    runtime_step_id=runtime_step_id,
                    command_id=command.command_id,
                    source_skill_step_id=command.source_skill_step_id,
                    skill=command.skill_name,
                    target=str(command.parameters["target"]),
                    success=success,
                    started_at=before,
                    finished_at=after,
                    duration_seconds=duration,
                    result=result,
                )
                reports.append(report)
                self._trace(
                    trace_path, command, runtime_step_id, success, duration,
                    result, before, after,
                )
                if not success:
                    error = result.get("error") or {}
                    state["failure"] = self._failure(
                        command,
                        runtime_step_id,
                        str(error.get("error_code", "INTERNAL_ERROR")),
                        str(error.get("error_message", "control step failed")),
                        bool(error.get("recoverable", False)),
                    )
                    return
                if viewer_state and mode == ViewerMode.STEP:
                    viewer, event = viewer_state
                    event.clear()
                    while viewer.is_running() and not event.is_set():
                        viewer.sync()
                        time.sleep(1.0 / session.runtime.playback_fps)
                    if not viewer.is_running():
                        state["failure"] = self._failure(
                            command, runtime_step_id, "EXECUTION_CANCELLED_BY_USER", "viewer closed by user"
                        )
                        return
            state["completed"] += 1

    @staticmethod
    def _failure(command, runtime_step_id, code, message, recoverable=False):
        return ExecutionFailure(
            command_id=command.command_id,
            source_skill_step_id=command.source_skill_step_id,
            runtime_step_id=runtime_step_id,
            skill_name=command.skill_name,
            target=str(command.parameters.get("target", "")),
            error_code=code,
            error_message=message,
            recoverable=recoverable,
        )

    @staticmethod
    def _error_code(exc: Exception) -> str:
        text = str(exc).lower()
        if isinstance(exc, TimeoutError) or "timeout" in text:
            return ErrorCode.RUNTIME_TIMEOUT
        if "ik" in text:
            return ErrorCode.EXECUTION_FAILED
        if "collision" in text:
            return ErrorCode.EXECUTION_FAILED
        if "gripper" in text:
            return ErrorCode.EXECUTION_FAILED
        return ErrorCode.INTERNAL_ERROR

    @staticmethod
    def _configure_camera(viewer: Any, runtime: Any) -> None:
        viewer.cam.lookat[:] = runtime.model.stat.center
        viewer.cam.distance = max(1.4 * runtime.model.stat.extent, 1.0)
        viewer.cam.azimuth = 135.0
        viewer.cam.elevation = -25.0

    @staticmethod
    def _trace(
        path, command, runtime_step_id, success, duration, result,
        started_at, finished_at,
    ):
        if path is None:
            return
        event = {
            "timestamp": datetime.now(UTC).isoformat(),
            "source_skill_step_id": command.source_skill_step_id,
            "command_id": command.command_id,
            "runtime_step_id": runtime_step_id,
            "skill": command.skill_name,
            "target": command.parameters.get("target"),
            "started": True,
            "completed": bool(success),
            "success": success,
            "started_at": started_at.isoformat(),
            "finished_at": finished_at.isoformat(),
            "duration_seconds": duration,
            "error": (result or {}).get("error") if result else None,
        }
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, ensure_ascii=False) + "\n")
