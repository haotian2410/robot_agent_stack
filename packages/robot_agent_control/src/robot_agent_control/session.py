"""Persistent MuJoCo control session for multi-turn orchestration."""

from __future__ import annotations

import json
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import imageio.v2 as imageio
from robot_agent_protocol import ErrorCode, scene_sha256

from .command.converter import SkillCommandConverter
from .contracts import CommandDocument, ExecutionFailure, ExecutionReport, RuntimeStepReport, ViewerMode, load_command_document
from .executor import ControlExecutor


class ControlSession:
    """Own one persistent ``SkillRuntime`` and execute many command bundles."""

    def __init__(self, command_document: CommandDocument | str | Path, *, headless: bool = True, viewer_mode: ViewerMode | str | None = None) -> None:
        self.executor = ControlExecutor()
        self.document = self._load(command_document)
        self.viewer_mode = ViewerMode.HEADLESS if headless and viewer_mode is None else ViewerMode(viewer_mode or ViewerMode.AUTO)
        self.registry, self.runtime = self.executor._preflight(self.document, headless=self.viewer_mode == ViewerMode.HEADLESS)
        self._viewer_context = None
        self._viewer = None
        self._viewer_thread = None
        self._viewer_stop = None
        # MuJoCo does not allow viewer.sync() to copy visual state while the
        # execution thread is mutating the same mjData.  The passive viewer
        # and all public state/render operations share this lock.
        self._mujoco_lock = threading.RLock()
        if self.viewer_mode != ViewerMode.HEADLESS:
            self._open_viewer()
        self.closed = False

    def _open_viewer(self) -> None:
        import threading
        import time
        import mujoco.viewer

        self._viewer_stop = threading.Event()
        self._continue_event = threading.Event()

        def on_key(keycode: int) -> None:
            if keycode in {32, 257}:
                self._continue_event.set()

        self._viewer_context = mujoco.viewer.launch_passive(self.runtime.model, self.runtime.data, key_callback=on_key)
        self._viewer = self._viewer_context.__enter__()
        self.executor._configure_camera(self._viewer, self.runtime)
        self.runtime.runtime.attach_viewer(self._viewer)
        with self._mujoco_lock:
            self._viewer.sync()
        def sync_loop() -> None:
            while self._viewer is not None and self._viewer.is_running() and not self._viewer_stop.is_set():
                with self._mujoco_lock:
                    if self._viewer is None or not self._viewer.is_running():
                        break
                    self._viewer.sync()
                # ``SkillRuntime`` owns the simulation wrapper as
                # ``runtime``; the playback rate is a property of that
                # wrapper, not of ``SkillRuntime`` itself.
                time.sleep(1.0 / max(self.runtime.runtime.playback_fps, 1.0))

        self._viewer_thread = threading.Thread(target=sync_loop, name="robot-agent-viewer", daemon=True)
        self._viewer_thread.start()

    def _close_viewer(self) -> None:
        if self._viewer_stop is not None:
            self._viewer_stop.set()
        if self._viewer_thread is not None:
            self._viewer_thread.join(timeout=2)
        if self.runtime is not None:
            self.runtime.runtime.attach_viewer(None)
        if self._viewer_context is not None:
            self._viewer_context.__exit__(None, None, None)
        self._viewer_context = self._viewer = self._viewer_thread = self._viewer_stop = self._continue_event = None

    @staticmethod
    def _load(value: CommandDocument | str | Path) -> CommandDocument:
        return load_command_document(value) if isinstance(value, (str, Path)) else CommandDocument.model_validate(value)

    def execute(self, command_document: CommandDocument | str | Path, *, output_dir: str | Path | None = None) -> ExecutionReport:
        """Execute a document on the already-open model/data pair."""
        if self.closed:
            raise RuntimeError("control session is closed")
        document = self._load(command_document)
        if Path(document.scene).resolve() != Path(self.document.scene).resolve():
            raise ValueError("persistent control session cannot switch scenes; reload the session")
        started = datetime.now(UTC)
        reports: list[RuntimeStepReport] = []
        state: dict[str, Any] = {"started": 0, "completed": 0, "failure": None}
        trace_path = Path(output_dir).resolve() / "skill_trace.jsonl" if output_dir else None
        if trace_path:
            trace_path.parent.mkdir(parents=True, exist_ok=True)
            trace_path.write_text("", encoding="utf-8")
        converter = SkillCommandConverter(
            self.registry.data,
            collision_checker=self.runtime.approach_checker,
            pose_provider=self.runtime.pose_provider,
        )
        try:
            viewer_state = (self._viewer, self._continue_event) if self._viewer is not None else None
            with self._mujoco_lock:
                self.executor._run(document, self.registry, self.runtime, converter, reports, state, viewer_state, self.viewer_mode, trace_path)
        except Exception as exc:
            state["failure"] = ExecutionFailure(error_code=ErrorCode.INTERNAL_ERROR, error_message=str(exc), recoverable=False)
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
            out = Path(output_dir).resolve()
            out.mkdir(parents=True, exist_ok=True)
            (out / "execution_report.json").write_text(report.model_dump_json(indent=2), encoding="utf-8")
        return report

    def snapshot(self, *, turn_index: int = 0, scene_version: int = 1, world_version: int = 0) -> dict[str, Any]:
        """Return a JSON-safe snapshot directly from the live MuJoCo ``MjData``."""
        with self._mujoco_lock:
            runtime = self.runtime.runtime
            mujoco.mj_forward(runtime.model, runtime.data)
            objects: dict[str, Any] = {}
            for object_id, item in self.registry.objects.items():
                body_name = item.get("body_name")
                body_id = mujoco.mj_name2id(runtime.model, mujoco.mjtObj.mjOBJ_BODY, str(body_name)) if body_name else -1
                if body_id < 0:
                    continue
                objects[object_id] = {
                    "object_id": object_id,
                    "body_name": body_name,
                    "position": [float(v) for v in runtime.data.xpos[body_id]],
                    "quaternion": [float(v) for v in runtime.data.xquat[body_id]],
                }
            joints: dict[str, Any] = {}
            for joint_id in range(runtime.model.njnt):
                name = mujoco.mj_id2name(runtime.model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
                if not name:
                    continue
                qpos_address = int(runtime.model.jnt_qposadr[joint_id])
                qvel_address = int(runtime.model.jnt_dofadr[joint_id])
                joints[name] = {"name": name, "position": float(runtime.data.qpos[qpos_address]), "velocity": float(runtime.data.qvel[qvel_address])}
            held_object = None
            gripper = getattr(runtime, "gripper_controller", None)
            held_body = getattr(gripper, "_held_body_id", None)
            if held_body is not None:
                held_name = mujoco.mj_id2name(runtime.model, mujoco.mjtObj.mjOBJ_BODY, int(held_body))
                held_object = next((oid for oid, item in self.registry.objects.items() if item.get("body_name") == held_name), None)
            return {
            "world_version": world_version,
            "scene_version": scene_version,
            "turn_index": turn_index,
            "sim_time": float(runtime.data.time),
            "objects": objects,
            "joints": joints,
            "robot_qpos": [float(v) for v in runtime.get_joint_positions()],
            "robot_qvel": [float(runtime.data.qvel[i]) for i in runtime.dof_indices],
            "held_object": held_object,
            }

    def observe(self, output_dir: str | Path) -> dict[str, Any]:
        with self._mujoco_lock:
            return self._observe_unlocked(output_dir)

    def _observe_unlocked(self, output_dir: str | Path) -> dict[str, Any]:
        """Render RGB/segmentation from the live model/data pair."""
        out = Path(output_dir).expanduser().resolve()
        out.mkdir(parents=True, exist_ok=True)
        runtime = self.runtime.runtime
        mujoco.mj_forward(runtime.model, runtime.data)
        renderer = mujoco.Renderer(runtime.model, height=480, width=640)
        camera = "scene_camera" if mujoco.mj_name2id(runtime.model, mujoco.mjtObj.mjOBJ_CAMERA, "scene_camera") >= 0 else -1
        renderer.update_scene(runtime.data, camera=camera)
        rgb = renderer.render().copy()
        renderer.enable_segmentation_rendering()
        renderer.update_scene(runtime.data, camera=camera)
        segmentation = renderer.render().copy()
        renderer.close()
        rgb_path = out / "rgb.png"
        segmentation_path = out / "segmentation.npy"
        segmentation_visualization_path = out / "segmentation.png"
        imageio.imwrite(rgb_path, rgb)
        np.save(segmentation_path, segmentation)
        imageio.imwrite(segmentation_visualization_path, segmentation[:, :, :3].astype(np.uint8))
        instances = []
        for object_id, item in self.registry.objects.items():
            body_name = item.get("body_name")
            body_id = mujoco.mj_name2id(runtime.model, mujoco.mjtObj.mjOBJ_BODY, str(body_name)) if body_name else -1
            if body_id < 0:
                continue
            body_ids = {body_id}
            for candidate in range(1, runtime.model.nbody):
                parent = candidate
                while parent > 0:
                    if parent == body_id:
                        body_ids.add(candidate)
                        break
                    parent = int(runtime.model.body_parentid[parent])
            geom_ids = [gid for gid in range(runtime.model.ngeom) if int(runtime.model.geom_bodyid[gid]) in body_ids]
            mask = np.isin(segmentation[:, :, 0], geom_ids)
            ys, xs = np.nonzero(mask)
            bbox = None if not len(xs) else [round(float(ys.min()) * 1000 / 480), round(float(xs.min()) * 1000 / 640), round(float(ys.max() + 1) * 1000 / 480), round(float(xs.max() + 1) * 1000 / 640)]
            instances.append({"object_id": object_id, "body_name": body_name, "bbox": bbox, "visible_pixel_count": int(mask.sum()), "world_position": [float(value) for value in runtime.data.xpos[body_id]]})
        instances_path = out / "instances.json"
        instances_path.write_text(json.dumps(instances, ensure_ascii=False, indent=2), encoding="utf-8")
        return {"rgb_path": str(rgb_path), "segmentation_path": str(segmentation_path), "segmentation_visualization_path": str(segmentation_visualization_path), "instances_path": str(instances_path), "instances": instances}

    def reload(self, command_document: CommandDocument | str | Path) -> dict[str, Any]:
        """Reload a changed scene and restore state by stable joint/body names."""
        if self.closed:
            raise RuntimeError("control session is closed")
        transfer = self._transfer_state()
        self._close_viewer()
        self.runtime.approach_checker.close()
        document = self._load(command_document)
        registry, runtime = self.executor._preflight(document, headless=self.viewer_mode == ViewerMode.HEADLESS)
        self.document = document
        self.registry = registry
        self.runtime = runtime
        self._restore_state(transfer)
        if self.viewer_mode != ViewerMode.HEADLESS:
            self._open_viewer()
        return self.snapshot()

    def reload_scene(self, scene: str | Path, registry: str | Path, *, robot: str | None = None) -> dict[str, Any]:
        """Reload topology without inventing an executable command.

        Empty-object scenes are valid session states; their reload path must
        not manufacture a locate command merely to satisfy the one-shot
        command-document contract.
        """
        if self.closed:
            raise RuntimeError("control session is closed")
        transfer = self._transfer_state()
        self._close_viewer()
        self.runtime.approach_checker.close()
        document = self.document.model_copy(update={
            "robot": robot or self.document.robot,
            "scene": str(Path(scene).expanduser().resolve()),
            "registry": str(Path(registry).expanduser().resolve()),
            "scene_fingerprint": scene_sha256(Path(scene)),
            "commands": [],
        })
        registry_obj, runtime = self.executor._preflight(document, headless=self.viewer_mode == ViewerMode.HEADLESS)
        self.document = document
        self.registry = registry_obj
        self.runtime = runtime
        self._restore_state(transfer)
        if self.viewer_mode != ViewerMode.HEADLESS:
            self._open_viewer()
        return self.snapshot()

    def _transfer_state(self) -> dict[str, Any]:
        runtime = self.runtime.runtime
        joints: dict[str, dict[str, list[float]]] = {}
        for joint_id in range(runtime.model.njnt):
            name = mujoco.mj_id2name(runtime.model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
            if not name:
                continue
            qpos_width = {int(mujoco.mjtJoint.mjJNT_FREE): 7, int(mujoco.mjtJoint.mjJNT_BALL): 4}.get(int(runtime.model.jnt_type[joint_id]), 1)
            qvel_width = {int(mujoco.mjtJoint.mjJNT_FREE): 6, int(mujoco.mjtJoint.mjJNT_BALL): 3}.get(int(runtime.model.jnt_type[joint_id]), 1)
            qa = int(runtime.model.jnt_qposadr[joint_id]); da = int(runtime.model.jnt_dofadr[joint_id])
            joints[name] = {"qpos": runtime.data.qpos[qa:qa + qpos_width].copy().tolist(), "qvel": runtime.data.qvel[da:da + qvel_width].copy().tolist()}
        return {"joints": joints, "held_object": self.snapshot().get("held_object"), "time": float(runtime.data.time)}

    def _restore_state(self, transfer: dict[str, Any]) -> None:
        runtime = self.runtime.runtime
        for name, values in transfer["joints"].items():
            joint_id = mujoco.mj_name2id(runtime.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if joint_id < 0:
                continue
            qa = int(runtime.model.jnt_qposadr[joint_id]); da = int(runtime.model.jnt_dofadr[joint_id])
            qpos = np.asarray(values["qpos"], dtype=float); qvel = np.asarray(values["qvel"], dtype=float)
            runtime.data.qpos[qa:qa + len(qpos)] = qpos
            runtime.data.qvel[da:da + len(qvel)] = qvel
        runtime.data.time = float(transfer.get("time", 0.0))
        mujoco.mj_forward(runtime.model, runtime.data)
        held_object = transfer.get("held_object")
        gripper = getattr(runtime, "gripper_controller", None)
        if held_object and gripper is not None and held_object in self.registry.objects:
            gripper._attach_object(held_object)
            gripper._holding = getattr(gripper, "_held_body_id", None) is not None
        elif gripper is not None:
            gripper._holding = False
            gripper._detach_object()
        self.runtime.refresh_move_skill()

    def close(self) -> None:
        if not self.closed:
            self._close_viewer()
            self.runtime.approach_checker.close()
            self.closed = True


__all__ = ["ControlSession"]
