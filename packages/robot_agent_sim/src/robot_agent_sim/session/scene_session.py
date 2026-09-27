"""Persistent scene lifecycle shared by generated and uploaded scenes."""

from __future__ import annotations

import json
import re
import uuid
from pathlib import Path
from typing import Any

from ..execution.compiler import compile_directory
from ..execution.compiler import compile_execution_bundle
from ..execution.placement_allocator import allocate_interior_slots
from ..execution.placement.resolver import PlacementResolver
from ..execution.placement.geometry import quaternion_inverse_rotate
from ..contracts.placement import PlacementTargetKind, PlacementTargetSpec
from ..semantics.placement_normalizer import TABLE_ENTITY
from .batch_executor import slice_skill_plan_for_operation
from ..contracts.grounded_task import GroundedTask
from ..contracts.skill_plan import SkillPlan
from ..execution.session_client import SessionExecutorClient
from ..pipeline.engine import PipelineEngine, PipelineResult
from ..models.budget import ModelCallMode
from ..scene.mutator import SceneMutator
from ..scene.registry import SceneRegistry
from ..grounding.segmentation import InstanceObservation, SceneObservation
from ..grounding.name_matching import exact_name_match
from .contracts import DialogueState, ObjectWorldState, SceneEditIntent, SceneEditType, SceneQueryIntent, SceneQueryType, SemanticMap, SemanticObject, SessionControlType, TurnKind, WorldState
from .referent_binder import DialogueBinding, ReferentBinder


class SceneQueryAmbiguous(ValueError):
    """A query needs one object but matched multiple scene instances."""


class SceneSession:
    """Unify Route A/B initialization with a persistent multi-turn runtime."""

    def __init__(self, *, robot: str = "ur5e", scene: str | Path | None = None, interaction_registry: str | Path | None = None, output_root: str | Path = "var/sessions", engine: PipelineEngine | None = None, planner: str = "recipe", viewer_mode: str = "headless") -> None:
        self.session_id = uuid.uuid4().hex[:12]
        self.robot = robot
        self.output_root = Path(output_root).expanduser().resolve() / self.session_id
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.engine = engine or PipelineEngine()
        self.planner = planner
        self.viewer_mode = viewer_mode
        self.origin: str | None = None
        self.scene_version = 0
        self.world_version = 0
        self.turn_index = 0
        self.scene_path: Path | None = Path(scene).expanduser().resolve() if scene else None
        self.interaction_registry: Path | None = Path(interaction_registry).expanduser().resolve() if interaction_registry else None
        self.scene_registry: dict[str, Any] = {}
        self.semantic_map = SemanticMap()
        self.world_state: WorldState | None = None
        self.dialogue_state = DialogueState()
        self.control: SessionExecutorClient | None = None
        self.paused = False
        self.closed = False
        self.execution_history: list[dict[str, Any]] = []
        self.next_instance_index: dict[str, int] = {}
        self._write_session()

    def run_turn(self, instruction: str, *, interaction_registry: str | Path | None = None) -> dict[str, Any]:
        if not instruction.strip():
            raise ValueError("instruction cannot be empty")
        if self.closed:
            raise RuntimeError("SESSION_CLOSED")
        self.turn_index += 1
        turn_dir = self.output_root / "turns" / f"{self.turn_index:04d}"
        turn_dir.mkdir(parents=True, exist_ok=True)
        # Resolve stable dialogue referents before the single understanding
        # call; the parser sees the same semantic instruction that planning
        # will consume, while DialogueState still records the original text.
        dialogue_binding = self._dialogue_binding(instruction)
        task_instruction = self._resolve_dialogue_instruction(instruction, dialogue_binding)
        parsed_turn = self.engine.understand_turn(task_instruction)
        referent_object_id = dialogue_binding.object_ids[0] if dialogue_binding and len(dialogue_binding.object_ids) == 1 else None
        if parsed_turn.turn_kind == TurnKind.SCENE_QUERY and parsed_turn.scene_query is not None and any(token in instruction for token in ("它", "刚才那个", "这个")):
            parsed_turn.scene_query.referent = True
        if parsed_turn.turn_kind == TurnKind.SCENE_EDIT:
            if parsed_turn.status != "accepted" or parsed_turn.scene_edit is None:
                raise ValueError(parsed_turn.raw_task or "invalid scene edit")
            return self._run_scene_edit(instruction, parsed_turn.scene_edit, turn_dir)
        if parsed_turn.turn_kind == TurnKind.SCENE_QUERY:
            try:
                query = self._run_scene_query(parsed_turn.scene_query, explicit_object_id=referent_object_id) or "当前场景状态尚未初始化。"
            except SceneQueryAmbiguous as exc:
                self._record_dialogue(instruction)
                self._write_session()
                return {"status": "clarification_required", "turn_type": "scene_query", "turn": self.turn_index, "error": str(exc), "scene_version": self.scene_version, "world_version": self.world_version}
            self._record_dialogue(instruction)
            self._write_session()
            return {"status": "query_answer", "turn": self.turn_index, "turn_type": "scene_query", "answer": query, "scene_version": self.scene_version, "world_version": self.world_version}
        if parsed_turn.turn_kind == TurnKind.SESSION_CONTROL:
            if parsed_turn.session_control is None:
                raise ValueError("session_control turn requires an action")
            action = parsed_turn.session_control.action
            if action == SessionControlType.PAUSE:
                self.paused = True
                self._write_session()
                return {"status": "session_paused", "turn": self.turn_index, "scene_version": self.scene_version, "world_version": self.world_version}
            if action == SessionControlType.RESUME:
                self.paused = False
                self._write_session()
                return {"status": "session_resumed", "turn": self.turn_index, "scene_version": self.scene_version, "world_version": self.world_version}
            self.close()
            return {"status": "session_closed", "turn": self.turn_index, "scene_version": self.scene_version, "world_version": self.world_version}
        if self.paused:
            self._write_session()
            return {"status": "session_paused", "turn": self.turn_index, "scene_version": self.scene_version, "world_version": self.world_version}
        explicit_bindings = (
            ReferentBinder.bind(parsed_turn, dialogue_binding)
            if parsed_turn.status == "accepted" else {}
        )
        excluded_object_ids = set()
        if "另一个" in instruction:
            previous_object = self.dialogue_state.referents.get("它") or self.dialogue_state.referents.get("刚才那个")
            if previous_object:
                excluded_object_ids.add(previous_object)
        if self.scene_path is None:
            scene = None
            self.origin = "uploaded" if interaction_registry else "generated"
        else:
            scene = self.scene_path
            if self.origin is None:
                self.origin = "uploaded"
        registry = Path(interaction_registry).expanduser().resolve() if interaction_registry else self.interaction_registry
        live_observation = None
        if self.control is not None and self.scene_path is not None:
            live = self.control.observe(turn_dir / "live_observation")
            live_observation = SceneObservation(
                scene_id=self.scene_registry.get("scene_id", self.scene_path.stem),
                camera_id="scene_camera",
                image_width_px=640,
                image_height_px=480,
                rgb_path=Path(live["observation"]["rgb_path"]),
                segmentation_path=Path(live["observation"]["segmentation_path"]),
                segmentation_visualization_path=Path(live["observation"]["segmentation_visualization_path"]),
                instance_index_path=Path(live["observation"]["instances_path"]),
                instances=[InstanceObservation.model_validate(item) for item in live["observation"]["instances"]],
            )
        plan_kwargs = dict(
            robot=self.robot,
            interaction_registry=registry,
            output_dir=turn_dir,
            planner=self.planner,
            world_positions={object_id: state.position for object_id, state in self.world_state.objects.items()} if self.world_state else None,
            live_observation=live_observation,
            semantic_map=self.semantic_map.model_dump(mode="json"),
            explicit_object_id=None,
            explicit_bindings=explicit_bindings,
            parsed_turn=parsed_turn,
            planning_mode=ModelCallMode.UPLOADED_INITIAL if scene is not None else ModelCallMode.GENERATED_INITIAL,
            held_object_id=self.world_state.held_object if self.world_state else None,
            excluded_object_ids=excluded_object_ids,
        )
        if self.scene_version == 0:
            result: PipelineResult = self.engine.plan(instruction, scene=scene, **plan_kwargs)
        else:
            result = self.engine.plan_current_scene(
                instruction,
                scene_path=self.scene_path,
                scene_registry=SceneRegistry.model_validate(self.scene_registry),
                origin=self.origin or "generated",
                **plan_kwargs,
            )
        if result.status != "accepted":
            self._record_dialogue(instruction)
            return {"status": result.status, "error": result.error, "turn": self.turn_index}
        self.scene_path = Path(result.source_scene).resolve() if result.source_scene else self.scene_path
        self.interaction_registry = Path(result.interaction_registry).resolve() if result.interaction_registry else registry
        if self.scene_version == 0:
            self.scene_version = 1
            self.scene_registry = result.scene_registry
            self._build_semantic_map()
            self._initialize_instance_indices()
        control_response, batch_report = self._execute_concrete_operations(result, turn_dir)
        (self.output_root / "state").mkdir(exist_ok=True)
        (self.output_root / "state" / "world_state.json").write_text(self.world_state.model_dump_json(indent=2), encoding="utf-8")
        (self.output_root / "state" / "semantic_map.json").write_text(self.semantic_map.model_dump_json(indent=2), encoding="utf-8")
        self._record_dialogue(instruction, result)
        self._learn_semantics(result)
        self.execution_history.append({"turn": self.turn_index, "instruction": instruction, "report": batch_report})
        self._write_session()
        return {"status": "accepted", "turn": self.turn_index, "turn_type": "robot_task", "scene_version": self.scene_version, "world_version": self.world_version, "result": result.model_dump(mode="json"), "report": batch_report, "world_state": self.world_state.model_dump(mode="json")}

    def _execute_concrete_operations(self, result: PipelineResult, turn_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
        """Execute concrete operations serially in one persistent control session.

        Each operation gets its own compiled bundle and world-state refresh.
        This keeps the low-level controller single-instance while making the
        batch boundary explicit and fail-fast.
        """
        grounded = GroundedTask.model_validate(result.grounded_task)
        skill = SkillPlan.model_validate(result.skill_plan)
        operation_ids = []
        for step in skill.steps:
            if step.operation_id not in operation_ids:
                operation_ids.append(step.operation_id)
        if not operation_ids:
            raise ValueError("execution_plan_empty")
        operation_map = {operation.operation_id: operation for operation in grounded.operations}
        subtasks = []
        combined_steps = []
        for index, operation_id in enumerate(operation_ids, 1):
            placement = None
            bundle_path = None
            try:
                operation = operation_map.get(operation_id)
                if operation is None:
                    parent = operation_id.split("__", 1)[0]
                    operation = next((value for value in grounded.operations if value.operation_id == parent), None)
                if operation is None:
                    raise ValueError(f"execution_operation_missing: {operation_id}")
                sub_skill = slice_skill_plan_for_operation(skill, operation_id)
                if not sub_skill.steps:
                    subtasks.append({"operation_id": operation_id, "status": "skipped", "reason": "empty_skill_plan"})
                    continue
                sub_task = grounded.model_copy(update={"operations": [operation]})
                sub_dir = turn_dir / "subtasks" / f"{index:04d}"
                registry_path, placement = self._prepare_live_placement(operation, grounded, sub_skill, sub_dir)
                bundle = compile_execution_bundle(
                    sub_skill, sub_task,
                    scene_path=self.scene_path,
                    interaction_registry_path=registry_path,
                    output_dir=sub_dir,
                    route=result.route,
                    robot=self.robot,
                )
                bundle_path = Path(bundle.task_dir) / "execution_bundle.json"
                if self.control is None:
                    self.control = SessionExecutorClient(viewer_mode=self.viewer_mode)
                    self.control.open(bundle_path)
                response = self.control.execute(bundle_path)
                report = response.get("report", {})
                success = bool(report.get("success", response.get("success", True)))
                subtasks.append({"operation_id": operation_id, "status": "succeeded" if success else "failed", "report": report, "bundle": str(bundle_path), "placement": placement})
                combined_steps.extend(report.get("steps", []))
                self.world_version += 1
                self.world_state = WorldState.model_validate({**response["snapshot"], "world_version": self.world_version, "scene_version": self.scene_version, "turn_index": self.turn_index})
                if not success:
                    break
            except Exception as exc:
                # A failed compile, placement allocation, or RPC must not
                # escape the batch: the live controller may already have
                # advanced while the session files have not. Refresh first,
                # then persist a partial report and mark later operations.
                snapshot = None
                if self.control is not None:
                    try:
                        snapshot = self.control.snapshot().get("snapshot")
                    except Exception:
                        snapshot = None
                if snapshot is not None:
                    self.world_version += 1
                    self.world_state = WorldState.model_validate({**snapshot, "world_version": self.world_version, "scene_version": self.scene_version, "turn_index": self.turn_index})
                subtasks.append({"operation_id": operation_id, "status": "failed", "error": str(exc), "bundle": str(bundle_path) if bundle_path else None, "placement": placement})
                for remaining in operation_ids[index:]:
                    subtasks.append({"operation_id": remaining, "status": "skipped", "reason": "previous_subtask_failed"})
                break
        completed = sum(item["status"] == "succeeded" for item in subtasks)
        failed = sum(item["status"] == "failed" for item in subtasks)
        batch_status = "success" if failed == 0 and completed == len(operation_ids) else "partial_failure"
        batch = {
            "status": batch_status,
            # Keep the legacy execution-report shape at the chat boundary so
            # callers can consume run and chat results uniformly.
            "success": batch_status == "success",
            "total": len(operation_ids),
            "completed": completed,
            "failed": failed,
            "skipped": max(0, len(operation_ids) - completed - failed),
            "commands_total": sum(item.get("report", {}).get("commands_total", 0) for item in subtasks),
            "commands_completed": sum(item.get("report", {}).get("commands_completed", 0) for item in subtasks),
            "steps": combined_steps,
            "subtasks": subtasks,
        }
        (turn_dir / "batch_execution_report.json").write_text(json.dumps(batch, ensure_ascii=False, indent=2), encoding="utf-8")
        last = subtasks[-1].get("report", {}) if subtasks else {}
        combined = dict(last)
        combined["success"] = batch["status"] == "success"
        combined["commands_total"] = sum(item.get("report", {}).get("commands_total", 0) for item in subtasks)
        combined["commands_completed"] = sum(item.get("report", {}).get("commands_completed", 0) for item in subtasks)
        combined["steps"] = combined_steps
        # Preserve the per-turn artifact consumed by existing tooling while
        # also keeping the new batch report alongside it.
        (turn_dir / "execution_report.json").write_text(json.dumps(combined, ensure_ascii=False, indent=2), encoding="utf-8")
        return {"snapshot": self.world_state.model_dump(mode="json"), "report": combined}, batch

    def _prepare_live_placement(self, operation, grounded: GroundedTask, skill: SkillPlan, sub_dir: Path) -> tuple[Path, dict[str, Any] | None]:
        """Resolve a semantic placement against the current live world state."""
        if self.interaction_registry is None or self.world_state is None or self.scene_registry is None:
            if self.interaction_registry is None or self.scene_registry is None:
                return self.interaction_registry, None
            if self.world_state is None:
                registry_model = SceneRegistry.model_validate(self.scene_registry)
                self.world_state = WorldState(
                    world_version=0,
                    scene_version=max(self.scene_version, 1),
                    turn_index=self.turn_index,
                    sim_time=0.0,
                    objects={item.object_id: ObjectWorldState(object_id=item.object_id, body_name=item.body_name, position=item.position, quaternion=(1.0, 0.0, 0.0, 0.0)) for item in registry_model.objects},
                )
        placement_steps = [step for step in skill.steps if step.semantic_target in {"placement_region", "container_interior", "support_surface"}]
        if not placement_steps or not operation.destination:
            return self.interaction_registry, None
        destination_entity = next((item for item in grounded.entities if item.entity_id == operation.destination), None)
        source_entity = next((item for item in grounded.entities if item.entity_id == (operation.source or operation.target)), None)
        if destination_entity is None or source_entity is None:
            return self.interaction_registry, None
        destination_id = destination_entity.object_id
        source_id = source_entity.object_id
        data = json.loads(Path(self.interaction_registry).read_text(encoding="utf-8"))
        source_item = next((item for item in SceneRegistry.model_validate(self.scene_registry).objects if item.object_id == source_id), None)
        source_dimensions = source_item.dimensions_m if source_item and source_item.dimensions_m else (0.06, 0.06, 0.06)
        spec = operation.placement_target
        if spec is None:
            # Backward-compatible uploaded/old artifacts: destination
            # containers retain the historical default semantics.
            spec = PlacementTargetSpec(kind=PlacementTargetKind.CONTAINER_INTERIOR, reference=operation.destination, relation="inside")
        else:
            # PlacementTargetSpec is expressed in semantic entity IDs, while
            # the live resolver and interaction registry are keyed by concrete
            # grounded object IDs.  Keep the semantic spec in the plan/report,
            # but resolve its host through the current binding before doing
            # geometry (this matters when an uploaded registry names an object
            # ``blue_cabinet_upper_compartment`` for entity
            # ``upper_compartment_01``).
            spec = spec.model_copy(update={
                "reference": destination_id if spec.reference == operation.destination else spec.reference,
                "support": destination_id if spec.support == operation.destination else spec.support,
            })
        resolved = PlacementResolver().resolve(
            spec,
            source_object_id=source_id,
            world_state=self.world_state,
            scene_registry=SceneRegistry.model_validate(self.scene_registry),
            interaction_registry=data,
            source_dimensions=source_dimensions,
            world_version=self.world_state.world_version,
        )
        target = data.setdefault("objects", {}).get(resolved.host_object_id)
        if not isinstance(target, dict):
            raise ValueError(f"placement_reference_missing: {resolved.host_object_id}")
        spatial = target.setdefault("spatial", {})
        anchors = spatial.setdefault("anchors", {})
        # ``ResolvedPlacement.world_position`` is the final payload pose.  A
        # registry anchor, however, is expressed in the host object's local
        # frame and is the pose of the end effector, not the payload body
        # origin.  Convert world-space relative placements back into that
        # frame before writing the temporary registry override.
        host_state = self.world_state.objects.get(resolved.host_object_id)
        if resolved.local_position is not None:
            anchor_position = list(resolved.local_position)
        elif host_state is not None:
            relative_world = tuple(
                resolved.world_position[index] - host_state.position[index]
                for index in range(3)
            )
            anchor_position = list(
                quaternion_inverse_rotate(relative_world, host_state.quaternion)
            )
        else:
            anchor_position = list(resolved.world_position)
        if resolved.kind == PlacementTargetKind.RELATIVE_OBJECT:
            # Relative placement is normally on the same tabletop plane as
            # the reference object.  Move to the source centre height so the
            # carried payload is released without driving the gripper into
            # the table or the neighbouring object.
            anchor_position[2] += float(source_dimensions[2]) / 2.0
        elif resolved.kind == PlacementTargetKind.CONTAINER_INTERIOR:
            # Keep the authored interior approach height (it is deliberately
            # above the container floor) while replacing only its allocated
            # XY slot.  This preserves the proven open-box clearance for
            # rotated and legacy registries.
            authored_anchor = (spatial.get("anchors", {}) or {}).get("interior", {})
            authored_position = authored_anchor.get("local_position") if isinstance(authored_anchor, dict) else None
            if isinstance(authored_position, (list, tuple)) and len(authored_position) == 3:
                anchor_position[2] = float(authored_position[2])
        if resolved.kind in {PlacementTargetKind.SUPPORT_SURFACE, PlacementTargetKind.FREE_SPACE}:
            # The resolved world position is the object's final contact pose;
            # the gripper must stop above that pose to avoid penetrating the
            # tabletop before release.
            anchor_position[2] = max(anchor_position[2] + 0.06, 0.10)
        anchors["resolved_placement"] = {
            "target_id": f"{resolved.host_object_id}_resolved_placement",
            "aliases": ["resolved placement", "placement_region"],
            "local_position": anchor_position,
        }
        # Keep legacy planner output executable while the semantic planner
        # migrates to ``placement_region``.  Both names intentionally point
        # to the same live allocation, so an old ``container_interior`` or
        # ``support_surface`` step cannot silently use a stale authored pose.
        if resolved.kind == PlacementTargetKind.CONTAINER_INTERIOR:
            anchors["interior"] = anchors["resolved_placement"]
        elif resolved.kind in {PlacementTargetKind.SUPPORT_SURFACE, PlacementTargetKind.FREE_SPACE}:
            anchors["support_surface"] = anchors["resolved_placement"]
        spatial["default_anchor"] = "resolved_placement"
        sub_dir.mkdir(parents=True, exist_ok=True)
        override_path = sub_dir / "interaction_registry_input.json"
        override_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        placement = resolved.model_dump(mode="json")
        (sub_dir / "placement_resolution.json").write_text(json.dumps(placement, ensure_ascii=False, indent=2), encoding="utf-8")
        (sub_dir / "placement_assignment.json").write_text(json.dumps({"resolved": placement}, ensure_ascii=False, indent=2), encoding="utf-8")
        return override_path, placement

    def _objects_inside_container(self, container_id: str, bounds=None, exclude: set[str] | None = None) -> list[tuple[tuple[float, float], tuple[float, float]]]:
        if self.world_state is None:
            return []
        registry = SceneRegistry.model_validate(self.scene_registry)
        container_state = self.world_state.objects.get(container_id)
        container_item = next((item for item in registry.objects if item.object_id == container_id), None)
        if container_state is None or container_item is None:
            return []
        if bounds is None:
            return []
        local_min, local_max = bounds
        occupied = []
        excluded = exclude or set()
        for item in registry.objects:
            if item.object_id == container_id or item.object_id in excluded or item.object_id not in self.world_state.objects:
                continue
            state = self.world_state.objects[item.object_id]
            dimensions = item.dimensions_m or (0.06, 0.06, 0.06)
            x = state.position[0] - container_state.position[0]
            y = state.position[1] - container_state.position[1]
            relative_z = state.position[2] - container_state.position[2]
            # The controller reports the object's center; a released object
            # resting on the floor can therefore have its center slightly
            # below the authored interior floor plane.
            z_half = dimensions[2] / 2
            if (local_min[0] <= x <= local_max[0]
                    and local_min[1] <= y <= local_max[1]
                    and local_min[2] - z_half <= relative_z <= local_max[2] + z_half):
                occupied.append(((x - dimensions[0] / 2, x + dimensions[0] / 2), (y - dimensions[1] / 2, y + dimensions[1] / 2)))
        return occupied

    def _run_scene_edit(self, instruction: str, edit: SceneEditIntent, turn_dir: Path) -> dict[str, Any]:
        if self.origin != "generated" or self.scene_path is None or self.control is None or self.world_state is None:
            raise ValueError("scene edit currently requires an initialized generated SceneSession")
        registry = SceneRegistry.model_validate(self.scene_registry)
        if edit.count > 1:
            raise ValueError("UNSUPPORTED_MULTI_OBJECT_SCENE_EDIT: count>1 is not implemented")
        positions = {key: value.position for key, value in self.world_state.objects.items()}
        mutator = SceneMutator(self.engine.assets)
        next_scene_version = self.scene_version + 1
        scene_dir = self.output_root / "scene" / f"v{next_scene_version:04d}"
        if edit.operation == SceneEditType.ADD:
            if not edit.reference or not edit.relation:
                message = (
                    f"要增加{self._zh_label(edit.semantic_name)}，请说明它相对于哪个场景物体以及放置方向；"
                    "例如：在篮子右边增加一个香蕉。"
                )
                self._record_dialogue(instruction)
                self._write_session()
                return {
                    "status": "clarification_required",
                    "turn": self.turn_index,
                    "turn_type": "scene_edit",
                    "error": message,
                    "scene_version": self.scene_version,
                    "world_version": self.world_version,
                }
            try:
                reference_id = self._resolve_semantic_object(edit.reference or "")
            except ValueError:
                message = (
                    f"当前场景中找不到唯一的参照物“{edit.reference}”。"
                    "请先使用场景中已有的物体名称，或重新说明参照物；"
                    "例如：在盒子右边增加一个香蕉。"
                )
                self._record_dialogue(instruction)
                self._write_session()
                return {
                    "status": "clarification_required",
                    "turn": self.turn_index,
                    "turn_type": "scene_edit",
                    "error": message,
                    "scene_version": self.scene_version,
                    "world_version": self.world_version,
                }
            base = self._object_base(edit.semantic_name)
            index = self.next_instance_index.get(base, 1)
            object_id = f"{base}_{index:02d}"
            self.next_instance_index[base] = index + 1
            updated, scene, interactions = mutator.add(
                registry,
                semantic_name=edit.semantic_name,
                category=edit.category,
                object_id=object_id,
                relation=edit.relation,
                reference_object=reference_id,
                current_positions=positions,
                output_dir=scene_dir,
            )
            self.semantic_map.objects[object_id] = SemanticObject(object_id=object_id, labels=[edit.semantic_name, self._zh_label(edit.semantic_name)], category=edit.category, source="user", confidence=1.0)
            self.dialogue_state.referents.update({"它": object_id, "刚才那个": object_id, f"刚才那个{self._zh_label(edit.semantic_name)}": object_id})
            patch = {"operation": "add", "object_id": object_id, "relation": edit.relation, "reference": reference_id}
        elif edit.operation == SceneEditType.REMOVE:
            object_id = self._resolve_semantic_object(edit.semantic_name)
            if self.world_state.held_object == object_id:
                raise ValueError(f"HELD_OBJECT_REMOVE_FORBIDDEN: cannot remove held object {object_id}")
            updated, scene, interactions = mutator.remove(registry, object_id, current_positions=positions, output_dir=scene_dir)
            self.semantic_map.objects.pop(object_id, None)
            patch = {"operation": "remove", "object_id": object_id}
        else:
            raise ValueError(f"unsupported scene edit: {edit.operation}")
        self.scene_registry = updated.model_dump(mode="json")
        self.scene_path = scene.resolve()
        self.interaction_registry = interactions.resolve()
        response = self.control.reload_scene(self.scene_path, self.interaction_registry, robot=self.robot)
        self.scene_version = next_scene_version
        self.world_version += 1
        self.world_state = WorldState.model_validate({**response["snapshot"], "world_version": self.world_version, "scene_version": self.scene_version, "turn_index": self.turn_index})
        (turn_dir / "scene_patch.json").write_text(json.dumps(patch, ensure_ascii=False, indent=2), encoding="utf-8")
        state_dir = self.output_root / "state"; state_dir.mkdir(parents=True, exist_ok=True)
        (state_dir / "world_state.json").write_text(self.world_state.model_dump_json(indent=2), encoding="utf-8")
        (state_dir / "semantic_map.json").write_text(self.semantic_map.model_dump_json(indent=2), encoding="utf-8")
        self._record_dialogue(instruction)
        self._write_session()
        return {"status": "scene_updated", "turn": self.turn_index, "turn_type": "scene_edit", "patch": patch, "scene_version": self.scene_version, "world_version": self.world_version, "runtime_reloaded": True, "state_restored": True}

    def _resolve_semantic_object(self, query: str) -> str:
        normalized = query.casefold()
        # The tabletop is an implicit scene support surface, not an
        # interactable object in SceneRegistry.  It is still a valid anchor
        # for placement edits such as “在桌子左边增加一个篮子”.
        if normalized in {"table", "桌子", "桌面", "台面", "work_table"}:
            return "__table__"
        aliases = {"basket": {"basket", "篮子", "篮"}, "banana": {"banana", "香蕉"}, "apple": {"apple", "苹果"}, "baseball": {"baseball", "棒球"}}
        tokens = aliases.get(normalized, {normalized})
        matches = []
        for object_id, item in self.semantic_map.objects.items():
            values = {object_id.casefold(), *(value.casefold() for value in item.labels)}
            if any(exact_name_match(token, value) for token in tokens for value in values if token and value):
                matches.append(object_id)
        if not matches:
            for item in self.scene_registry.get("objects", []):
                values = {str(item.get("object_id", "")).casefold(), str(item.get("semantic_name", "")).casefold(), str(item.get("model_name", "")).casefold()}
                if any(exact_name_match(token, value) for token in tokens for value in values if token and value):
                    matches.append(item["object_id"])
        if len(matches) != 1:
            raise ValueError(f"scene edit reference is {'missing' if not matches else 'ambiguous'}: {query} -> {matches}")
        return matches[0]

    def _initialize_instance_indices(self) -> None:
        for item in self.scene_registry.get("objects", []):
            match = re.match(r"(.+)_([0-9]+)$", item.get("object_id", ""))
            if match:
                self.next_instance_index[match.group(1)] = max(self.next_instance_index.get(match.group(1), 1), int(match.group(2)) + 1)

    @staticmethod
    def _object_base(name: str) -> str:
        return re.sub(r"[^a-z0-9]+", "_", name.casefold()).strip("_") or "scene_object"

    @staticmethod
    def _zh_label(name: str) -> str:
        return {"banana": "香蕉", "apple": "苹果", "baseball": "棒球"}.get(name, name)

    def snapshot(self) -> WorldState | None:
        if self.control is None:
            return self.world_state
        self.world_version += 1
        raw = self.control.snapshot()["snapshot"]
        self.world_state = WorldState.model_validate({**raw, "world_version": self.world_version, "scene_version": max(self.scene_version, 1), "turn_index": self.turn_index})
        return self.world_state

    def close(self) -> None:
        if self.closed:
            return
        if self.control is not None:
            self.control.close()
            self.control = None
        self.closed = True
        self._write_session()

    def _build_semantic_map(self) -> None:
        objects = self.scene_registry.get("objects", [])
        if isinstance(objects, dict):
            iterable = objects.values()
        else:
            iterable = objects
        for item in iterable:
            object_id = item.get("object_id")
            if not object_id:
                continue
            labels = [value for value in (item.get("semantic_name"), *(item.get("aliases") or [])) if value]
            self.semantic_map.objects[object_id] = SemanticObject(object_id=object_id, labels=labels, source="authored")
        if self.interaction_registry and self.interaction_registry.is_file():
            payload = json.loads(self.interaction_registry.read_text(encoding="utf-8"))
            for object_id, item in payload.get("objects", {}).items():
                current = self.semantic_map.objects.get(object_id)
                labels = [str(value) for value in item.get("aliases", []) if value]
                if item.get("semantic_name"):
                    labels.insert(0, str(item["semantic_name"]))
                if current is None:
                    self.semantic_map.objects[object_id] = SemanticObject(object_id=object_id, labels=labels, source="authored", confidence=0.95)
                else:
                    current.labels = list(dict.fromkeys([*current.labels, *labels]))
                    current.confidence = max(current.confidence or 0.0, 0.95)

    def _record_dialogue(self, instruction: str, result: PipelineResult | None = None) -> None:
        self.dialogue_state.recent_turns = (self.dialogue_state.recent_turns + [instruction])[-20:]
        if result and result.grounded_task:
            grounded_entities = result.grounded_task.get("entities", [])
            for entity in result.grounded_task.get("entities", []):
                self.dialogue_state.last_grounded_objects[entity["entity_id"]] = entity["object_id"]
            source_id = next((op.get("source") or op.get("target") for op in result.task_intent.get("operations", []) if op.get("source") or op.get("target")), None)
            referents = [entity for entity in grounded_entities if entity.get("entity_id") == source_id or entity.get("semantic_entity_id") == source_id]
            if not referents and grounded_entities:
                referents = [grounded_entities[0]]
            object_ids = list(dict.fromkeys(entity["object_id"] for entity in referents))
            if len(object_ids) == 1:
                self.dialogue_state.referents.update({"它": object_ids[0], "这个": object_ids[0], "刚才那个": object_ids[0]})
            elif object_ids:
                self.dialogue_state.referent_sets.update({"它们": object_ids, "这些": object_ids, "刚才那些": object_ids})
        (self.output_root / "state").mkdir(parents=True, exist_ok=True)
        (self.output_root / "state" / "dialogue_state.json").write_text(self.dialogue_state.model_dump_json(indent=2), encoding="utf-8")

    def _learn_semantics(self, result: PipelineResult) -> None:
        for entity in (result.grounded_task or {}).get("entities", []):
            object_id = entity["object_id"]
            current = self.semantic_map.objects.get(object_id)
            if current is None:
                current = SemanticObject(object_id=object_id, labels=[entity["semantic_name"]], category=entity.get("category"), source="vision", identity_iou=entity.get("bbox_iou"), last_verified_world_version=self.world_version)
                self.semantic_map.objects[object_id] = current
            elif entity.get("semantic_name") and entity["semantic_name"] not in current.labels:
                current.labels.append(entity["semantic_name"])
            if entity.get("category"):
                current.category = entity["category"]
            if entity.get("color"):
                current.attributes["color"] = entity["color"]
            if entity.get("bbox_iou") is not None:
                current.identity_iou = entity["bbox_iou"]
            current.last_verified_world_version = self.world_version
            if entity.get("grounding_method") == "vlm_iou":
                current.source = "vision"
        state_dir = self.output_root / "state"
        state_dir.mkdir(parents=True, exist_ok=True)
        (state_dir / "semantic_map.json").write_text(self.semantic_map.model_dump_json(indent=2), encoding="utf-8")

    def _dialogue_binding(self, instruction: str) -> DialogueBinding | None:
        plural_token = next((token for token in ("刚才那些", "这些", "它们") if token in instruction), None)
        singular_token = next((token for token in ("刚才那个", "这个", "它") if token in instruction), None)
        if plural_token is None and singular_token is None:
            return None
        if plural_token is not None:
            object_ids = self.dialogue_state.referent_sets.get(plural_token) or self.dialogue_state.referent_sets.get("它们")
        else:
            object_id = self.dialogue_state.referents.get(singular_token or "它") or self.dialogue_state.referents.get("它")
            object_ids = [object_id] if object_id else None
        if not object_ids:
            # A plural pronoun can refer to a set introduced earlier in the
            # same turn ("两个球先右移，再把它们前移").  Let the normal
            # parser preserve that explicit noun phrase; only reject a
            # genuinely context-free follow-up such as a fresh turn starting
            # with “把它们…”.
            if plural_token and any(token in instruction[: instruction.find(plural_token)] for token in ("苹果", "香蕉", "棒球", "球", "方块", "盒子", "篮子")):
                return None
            raise ValueError("dialogue_binding_unresolved: no prior referent")
        semantic = self.semantic_map.objects.get(object_ids[0])
        if semantic is None or not semantic.labels:
            raise ValueError(f"dialogue_binding_unresolved: no semantic label for {object_ids[0]}")
        label = next(
            (value for value in semantic.labels if any("\u4e00" <= char <= "\u9fff" for char in value)),
            self._zh_label(re.sub(r"_[0-9]+$", "", object_ids[0])),
        )
        return DialogueBinding(object_ids=object_ids, semantic_label=label, plural=len(object_ids) > 1)

    def _resolve_dialogue_instruction(self, instruction: str, binding: DialogueBinding | None) -> str:
        if binding is None:
            return instruction
        resolved = instruction
        if binding.plural:
            for token in ("刚才那些", "这些", "它们"):
                resolved = resolved.replace(token, f"[dialogue_ref_set={binding.semantic_label}]")
            return resolved
        for token in ("刚才那个", "这个", "它"):
            resolved = resolved.replace(token, f"[dialogue_ref={binding.semantic_label}]")
        return resolved

    def _run_scene_query(self, query: SceneQueryIntent | None, *, explicit_object_id: str | None = None) -> str | None:
        if query is None or self.world_state is None:
            return "当前场景状态尚未初始化。"
        if query.referent and explicit_object_id:
            matches = [explicit_object_id] if explicit_object_id in self.world_state.objects else []
        else:
            semantic = query.semantic_name or ""
            matches = [
                object_id for object_id, item in self.semantic_map.objects.items()
                if object_id in self.world_state.objects
                and (
                    not semantic
                    or exact_name_match(semantic, object_id)
                    or any(exact_name_match(semantic, label) for label in item.labels)
                )
                and (
                    query.category is None
                    or item.category == query.category
                    # An exact semantic-name match is sufficient when the
                    # imported registry omitted the category metadata.
                    or (item.category is None and (
                        exact_name_match(semantic, object_id)
                        or any(exact_name_match(semantic, label) for label in item.labels)
                    ))
                )
            ]
        label = self._zh_label(query.semantic_name or "目标")
        if query.query_type == SceneQueryType.COUNT:
            return f"当前有 {len(matches)} 个{label}。"
        if query.query_type == SceneQueryType.EXISTENCE:
            return f"当前{'存在' if matches else '不存在'}{label}。"
        if query.query_type == SceneQueryType.STATE:
            if not matches:
                return f"当前未找到{label}。"
            if len(matches) > 1:
                raise SceneQueryAmbiguous(f"当前找到 {len(matches)} 个{label}，请进一步说明是哪个{label}。")
            held = self.world_state.held_object in matches
            return f"{label}当前{'正在被抓取' if held else '未被抓取'}。"
        if query.query_type == SceneQueryType.POSITION and len(matches) > 1:
            raise SceneQueryAmbiguous(f"当前找到 {len(matches)} 个{label}，请进一步说明是哪个{label}。")
        positions = [self.world_state.objects[object_id].position for object_id in matches]
        return f"当前{label}位置：{positions[0]}。" if positions else f"当前未找到{label}。"

    def _write_session(self) -> None:
        payload = {"session_id": self.session_id, "origin": self.origin, "scene_version": self.scene_version, "world_version": self.world_version, "turn_index": self.turn_index, "paused": self.paused, "closed": self.closed, "scene_path": str(self.scene_path) if self.scene_path else None, "interaction_registry": str(self.interaction_registry) if self.interaction_registry else None, "next_instance_index": self.next_instance_index, "execution_history": self.execution_history}
        (self.output_root / "session.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


__all__ = ["SceneSession"]
