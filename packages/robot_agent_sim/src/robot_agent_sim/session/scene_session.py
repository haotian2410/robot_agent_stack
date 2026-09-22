"""Persistent scene lifecycle shared by generated and uploaded scenes."""

from __future__ import annotations

import json
import re
import uuid
from pathlib import Path
from typing import Any

from ..execution.compiler import compile_directory
from ..execution.session_client import SessionExecutorClient
from ..pipeline.engine import PipelineEngine, PipelineResult
from ..models.budget import ModelCallMode
from ..scene.mutator import SceneMutator
from ..scene.registry import SceneRegistry
from ..grounding.segmentation import InstanceObservation, SceneObservation
from ..grounding.name_matching import exact_name_match
from .contracts import DialogueState, SceneEditIntent, SceneEditType, SceneQueryIntent, SceneQueryType, SemanticMap, SemanticObject, SessionControlType, TurnKind, WorldState
from .referent_binder import DialogueBinding, ReferentBinder


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
        referent_object_id = dialogue_binding.object_id if dialogue_binding else None
        if parsed_turn.turn_kind == TurnKind.SCENE_QUERY and parsed_turn.scene_query is not None and any(token in instruction for token in ("它", "刚才那个", "这个")):
            parsed_turn.scene_query.referent = True
        if parsed_turn.turn_kind == TurnKind.SCENE_EDIT:
            if parsed_turn.status != "accepted" or parsed_turn.scene_edit is None:
                raise ValueError(parsed_turn.raw_task or "invalid scene edit")
            return self._run_scene_edit(instruction, parsed_turn.scene_edit, turn_dir)
        if parsed_turn.turn_kind == TurnKind.SCENE_QUERY:
            query = self._run_scene_query(parsed_turn.scene_query, explicit_object_id=referent_object_id) or "当前场景状态尚未初始化。"
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
        bundle = compile_directory(turn_dir)
        bundle_path = Path(bundle.task_dir) / "execution_bundle.json"
        if self.control is None:
            self.control = SessionExecutorClient(viewer_mode=self.viewer_mode)
            self.control.open(bundle_path)
            control_response = self.control.execute(bundle_path)
        else:
            control_response = self.control.execute(bundle_path)
        self.world_version += 1
        self.world_state = WorldState.model_validate({**control_response["snapshot"], "world_version": self.world_version, "scene_version": self.scene_version, "turn_index": self.turn_index})
        (self.output_root / "state").mkdir(exist_ok=True)
        (self.output_root / "state" / "world_state.json").write_text(self.world_state.model_dump_json(indent=2), encoding="utf-8")
        (self.output_root / "state" / "semantic_map.json").write_text(self.semantic_map.model_dump_json(indent=2), encoding="utf-8")
        self._record_dialogue(instruction, result)
        self._learn_semantics(result)
        self.execution_history.append({"turn": self.turn_index, "instruction": instruction, "report": control_response.get("report", {})})
        self._write_session()
        return {"status": "accepted", "turn": self.turn_index, "turn_type": "robot_task", "scene_version": self.scene_version, "world_version": self.world_version, "result": result.model_dump(mode="json"), "report": control_response.get("report"), "world_state": self.world_state.model_dump(mode="json")}

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
            reference_id = self._resolve_semantic_object(edit.reference or "")
            base = self._object_base(edit.semantic_name)
            index = self.next_instance_index.get(base, 1)
            object_id = f"{base}_{index:02d}"
            self.next_instance_index[base] = index + 1
            updated, scene, interactions = mutator.add(
                registry,
                semantic_name=edit.semantic_name,
                category=edit.category,
                object_id=object_id,
                relation=edit.relation or "right_of",
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
            source_id = next((op.get("source") for op in result.task_intent.get("operations", []) if op.get("source")), None)
            referent = next((entity for entity in grounded_entities if entity["entity_id"] == source_id), grounded_entities[0] if grounded_entities else None)
            if referent:
                object_id = referent["object_id"]
                self.dialogue_state.referents.update({"它": object_id, "这个": object_id, "刚才那个": object_id})
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
        if not any(token in instruction for token in ("它", "刚才那个", "这个")):
            return None
        object_id = self.dialogue_state.referents.get("它") or self.dialogue_state.referents.get("刚才那个")
        if not object_id:
            raise ValueError("dialogue_binding_unresolved: no prior referent")
        semantic = self.semantic_map.objects.get(object_id)
        if semantic is None or not semantic.labels:
            raise ValueError(f"dialogue_binding_unresolved: no semantic label for {object_id}")
        label = next(
            (value for value in semantic.labels if any("\u4e00" <= char <= "\u9fff" for char in value)),
            self._zh_label(re.sub(r"_[0-9]+$", "", object_id)),
        )
        return DialogueBinding(object_id=object_id, semantic_label=label)

    def _resolve_dialogue_instruction(self, instruction: str, binding: DialogueBinding | None) -> str:
        if binding is None:
            return instruction
        resolved = instruction
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
                and (not semantic or exact_name_match(semantic, object_id) or any(exact_name_match(semantic, label) for label in item.labels))
                and (query.category is None or item.category in {None, query.category})
            ]
        label = self._zh_label(query.semantic_name or "目标")
        if query.query_type == SceneQueryType.COUNT:
            return f"当前有 {len(matches)} 个{label}。"
        if query.query_type == SceneQueryType.EXISTENCE:
            return f"当前{'存在' if matches else '不存在'}{label}。"
        if query.query_type == SceneQueryType.STATE:
            if not matches:
                return f"当前未找到{label}。"
            held = self.world_state.held_object in matches
            return f"{label}当前{'正在被抓取' if held else '未被抓取'}。"
        positions = [self.world_state.objects[object_id].position for object_id in matches]
        return f"当前{label}位置：{positions[0]}。" if positions else f"当前未找到{label}。"

    def _write_session(self) -> None:
        payload = {"session_id": self.session_id, "origin": self.origin, "scene_version": self.scene_version, "world_version": self.world_version, "turn_index": self.turn_index, "paused": self.paused, "closed": self.closed, "scene_path": str(self.scene_path) if self.scene_path else None, "interaction_registry": str(self.interaction_registry) if self.interaction_registry else None, "next_instance_index": self.next_instance_index, "execution_history": self.execution_history}
        (self.output_root / "session.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


__all__ = ["SceneSession"]
