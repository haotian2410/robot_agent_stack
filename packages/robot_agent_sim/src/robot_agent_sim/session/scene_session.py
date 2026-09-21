"""Persistent scene lifecycle shared by generated and uploaded scenes."""

from __future__ import annotations

import json
import re
import uuid
from pathlib import Path
from typing import Any

from robot_agent_protocol import CommandDocument, ExecutionBundle, SkillCommand, scene_sha256

from ..execution.compiler import compile_directory
from ..execution.session_client import SessionExecutorClient
from ..pipeline.engine import PipelineEngine, PipelineResult
from ..scene.mutator import SceneMutator
from ..scene.registry import SceneRegistry
from ..grounding.segmentation import InstanceObservation, SceneObservation
from .contracts import DialogueState, SceneEditIntent, SceneEditType, SemanticMap, SemanticObject, WorldState


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
        self.execution_history: list[dict[str, Any]] = []
        self.next_instance_index: dict[str, int] = {}
        self._write_session()

    def run_turn(self, instruction: str, *, interaction_registry: str | Path | None = None) -> dict[str, Any]:
        if not instruction.strip():
            raise ValueError("instruction cannot be empty")
        self.turn_index += 1
        turn_dir = self.output_root / "turns" / f"{self.turn_index:04d}"
        turn_dir.mkdir(parents=True, exist_ok=True)
        edit = self._parse_scene_edit(instruction)
        if edit is not None:
            return self._run_scene_edit(instruction, edit, turn_dir)
        query = self._run_scene_query(instruction)
        if query is not None:
            self._record_dialogue(instruction)
            self._write_session()
            return {"status": "query_answer", "turn": self.turn_index, "turn_type": "scene_query", "answer": query, "scene_version": self.scene_version, "world_version": self.world_version}
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
        task_instruction = self._resolve_dialogue_instruction(instruction)
        result: PipelineResult = self.engine.plan(
            task_instruction,
            robot=self.robot,
            scene=scene,
            interaction_registry=registry,
            output_dir=turn_dir,
            planner=self.planner,
            world_positions={object_id: state.position for object_id, state in self.world_state.objects.items()} if self.world_state else None,
            live_observation=live_observation,
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
        self.execution_history.append({"turn": self.turn_index, "instruction": instruction, "report": control_response.get("report", {})})
        self._write_session()
        return {"status": "accepted", "turn": self.turn_index, "scene_version": self.scene_version, "world_version": self.world_version, "result": result.model_dump(mode="json"), "report": control_response.get("report"), "world_state": self.world_state.model_dump(mode="json")}

    def _run_scene_edit(self, instruction: str, edit: SceneEditIntent, turn_dir: Path) -> dict[str, Any]:
        if self.origin != "generated" or self.scene_path is None or self.control is None or self.world_state is None:
            raise ValueError("scene edit currently requires an initialized generated SceneSession")
        registry = SceneRegistry.model_validate(self.scene_registry)
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
            updated, scene, interactions = mutator.remove(registry, object_id, current_positions=positions, output_dir=scene_dir)
            self.semantic_map.objects.pop(object_id, None)
            patch = {"operation": "remove", "object_id": object_id}
        else:
            raise ValueError(f"unsupported scene edit: {edit.operation}")
        self.scene_registry = updated.model_dump(mode="json")
        self.scene_path = scene.resolve()
        self.interaction_registry = interactions.resolve()
        reload_bundle = self._write_reload_bundle(turn_dir)
        response = self.control.reload(reload_bundle)
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

    def _write_reload_bundle(self, turn_dir: Path) -> Path:
        first_object = next(iter(json.loads(self.interaction_registry.read_text(encoding="utf-8"))["objects"]))
        document = CommandDocument(
            robot=self.robot,
            scene=str(self.scene_path),
            registry=str(self.interaction_registry),
            scene_fingerprint=scene_sha256(self.scene_path),
            commands=[SkillCommand(command_id="reload-locate", skill_name="locate", parameters={"target": first_object})],
        )
        commands_path = turn_dir / "reload.commands.json"
        commands_path.write_text(document.model_dump_json(indent=2), encoding="utf-8")
        bundle = ExecutionBundle(robot=self.robot, route="A", task_dir=str(turn_dir.resolve()), scene=str(self.scene_path), scene_fingerprint=document.scene_fingerprint, interaction_registry=str(self.interaction_registry), commands=str(commands_path.resolve()))
        bundle_path = turn_dir / "reload.execution_bundle.json"
        bundle_path.write_text(bundle.model_dump_json(indent=2), encoding="utf-8")
        return bundle_path

    @staticmethod
    def _parse_scene_edit(instruction: str) -> SceneEditIntent | None:
        operation = SceneEditType.ADD if re.search(r"增加|添加|加(?:一个|一只|个)", instruction) else (SceneEditType.REMOVE if re.search(r"删除|移除", instruction) else None)
        if operation is None:
            return None
        names = (("香蕉", "banana", "fruit"), ("苹果", "apple", "fruit"), ("棒球", "baseball", "ball"))
        matched = next(((en, category) for zh, en, category in names if zh in instruction or en in instruction.casefold()), None)
        if matched is None:
            raise ValueError("scene edit entity is unsupported or has no asset")
        semantic_name, category = matched
        relation = "right_of" if "右" in instruction else ("left_of" if "左" in instruction else "right_of")
        reference = "basket" if "篮" in instruction else ""
        return SceneEditIntent(operation=operation, semantic_name=semantic_name, category=category, relation=relation, reference=reference)

    def _resolve_semantic_object(self, query: str) -> str:
        normalized = query.casefold()
        aliases = {"basket": {"basket", "篮子", "篮"}, "banana": {"banana", "香蕉"}, "apple": {"apple", "苹果"}, "baseball": {"baseball", "棒球"}}
        tokens = aliases.get(normalized, {normalized})
        matches = []
        for object_id, item in self.semantic_map.objects.items():
            values = {object_id.casefold(), *(value.casefold() for value in item.labels)}
            if any(token in value or value in token for token in tokens for value in values if token and value):
                matches.append(object_id)
        if not matches:
            for item in self.scene_registry.get("objects", []):
                values = {str(item.get("object_id", "")).casefold(), str(item.get("semantic_name", "")).casefold(), str(item.get("model_name", "")).casefold()}
                if any(token in value or value in token for token in tokens for value in values if token and value):
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
        if self.control is not None:
            self.control.close()
            self.control = None
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

    def _resolve_dialogue_instruction(self, instruction: str) -> str:
        object_id = self.dialogue_state.referents.get("它") or self.dialogue_state.referents.get("刚才那个")
        if not object_id:
            return instruction
        semantic = self.semantic_map.objects.get(object_id)
        if semantic is None or not semantic.labels:
            return instruction
        label = next((value for value in semantic.labels if any("\u4e00" <= char <= "\u9fff" for char in value)), semantic.labels[0])
        resolved = instruction
        for token in ("刚才那个", "这个", "它"):
            resolved = resolved.replace(token, label)
        return resolved

    def _run_scene_query(self, instruction: str) -> str | None:
        if not re.search(r"几个|多少|数量|在哪里|位置|状态", instruction):
            return None
        normalized = instruction.casefold()
        target = next((key for key, aliases in {"apple": ("苹果", "apple"), "banana": ("香蕉", "banana"), "baseball": ("棒球", "baseball"), "basket": ("篮子", "basket")}.items() if any(alias in instruction or alias in normalized for alias in aliases)), None)
        if target is None or self.world_state is None:
            return "当前场景状态尚未初始化。"
        matches = [object_id for object_id, item in self.semantic_map.objects.items() if target in object_id.casefold() or any(target in label.casefold() for label in item.labels)]
        if "几个" in instruction or "多少" in instruction or "数量" in instruction:
            return f"当前有 {len(matches)} 个{self._zh_label(target)}。"
        positions = [self.world_state.objects[object_id].position for object_id in matches if object_id in self.world_state.objects]
        return f"当前{self._zh_label(target)}位置：{positions[0]}。" if positions else f"当前未找到{self._zh_label(target)}。"

    def _write_session(self) -> None:
        payload = {"session_id": self.session_id, "origin": self.origin, "scene_version": self.scene_version, "world_version": self.world_version, "turn_index": self.turn_index, "scene_path": str(self.scene_path) if self.scene_path else None, "interaction_registry": str(self.interaction_registry) if self.interaction_registry else None, "next_instance_index": self.next_instance_index, "execution_history": self.execution_history}
        (self.output_root / "session.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


__all__ = ["SceneSession"]
