from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError

from ..assets.registry import AssetRegistry
from ..assets.resolver import AssetResolver
from ..backends.mujoco.backend import MujocoSceneBackend
from ..contracts.grounded_task import GroundedEntity, GroundedTask
from ..contracts.task_intent import TaskStatus
from ..contracts.turn import TurnKind
from ..grounding.iou import match_detections
from ..grounding.segmentation import SceneObservation
from ..grounding.interaction_registry import ground_partial_with_interaction_registry, ground_with_interaction_registry
from ..grounding.world_relation import WorldRelationResolver
from ..execution.interaction_registry_builder import build_generated_registry
from ..models.budget import ModelCallBudget, ModelCallBudgetExceeded, ModelCallMode
from ..models.fake import FakeSkillPlanningProvider, FakeTaskUnderstandingProvider, FakeVisionGroundingProvider
from ..models.skill_planning import SkillPlanningRequest, enrich_skill_plan
from ..models.task_understanding import TaskParseLLMOutput, TaskUnderstandingRequest, enrich_task
from ..models.vision_grounding import VisionGroundingRequest, VisionQuery
from ..planning.context_builder import build_planner_context
from ..planning.recipe_planner import RecipePlanner
from ..planning.semantic_validator import validate_semantic_plan
from ..scene.composer import SceneComposer
from ..skills.registry import REGISTRY


class PipelineResult(BaseModel):
    task_intent: dict
    scene_registry: dict = Field(default_factory=dict)
    grounded_task: dict | None = None
    visual_grounding: dict | None = None
    skill_plan: dict | None = None
    model_call_count: int = 0
    model_usage: dict = Field(default_factory=dict)
    planner: str = "recipe"
    route: str = "A"
    status: str = "accepted"
    artifacts: dict[str, str] = Field(default_factory=dict)
    error: str | None = None
    source_scene: str | None = None
    interaction_registry: str | None = None


class PipelineEngine:
    def __init__(self, understanding=None, vision=None, planner=None, asset_registry=None):
        self.understanding = understanding or FakeTaskUnderstandingProvider()
        self.vision = vision or FakeVisionGroundingProvider()
        self.planner = planner or FakeSkillPlanningProvider()
        self.assets = asset_registry or AssetRegistry()
        self.backend = MujocoSceneBackend()

    def understand_turn(self, instruction: str) -> TaskParseLLMOutput:
        """Run the one Task Understanding call shared by turn routing and planning."""
        return self.understanding.understand(TaskUnderstandingRequest(instruction=instruction))

    def plan_current_scene(self, instruction: str, *, scene_path: Path, scene_registry, origin: str, **kwargs) -> PipelineResult:
        """Plan against an already initialized session scene.

        ``origin`` remains ``generated`` or ``uploaded`` metadata; it is not
        recomputed from the presence of ``scene_path`` and no new scene is
        composed or uploaded during later turns.
        """
        kwargs.pop("planning_mode", None)
        return self.plan(
            instruction,
            robot=kwargs.pop("robot", "ur5e"),
            scene=Path(scene_path),
            current_registry=scene_registry,
            session_origin=origin,
            planning_mode=ModelCallMode.CURRENT_SCENE,
            **kwargs,
        )

    def plan(self, instruction: str, robot: str = "panda", scene: Path | None = None, seed: int = 0, output_dir: Path | str | None = None, planner: str = "recipe", interaction_registry: Path | None = None, world_positions: dict[str, tuple[float, float, float] | list[float]] | None = None, live_observation: SceneObservation | None = None, semantic_map: dict[str, Any] | None = None, current_registry=None, session_origin: str | None = None, explicit_bindings: dict[str, str] | None = None, explicit_object_id: str | None = None, parsed_turn: TaskParseLLMOutput | None = None, planning_mode: ModelCallMode | None = None) -> PipelineResult:
        out = Path(output_dir or "var"); out.mkdir(parents=True, exist_ok=True)
        intent = None; registry = None; observation = None; visual_grounding = None
        planner_artifacts: dict[str, str] = {}
        if planner not in {"recipe", "qwen", "auto"}:
            raise ValueError(f"unsupported planner: {planner}")
        if planning_mode is None:
            planning_mode = ModelCallMode.CURRENT_SCENE if current_registry is not None else (ModelCallMode.UPLOADED_INITIAL if scene is not None else ModelCallMode.GENERATED_INITIAL)
        budget = ModelCallBudget.for_mode(planning_mode, planner)
        if session_origin is not None:
            route = "A" if session_origin.casefold() == "generated" else "B"
        else:
            route = "B" if scene is not None else "A"
        planner_used = planner
        try:
            budget.consume("task_understanding")
            parsed = parsed_turn or self.understand_turn(instruction)
            self._capture(budget, "task_understanding", self.understanding)
            if parsed.turn_kind != TurnKind.ROBOT_TASK:
                raise ValueError(f"turn kind {parsed.turn_kind.value} must be handled by SceneSession")
            intent = enrich_task(parsed, instruction)
            if intent.status != TaskStatus.ACCEPTED:
                return self._write_result(PipelineResult(task_intent=intent.model_dump(mode="json"), status=intent.status.value, model_call_count=budget.calls, model_usage=budget.summary(), planner=planner_used, route=route), out)
            explicit_entity_id = next((op.source or op.target for op in intent.operations if op.source or op.target), None) if explicit_object_id else None
            ground_entities = [entity for entity in intent.entities if entity.entity_id != explicit_entity_id] if explicit_entity_id else intent.entities

            if scene is None and current_registry is None:
                assets = AssetResolver(self.assets).resolve_entities(intent.entities)
                registry = SceneComposer().compose(intent, assets, robot, seed)
                xml_path = self.backend.compose(registry, assets, out)
                generated_interactions = build_generated_registry(
                    xml_path, registry, out / "interaction_registry.json"
                )
                observation = self.backend.renderer.render(xml_path, registry, out)
                grounded = []
                for entity in intent.entities:
                    object_id = registry.bindings.get(entity.entity_id)
                    if object_id is None: raise ValueError(f"no selected object for entity {entity.entity_id}")
                    item = registry.by_object_id(object_id)
                    instance = next(instance for instance in observation.instances if instance.object_id == object_id)
                    grounded.append(GroundedEntity(entity_id=entity.entity_id, semantic_name=entity.semantic_name, object_id=object_id, body_name=item.body_name, model_id=item.model_id, model_name=item.model_name, grounding_method="asset_scene_binding", instance_bbox=instance.bbox))
                asset_bindings = {entity.entity_id: {"model_id": assets[entity.entity_id].model_id, "model_name": assets[entity.entity_id].model_name} for entity in intent.entities}
            else:
                registry = current_registry or self.backend.load_uploaded(Path(scene), robot)
                observation = live_observation or self.backend.renderer.render(Path(scene), registry, out)
                if interaction_registry is not None:
                    authored_grounded, missing_entities = ground_partial_with_interaction_registry(
                        ground_entities,
                        interaction_registry,
                        scene,
                        intent=intent,
                        positions=world_positions or {item.object_id: item.world_position for item in observation.instances},
                    )
                    visual_grounding = {
                        "method": "interaction_registry",
                        "registry": str(Path(interaction_registry).resolve()),
                    }
                    asset_bindings = {}
                    if missing_entities:
                        cached_missing = _semantic_cache_candidates(missing_entities, semantic_map, registry)
                        instance_by_id = {item.object_id: item for item in observation.instances}
                        still_missing = []
                        for entity in missing_entities:
                            values = cached_missing.get(entity.entity_id, [])
                            if len(values) == 1 and values[0]["object_id"] in instance_by_id:
                                object_id = values[0]["object_id"]
                                item = next(item for item in registry.objects if item.object_id == object_id)
                                authored_grounded.append(GroundedEntity(entity_id=entity.entity_id, semantic_name=entity.semantic_name, object_id=object_id, body_name=item.body_name, grounding_method="semantic_cache", instance_bbox=instance_by_id[object_id].bbox))
                            else:
                                still_missing.append(entity)
                        geometry_missing = _geometry_candidates(still_missing, registry)
                        still_missing_after_geometry = []
                        for entity in still_missing:
                            values = geometry_missing.get(entity.entity_id, [])
                            if len(values) == 1 and values[0]["object_id"] in instance_by_id:
                                object_id = values[0]["object_id"]
                                item = next(item for item in registry.objects if item.object_id == object_id)
                                authored_grounded.append(GroundedEntity(entity_id=entity.entity_id, semantic_name=entity.semantic_name, object_id=object_id, body_name=item.body_name, grounding_method="detector_iou", instance_bbox=instance_by_id[object_id].bbox))
                            else:
                                still_missing_after_geometry.append(entity)
                        missing_entities = still_missing_after_geometry
                    if missing_entities:
                        # Keep authored interaction semantics for known objects;
                        # only unresolved entities consume the vision budget.
                        queries = [VisionQuery(id=entity.entity_id, name=entity.semantic_name, category=entity.category, color=entity.color, all=True) for entity in missing_entities]
                        budget.consume("vision_grounding")
                        detection = self.vision.detect(VisionGroundingRequest(entities=queries, rgb_path=str(observation.rgb_path)))
                        self._capture(budget, "vision_grounding", self.vision)
                        truth = [{"object_id": item.object_id, "bbox": item.bbox} for item in observation.instances if item.bbox is not None]
                        relevant = [item for item in detection.detections if item.entity_id in {entity.entity_id for entity in missing_entities}]
                        for index, item in enumerate(relevant, 1):
                            if item.detection_id is None:
                                item.detection_id = f"d-{index}"
                        matches, unmatched, ambiguous = match_detections(relevant, truth)
                        visual_grounding.update({
                            "fallback_entities": [entity.entity_id for entity in missing_entities],
                            "detections": [item.model_dump(mode="json") for item in detection.detections],
                            "truth": truth,
                            "matches": [{"detection_id": detection_id, "object_id": object_id, "iou": score} for detection_id, object_id, score in matches],
                            "unmatched": unmatched,
                            "ambiguous": ambiguous,
                        })
                        if unmatched or ambiguous:
                            result = PipelineResult(task_intent=intent.model_dump(mode="json"), scene_registry=registry.model_dump(mode="json"), visual_grounding=visual_grounding, status="grounding_ambiguous" if ambiguous else "grounding_failed", model_call_count=budget.calls, model_usage=budget.summary(), planner=planner_used, route=route, error=f"unmatched={unmatched}; ambiguous={ambiguous}", source_scene=str(Path(scene).resolve()))
                            self._add_observation_artifacts(result.artifacts, observation)
                            return self._write_result(result, out)
                        instance_by_id = {item.object_id: item for item in observation.instances}
                        for detection_id, object_id, score in matches:
                            candidate = next(item for item in relevant if item.detection_id == detection_id)
                            instance = instance_by_id[object_id]
                            authored_grounded.append(GroundedEntity(entity_id=candidate.entity_id, semantic_name=next(entity.semantic_name for entity in missing_entities if entity.entity_id == candidate.entity_id), object_id=object_id, body_name=instance.body_name, grounding_method="vlm_iou", detection_bbox=candidate.bbox, instance_bbox=instance.bbox, bbox_iou=score))
                    grounded = authored_grounded
                else:
                    cached_candidates = _semantic_cache_candidates(ground_entities, semantic_map, registry)
                    geometry_candidates = _geometry_candidates(ground_entities, registry)
                    for entity_id, values in geometry_candidates.items():
                        if not cached_candidates.get(entity_id):
                            cached_candidates[entity_id] = values
                    current_positions = world_positions or {item.object_id: item.world_position for item in observation.instances}
                    cached_entities = {entity_id for entity_id, values in cached_candidates.items() if values}
                    semantic_only_candidates = _semantic_cache_candidates(ground_entities, semantic_map, registry)
                    geometry_entities = {
                        entity_id for entity_id, values in geometry_candidates.items()
                        if values and not semantic_only_candidates.get(entity_id)
                    }
                    unresolved_entities = [entity for entity in ground_entities if entity.entity_id not in cached_entities]
                    if cached_candidates and not unresolved_entities:
                        selected = WorldRelationResolver().resolve(intent, cached_candidates, current_positions)
                        instance_by_id = {item.object_id: item for item in observation.instances}
                        grounded = [GroundedEntity(entity_id=entity.entity_id, semantic_name=entity.semantic_name, object_id=selected[entity.entity_id]["object_id"], body_name=instance_by_id[selected[entity.entity_id]["object_id"]].body_name, grounding_method="semantic_cache", instance_bbox=instance_by_id[selected[entity.entity_id]["object_id"]].bbox) for entity in ground_entities]
                        visual_grounding = {"method": "semantic_cache", "vision_used": False}
                        asset_bindings = {}
                    else:
                        selection_entities = {
                        value for relation in intent.spatial_relations if relation.scope == "selection"
                        for value in (relation.subject, relation.reference) if value
                        }
                        queries = [VisionQuery(id=entity.entity_id, name=entity.semantic_name, category=entity.category, color=entity.color, all=entity.entity_id in selection_entities) for entity in unresolved_entities]
                        budget.consume("vision_grounding")
                        detection = self.vision.detect(VisionGroundingRequest(entities=queries, rgb_path=str(observation.rgb_path)))
                        self._capture(budget, "vision_grounding", self.vision)
                        truth = [{"object_id": item.object_id, "bbox": item.bbox} for item in observation.instances if item.bbox is not None]
                        expected = {entity.entity_id for entity in unresolved_entities}
                        relevant = [item for item in detection.detections if item.entity_id in expected]
                        for index, item in enumerate(relevant, 1):
                            if item.detection_id is None:
                                item.detection_id = f"d-{index}"
                        matches, unmatched, ambiguous = match_detections(relevant, truth)
                        visual_grounding = {"detections": [item.model_dump(mode="json") for item in detection.detections], "truth": truth, "matches": [{"detection_id": detection_id, "object_id": object_id, "iou": score} for detection_id, object_id, score in matches], "unmatched": unmatched, "ambiguous": ambiguous, "minimum_iou": 0.2, "ambiguity_margin": 0.05}
                        if unmatched or ambiguous:
                            result = PipelineResult(task_intent=intent.model_dump(mode="json"), scene_registry=registry.model_dump(mode="json"), visual_grounding=visual_grounding, status="grounding_ambiguous" if ambiguous else "grounding_failed", model_call_count=budget.calls, model_usage=budget.summary(), planner=planner_used, route=route, error=f"unmatched={unmatched}; ambiguous={ambiguous}", source_scene=str(Path(scene).resolve()))
                            self._add_observation_artifacts(result.artifacts, observation); return self._write_result(result, out)
                        instance_by_id = {item.object_id: item for item in observation.instances}
                        positions = {item.object_id: item.world_position for item in observation.instances}
                        candidate_map = {entity.entity_id: list(cached_candidates.get(entity.entity_id, [])) for entity in ground_entities}
                        for detection_id, object_id, score in matches:
                            candidate = next(item for item in relevant if item.detection_id == detection_id)
                            candidate_map[candidate.entity_id].append({"object_id": object_id, "detection_bbox": tuple(candidate.bbox), "instance_bbox": instance_by_id[object_id].bbox, "bbox_iou": score})
                        selected = WorldRelationResolver().resolve(intent, candidate_map, positions)
                        grounded = []
                        for entity in ground_entities:
                            choice = selected[entity.entity_id]; instance = instance_by_id[choice["object_id"]]
                            method = "semantic_cache" if entity.entity_id in cached_entities and entity.entity_id not in geometry_entities else ("detector_iou" if entity.entity_id in geometry_entities else "vlm_iou")
                            grounded.append(GroundedEntity(entity_id=entity.entity_id, semantic_name=entity.semantic_name, object_id=choice["object_id"], body_name=instance.body_name, grounding_method=method, detection_bbox=choice.get("detection_bbox"), instance_bbox=instance.bbox, bbox_iou=choice.get("bbox_iou")))
                        asset_bindings = {}

            if explicit_object_id:
                explicit_entity_id = next((op.source or op.target for op in intent.operations if op.source or op.target), None)
                explicit_item = next((item for item in registry.objects if item.object_id == explicit_object_id), None)
                explicit_instance = next((item for item in observation.instances if item.object_id == explicit_object_id), None)
                if explicit_entity_id and explicit_item is not None and explicit_instance is not None:
                    grounded = [entity for entity in grounded if entity.entity_id != explicit_entity_id]
                    grounded.insert(0, GroundedEntity(entity_id=explicit_entity_id, semantic_name=next(entity.semantic_name for entity in intent.entities if entity.entity_id == explicit_entity_id), object_id=explicit_object_id, body_name=explicit_item.body_name, model_id=explicit_item.model_id, model_name=explicit_item.model_name, grounding_method="dialogue_binding", instance_bbox=explicit_instance.bbox))
            task = GroundedTask(instruction=intent.instruction, task_types=intent.task_types, entities=grounded, operations=intent.operations, spatial_relations=intent.spatial_relations, scene_id=registry.scene_id)
            recipe_supported = RecipePlanner.supports(task)
            if planner == "recipe" or (planner == "auto" and recipe_supported):
                if not recipe_supported:
                    raise ValueError("unsupported_recipe")
                skill = RecipePlanner().plan(task)
                planner_used = "recipe"
            else:
                planner_used = "qwen"
                context = build_planner_context(
                    task,
                    generated_interactions if scene is None else interaction_registry,
                )
                catalog = REGISTRY.prompt_catalog()
                context_path = out / "planner_context.json"
                catalog_path = out / "planner_skill_catalog.txt"
                context_path.write_text(
                    json.dumps(context.model_dump(mode="json"), ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                catalog_path.write_text(catalog + "\n", encoding="utf-8")
                planner_artifacts.update({
                    "planner_context.json": str(context_path),
                    "planner_skill_catalog.txt": str(catalog_path),
                })
                budget.consume("skill_planning")
                raw_plan = self.planner.plan(SkillPlanningRequest(
                    context=context,
                    skill_catalog=catalog,
                ))
                self._capture(budget, "skill_planning", self.planner)
                raw_path = out / "raw_skill_plan.json"
                raw_path.write_text(
                    json.dumps(raw_plan.model_dump(mode="json"), ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                planner_artifacts["raw_skill_plan.json"] = str(raw_path)
                skill = enrich_skill_plan(raw_plan, task)
                validate_semantic_plan(skill, task, context, REGISTRY)
            result = PipelineResult(task_intent=intent.model_dump(mode="json"), scene_registry=registry.model_dump(mode="json"), grounded_task=task.model_dump(mode="json"), visual_grounding=visual_grounding, skill_plan=skill.model_dump(mode="json"), model_call_count=budget.calls, model_usage=budget.summary(), planner=planner_used, route=route, source_scene=str((xml_path if scene is None else Path(scene)).resolve()), interaction_registry=str(generated_interactions) if scene is None else (str(Path(interaction_registry).resolve()) if interaction_registry else None))
            result.artifacts["asset_bindings.json"] = str(out / "asset_bindings.json")
            result.artifacts.update(planner_artifacts)
            Path(result.artifacts["asset_bindings.json"]).write_text(json.dumps(asset_bindings, ensure_ascii=False, indent=2), encoding="utf-8")
            self._add_observation_artifacts(result.artifacts, observation)
            if scene is None:
                result.artifacts["scene.xml"] = str(xml_path)
                result.artifacts["interaction_registry.json"] = str(generated_interactions)
            return self._write_result(result, out)
        except (OSError, ValueError, KeyError, RuntimeError, ValidationError, ModelCallBudgetExceeded) as exc:
            if planner_used == "qwen":
                # The HTTP call can succeed while schema or semantic
                # validation fails. Capture usage and the final JSON response
                # before constructing the failure result.
                self._capture(budget, "skill_planning", self.planner)
                raw_value = getattr(self.planner, "last_raw_values", {}).get("skill_planning")
                if raw_value is not None:
                    raw_path = out / "raw_skill_plan.json"
                    raw_path.write_text(json.dumps(raw_value, ensure_ascii=False, indent=2), encoding="utf-8")
                    planner_artifacts["raw_skill_plan.json"] = str(raw_path)
            message = str(exc)
            status = (
                "model_call_budget_exceeded" if isinstance(exc, ModelCallBudgetExceeded)
                else "clarification_required" if "semantic_conflict:" in message
                else "task_semantic_invalid" if "task_semantic_invalid:" in message
                else "asset_missing" if "asset_missing" in message
                else "unsupported_recipe" if "unsupported_recipe" in message
                else "grounding_ambiguous" if "grounding_ambiguous" in message
                else "relation_not_satisfied" if "relation_not_satisfied" in message
                else "planning_failed"
            )
            failure_scene = None
            if scene is not None:
                failure_scene = str(Path(scene).resolve())
            elif "xml_path" in locals():
                failure_scene = str(Path(xml_path).resolve())
            failure_registry = None
            if scene is not None and interaction_registry is not None:
                failure_registry = str(Path(interaction_registry).resolve())
            elif scene is None and "generated_interactions" in locals():
                failure_registry = str(Path(generated_interactions).resolve())
            result = PipelineResult(task_intent=intent.model_dump(mode="json") if intent else {"instruction": instruction}, scene_registry=registry.model_dump(mode="json") if registry else {}, status=status, model_call_count=budget.calls, model_usage=budget.summary(), planner=planner_used, route=route, error=str(exc), source_scene=failure_scene, interaction_registry=failure_registry)
            result.artifacts.update(planner_artifacts)
            if observation is not None: self._add_observation_artifacts(result.artifacts, observation)
            return self._write_result(result, out)

    @staticmethod
    def _capture(budget, stage, provider):
        calls = getattr(provider, "calls", [])
        budget.update(stage, calls[-1] if calls and calls[-1].get("stage") == stage else None)

    @staticmethod
    def _write_result(result, out):
        result.model_usage.setdefault("route", result.route)
        result.model_usage.setdefault("planner", result.planner)
        provenance = {
            "task_understanding": "qwen" if getattr(result, "planner", "recipe") == "qwen" and result.model_usage.get("stages") else "fake",
            "grounding": "asset_scene_binding" if result.route == "A" else ("interaction_registry" if result.interaction_registry else "visual_grounding"),
            "skill_planner": result.planner,
            "validator": "semantic" if result.planner == "qwen" else "recipe",
            "recipe_used": result.planner == "recipe",
        }
        payloads = {"task_intent.json": result.task_intent, "scene_registry.json": result.scene_registry, "grounded_task.json": result.grounded_task, "visual_grounding.json": result.visual_grounding, "skill_plan.json": result.skill_plan, "model_usage.json": result.model_usage, "summary.json": {"status": result.status, "route": result.route, "planner": result.planner, "model_call_count": result.model_call_count, "model_usage": result.model_usage, "planning_provenance": provenance, "error": result.error, "source_scene": result.source_scene, "interaction_registry": result.interaction_registry}}
        for name, payload in payloads.items():
            path = out / name
            if payload is None:
                if path.exists(): path.unlink()
            else:
                path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"); result.artifacts[name] = str(path)
        return result

    @staticmethod
    def _add_observation_artifacts(artifacts, observation):
        for name, path in {"rgb.png": observation.rgb_path, "segmentation.npy": observation.segmentation_path, "segmentation.png": observation.segmentation_visualization_path, "instances.json": observation.instance_index_path}.items(): artifacts[name] = str(path)


def _semantic_cache_candidates(entities, semantic_map: dict[str, Any] | None, registry) -> dict[str, list[dict[str, str]]]:
    if not semantic_map:
        return {}
    objects = semantic_map.get("objects", semantic_map)
    available = {item.object_id for item in registry.objects}
    result: dict[str, list[dict[str, str]]] = {}
    for entity in entities:
        query = {str(entity.semantic_name).casefold(), *(str(alias).casefold() for alias in entity.aliases)}
        matches = []
        for object_id, value in objects.items():
            if object_id not in available:
                continue
            labels = {str(object_id).casefold(), *(str(label).casefold() for label in value.get("labels", []))}
            if any(token and (token in label or label in token) for token in query for label in labels):
                matches.append({"object_id": object_id})
        result[entity.entity_id] = matches
    return result


def _geometry_candidates(entities, registry) -> dict[str, list[dict[str, str]]]:
    """Use stable authored body/object names before paying for vision."""
    result: dict[str, list[dict[str, str]]] = {}
    for entity in entities:
        queries = {str(entity.semantic_name).casefold(), *(str(alias).casefold() for alias in entity.aliases)}
        values = []
        for item in registry.objects:
            if item.object_id.startswith("scene_object_") and str(item.semantic_name).casefold() == item.body_name.casefold():
                continue
            names = {item.object_id.casefold(), item.body_name.casefold(), str(item.semantic_name).casefold()}
            if any(query and (query in name or name in query) for query in queries for name in names):
                values.append({"object_id": item.object_id})
        result[entity.entity_id] = values
    return result
