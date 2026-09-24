from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from ..assets.registry import AssetRegistry
from ..assets.resolver import AssetResolver
from ..backends.mujoco.backend import MujocoSceneBackend
from ..contracts.grounded_task import GroundedEntity, GroundedTask
from ..contracts.goal import GoalCondition
from ..contracts.task_intent import TaskStatus
from ..contracts.turn import TurnKind
from ..grounding.iou import match_detections
from ..grounding.name_matching import exact_name_match
from ..grounding.segmentation import SceneObservation
from ..grounding.interaction_registry import collect_interaction_candidates, source_body_name
from ..grounding.candidates import GroundingCandidate, merge_candidate
from ..grounding.world_relation import WorldRelationResolver
from ..execution.interaction_registry_builder import build_generated_registry
from ..models.budget import ModelCallBudget, ModelCallBudgetExceeded, ModelCallMode
from ..models.fake import FakeSkillPlanningProvider, FakeTaskUnderstandingProvider, FakeVisionGroundingProvider
from ..models.skill_planning import SkillPlanningRequest, enrich_skill_plan
from ..models.task_understanding import TaskParseLLMOutput, TaskUnderstandingRequest, enrich_task
from ..models.motion_policy import MotionPolicy
from ..models.vision_grounding import VisionGroundingRequest, VisionQuery
from ..planning.context_builder import PlannerInitialState, build_planner_context
from ..planning.recipe_planner import RecipePlanner
from ..planning.semantic_validator import validate_semantic_plan
from ..scene.composer import SceneComposer
from ..scene.constraints import SceneConstraintError, validate_generated_scene
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


class GroundingResolutionError(ValueError):
    def __init__(self, message: str, visual_grounding: dict):
        super().__init__(message)
        self.visual_grounding = visual_grounding


class PipelineEngine:
    def __init__(self, understanding=None, vision=None, planner=None, asset_registry=None, motion_policy: MotionPolicy | None = None):
        self.understanding = understanding or FakeTaskUnderstandingProvider()
        self.vision = vision or FakeVisionGroundingProvider()
        self.planner = planner or FakeSkillPlanningProvider()
        self.assets = asset_registry or AssetRegistry()
        self.backend = MujocoSceneBackend()
        self.motion_policy = motion_policy or MotionPolicy()

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

    def plan(self, instruction: str, robot: str = "panda", scene: Path | None = None, seed: int = 0, output_dir: Path | str | None = None, planner: str = "recipe", interaction_registry: Path | None = None, world_positions: dict[str, tuple[float, float, float] | list[float]] | None = None, live_observation: SceneObservation | None = None, semantic_map: dict[str, Any] | None = None, current_registry=None, session_origin: str | None = None, explicit_bindings: dict[str, str] | None = None, explicit_object_id: str | None = None, parsed_turn: TaskParseLLMOutput | None = None, planning_mode: ModelCallMode | None = None, held_object_id: str | None = None, excluded_object_ids: set[str] | None = None) -> PipelineResult:
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
            raw_task_path = out / "raw_task_understanding.json"
            raw_task = getattr(self.understanding, "last_raw_values", {}).get("task_understanding", parsed.model_dump(mode="json"))
            raw_task_path.write_text(json.dumps(raw_task, ensure_ascii=False, indent=2), encoding="utf-8")
            raw_task_text = getattr(self.understanding, "last_raw_text", {}).get("task_understanding")
            if raw_task_text is not None:
                (out / "raw_task_understanding.txt").write_text(raw_task_text, encoding="utf-8")
            normalized_task_path = out / "normalized_task_parse.json"
            normalized_task_path.write_text(json.dumps(parsed.model_dump(mode="json"), ensure_ascii=False, indent=2), encoding="utf-8")
            if parsed.turn_kind != TurnKind.ROBOT_TASK:
                raise ValueError(f"turn kind {parsed.turn_kind.value} must be handled by SceneSession")
            intent = enrich_task(parsed, instruction, self.motion_policy)
            if intent.status != TaskStatus.ACCEPTED:
                return self._write_result(PipelineResult(task_intent=intent.model_dump(mode="json"), status=intent.status.value, model_call_count=budget.calls, model_usage=budget.summary(), planner=planner_used, route=route), out)
            bindings = dict(explicit_bindings or {})
            if explicit_object_id:
                source_entity_id = next((op.source or op.target for op in intent.operations if op.source or op.target), None)
                if source_entity_id:
                    bindings.setdefault(source_entity_id, explicit_object_id)
            explicit_entity_ids = set(bindings)
            ground_entities = [entity for entity in intent.entities if entity.entity_id not in explicit_entity_ids]

            if scene is None and current_registry is None:
                assets = AssetResolver(self.assets).resolve_entities(intent.entities)
                registry = SceneComposer().compose(intent, assets, robot, seed)
                validate_generated_scene(intent, registry)
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
                    grounded.append(GroundedEntity(entity_id=entity.entity_id, semantic_name=entity.semantic_name, object_id=object_id, body_name=item.body_name, model_id=item.model_id, model_name=item.model_name, category=entity.category, color=entity.color, aliases=entity.aliases, quantity_mode=entity.quantity_mode, grounding_method="asset_scene_binding", instance_bbox=instance.bbox))
                asset_bindings = {entity.entity_id: {"model_id": assets[entity.entity_id].model_id, "model_name": assets[entity.entity_id].model_name} for entity in intent.entities}
            else:
                registry = current_registry or self.backend.load_uploaded(Path(scene), robot)
                observation = live_observation or self.backend.renderer.render(Path(scene), registry, out)
                candidate_map = {entity.entity_id: [] for entity in intent.entities}
                for entity_id, object_id in bindings.items():
                    merge_candidate(candidate_map[entity_id], GroundingCandidate(object_id=object_id, sources={"dialogue"}))
                if interaction_registry is not None:
                    authored_candidates, _, authored_model, _ = collect_interaction_candidates(
                        ground_entities, interaction_registry, scene, excluded_object_ids=excluded_object_ids
                    )
                    for entity_id, values in authored_candidates.items():
                        for object_id, metadata in values:
                            spatial = metadata.get("spatial", {}) if isinstance(metadata, dict) else {}
                            reference_position = (spatial.get("reference_pose") or {}).get("position")
                            merge_candidate(candidate_map[entity_id], GroundingCandidate(
                                object_id=object_id,
                                sources={"interaction_registry"},
                                body_name=source_body_name(authored_model, metadata),
                                world_position=tuple(reference_position) if reference_position else None,
                            ))
                for provider_name, provider_values in (
                    ("semantic_map", _semantic_cache_candidates(ground_entities, semantic_map, registry, excluded_object_ids)),
                    ("geometry", _geometry_candidates(ground_entities, registry, excluded_object_ids)),
                ):
                    for entity_id, values in provider_values.items():
                        for value in values:
                            merge_candidate(candidate_map[entity_id], GroundingCandidate(object_id=value["object_id"], sources={provider_name}))
                current_positions = world_positions or {item.object_id: item.world_position for item in observation.instances}
                grounded, visual_grounding = self._resolve_candidate_map(
                    intent, intent.entities, candidate_map, registry, observation, current_positions, budget, excluded_object_ids
                )
                asset_bindings = {}

            if bindings:
                for explicit_entity_id, explicit_object_id in bindings.items():
                    explicit_item = next((item for item in registry.objects if item.object_id == explicit_object_id), None)
                    explicit_instance = next((item for item in observation.instances if item.object_id == explicit_object_id), None)
                    if explicit_item is None or explicit_instance is None:
                        raise ValueError(f"dialogue binding object is missing: {explicit_object_id}")
                    grounded = [entity for entity in grounded if entity.entity_id != explicit_entity_id]
                    source_entity = next(entity for entity in intent.entities if entity.entity_id == explicit_entity_id)
                    grounded.insert(0, GroundedEntity(entity_id=explicit_entity_id, semantic_name=source_entity.semantic_name, object_id=explicit_object_id, body_name=explicit_item.body_name, model_id=explicit_item.model_id, model_name=explicit_item.model_name, category=source_entity.category, color=source_entity.color, aliases=source_entity.aliases, quantity_mode=source_entity.quantity_mode, grounding_method="dialogue_binding", instance_bbox=explicit_instance.bbox))
            task = GroundedTask(instruction=intent.instruction, task_types=intent.task_types, entities=grounded, operations=intent.operations, spatial_relations=intent.spatial_relations, scene_id=registry.scene_id)
            recipe_supported = RecipePlanner.supports(task)
            if planner == "recipe" or (planner == "auto" and recipe_supported):
                if not recipe_supported:
                    raise ValueError("unsupported_recipe")
                held_entity = next((entity.entity_id for entity in task.entities if entity.object_id == held_object_id), None)
                skill = RecipePlanner().plan(task, initial_held_entity=held_entity)
                planner_used = "recipe"
            else:
                planner_used = "qwen"
                held_entity = next((entity.entity_id for entity in task.entities if entity.object_id == held_object_id), None)
                context = build_planner_context(
                    task,
                    generated_interactions if scene is None else interaction_registry,
                    initial_state=PlannerInitialState(held_entity=held_entity),
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
                validation_path = out / "semantic_plan_validation.json"
                validation_path.write_text(json.dumps({"status": "accepted"}, ensure_ascii=False, indent=2), encoding="utf-8")
            result = PipelineResult(task_intent=intent.model_dump(mode="json"), scene_registry=registry.model_dump(mode="json"), grounded_task=task.model_dump(mode="json"), visual_grounding=visual_grounding, skill_plan=skill.model_dump(mode="json"), model_call_count=budget.calls, model_usage=budget.summary(), planner=planner_used, route=route, source_scene=str((xml_path if scene is None else Path(scene)).resolve()), interaction_registry=str(generated_interactions) if scene is None else (str(Path(interaction_registry).resolve()) if interaction_registry else None))
            result.artifacts["asset_bindings.json"] = str(out / "asset_bindings.json")
            result.artifacts["raw_task_understanding.json"] = str(raw_task_path)
            if (out / "raw_task_understanding.txt").is_file():
                result.artifacts["raw_task_understanding.txt"] = str(out / "raw_task_understanding.txt")
            result.artifacts["normalized_task_parse.json"] = str(normalized_task_path)
            if "validation_path" in locals():
                result.artifacts["semantic_plan_validation.json"] = str(validation_path)
            if intent.semantic_repairs:
                repairs_path = out / "semantic_repairs.json"
                repairs_path.write_text(json.dumps(intent.semantic_repairs, ensure_ascii=False, indent=2), encoding="utf-8")
                result.artifacts["semantic_repairs.json"] = str(repairs_path)
            result.artifacts.update(planner_artifacts)
            if "raw_task_path" in locals():
                result.artifacts["raw_task_understanding.json"] = str(raw_task_path)
            if (out / "raw_task_understanding.txt").is_file():
                result.artifacts["raw_task_understanding.txt"] = str(out / "raw_task_understanding.txt")
            if (out / "normalized_task_parse.json").is_file():
                result.artifacts["normalized_task_parse.json"] = str(out / "normalized_task_parse.json")
            Path(result.artifacts["asset_bindings.json"]).write_text(json.dumps(asset_bindings, ensure_ascii=False, indent=2), encoding="utf-8")
            self._add_observation_artifacts(result.artifacts, observation)
            if scene is None:
                result.artifacts["scene.xml"] = str(xml_path)
                result.artifacts["interaction_registry.json"] = str(generated_interactions)
            return self._write_result(result, out)
        except (OSError, ValueError, KeyError, RuntimeError, ValidationError, ModelCallBudgetExceeded) as exc:
            if isinstance(exc, GroundingResolutionError):
                visual_grounding = exc.visual_grounding
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
                else "model_output_truncated" if "model_output_truncated" in message
                else "clarification_required" if "semantic_conflict:" in message
                else "task_semantic_invalid" if "task_semantic_invalid:" in message
                else "asset_missing" if "asset_missing" in message
                else "unsupported_recipe" if "unsupported_recipe" in message
                else "grounding_ambiguous" if "grounding_ambiguous" in message
                else "grounding_candidate_incomplete" if "grounding_candidate_incomplete" in message
                else "grounding_candidate_count_mismatch" if "grounding_candidate_count_mismatch" in message
                else "grounding_failed" if "grounding_failed" in message
                else "relation_not_satisfied" if "relation_not_satisfied" in message
                else "scene_generation_constraint_failed" if isinstance(exc, SceneConstraintError) else "planning_failed"
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
            result = PipelineResult(task_intent=intent.model_dump(mode="json") if intent else {"instruction": instruction}, scene_registry=registry.model_dump(mode="json") if registry else {}, visual_grounding=visual_grounding, status=status, model_call_count=budget.calls, model_usage=budget.summary(), planner=planner_used, route=route, error=str(exc), source_scene=failure_scene, interaction_registry=failure_registry)
            result.artifacts.update(planner_artifacts)
            if "raw_task_path" in locals():
                result.artifacts["raw_task_understanding.json"] = str(raw_task_path)
            if planner_used == "qwen":
                validation_path = out / "semantic_plan_validation.json"
                validation_path.write_text(json.dumps({"status": status, "error": message}, ensure_ascii=False, indent=2), encoding="utf-8")
                result.artifacts["semantic_plan_validation.json"] = str(validation_path)
            if observation is not None: self._add_observation_artifacts(result.artifacts, observation)
            return self._write_result(result, out)

    def _resolve_candidate_map(self, intent, entities, candidate_map, registry, observation, positions, budget, excluded_object_ids=None):
        """Complete provider candidates, then perform one final relation resolve."""
        instance_by_id = {item.object_id: item for item in observation.instances}
        positions = dict(positions)
        for values in candidate_map.values():
            for value in values:
                if value.get("world_position") is not None:
                    positions.setdefault(value["object_id"], value["world_position"])
        def required_count(entity):
            return max(int(entity.count), 2) if entity.quantity_mode.value == "candidate_pool" else 1

        unresolved = [
            entity for entity in entities
            if len({item.get("object_id") for item in candidate_map.get(entity.entity_id, [])}) < required_count(entity)
        ]
        vision_used = bool(unresolved)
        visual_grounding = {
            "method": "candidate_map",
            "providers": {},
            "vision_used": vision_used,
            "candidates": candidate_map,
        }
        if unresolved:
            selection_entities = {
                value for relation in intent.spatial_relations if relation.scope == "selection"
                for value in (relation.subject, relation.reference) if value
            }
            queries = [VisionQuery(id=entity.entity_id, name=entity.semantic_name, category=entity.category, color=entity.color, all=(entity.quantity_mode.value == "candidate_pool" or entity.entity_id in selection_entities)) for entity in unresolved]
            budget.consume("vision_grounding")
            detection = self.vision.detect(VisionGroundingRequest(entities=queries, rgb_path=str(observation.rgb_path)))
            self._capture(budget, "vision_grounding", self.vision)
            truth = [{"object_id": item.object_id, "bbox": item.bbox} for item in observation.instances if item.bbox is not None]
            relevant = [item for item in detection.detections if item.entity_id in {entity.entity_id for entity in unresolved}]
            for index, item in enumerate(relevant, 1):
                if item.detection_id is None:
                    item.detection_id = f"d-{index}"
            matches, unmatched, ambiguous = match_detections(relevant, truth)
            visual_grounding.update({
                "detections": [item.model_dump(mode="json") for item in detection.detections],
                "truth": truth,
                "matches": [{"detection_id": detection_id, "object_id": object_id, "iou": score} for detection_id, object_id, score in matches],
                "unmatched": unmatched, "ambiguous": ambiguous,
            })
            if unmatched or ambiguous:
                prefix = "grounding_ambiguous" if ambiguous else "grounding_failed"
                raise GroundingResolutionError(
                    f"{prefix}: unmatched={unmatched}; ambiguous={ambiguous}",
                    visual_grounding,
                )
            for detection_id, object_id, score in matches:
                candidate = next(item for item in relevant if item.detection_id == detection_id)
                if object_id in (excluded_object_ids or set()):
                    continue
                merge_candidate(candidate_map[candidate.entity_id], GroundingCandidate(
                    object_id=object_id, sources={"vision"}, detection_bbox=tuple(candidate.bbox),
                    instance_bbox=instance_by_id[object_id].bbox, bbox_iou=score,
                ))
        incomplete = [
            (entity.entity_id, required_count(entity), len({item.get("object_id") for item in candidate_map.get(entity.entity_id, [])}))
            for entity in entities
            if len({item.get("object_id") for item in candidate_map.get(entity.entity_id, [])}) < required_count(entity)
        ]
        if incomplete:
            raise GroundingResolutionError(
                "grounding_candidate_incomplete: " + "; ".join(
                    f"{entity_id} expected at least {expected}, resolved {resolved}"
                    for entity_id, expected, resolved in incomplete
                ),
                visual_grounding,
            )
        mismatched = [
            (entity.entity_id, int(entity.count), len({item.get("object_id") for item in candidate_map.get(entity.entity_id, [])}))
            for entity in entities
            if entity.quantity_mode.value == "candidate_pool"
            and int(entity.count) > 1
            and len({item.get("object_id") for item in candidate_map.get(entity.entity_id, [])}) > int(entity.count)
        ]
        if mismatched:
            raise GroundingResolutionError(
                "grounding_candidate_count_mismatch: " + "; ".join(
                    f"{entity_id} expected exactly {expected}, resolved {resolved}"
                    for entity_id, expected, resolved in mismatched
                ),
                visual_grounding,
            )
        selected = WorldRelationResolver().resolve(
            intent, candidate_map, positions, bounds=_registry_bounds(registry, positions)
        )
        grounded = []
        for entity in entities:
            choice = selected[entity.entity_id]
            object_id = choice["object_id"]
            instance = instance_by_id.get(object_id)
            sources = set(choice.get("sources", []))
            if instance is None and "interaction_registry" not in sources:
                raise ValueError(f"grounding_failed: candidate is not visible in scene: {object_id}")
            method = "dialogue_binding" if "dialogue" in sources else "interaction_registry" if "interaction_registry" in sources else "semantic_cache" if "semantic_map" in sources else "detector_iou" if "geometry" in sources else "vlm_iou"
            grounded.append(GroundedEntity(
                entity_id=entity.entity_id, semantic_name=entity.semantic_name, object_id=object_id,
                body_name=instance.body_name if instance is not None else choice.get("body_name"),
                category=entity.category, color=entity.color,
                aliases=entity.aliases, quantity_mode=entity.quantity_mode, grounding_method=method,
                detection_bbox=choice.get("detection_bbox"),
                instance_bbox=instance.bbox if instance is not None else None,
                bbox_iou=choice.get("bbox_iou"),
            ))
            visual_grounding["providers"][entity.entity_id] = sorted(sources)
        provider_sets = list(visual_grounding["providers"].values())
        if not vision_used and provider_sets:
            if all("dialogue" in sources for sources in provider_sets):
                visual_grounding["method"] = "dialogue_binding"
            elif all("interaction_registry" in sources for sources in provider_sets):
                visual_grounding["method"] = "interaction_registry"
            elif all("semantic_map" in sources for sources in provider_sets):
                visual_grounding["method"] = "semantic_cache"
        return grounded, visual_grounding

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
        operation_outcomes = []
        if result.status == "accepted" and isinstance(result.skill_plan, dict):
            operation_ids = []
            for step in result.skill_plan.get("steps", []):
                if step.get("operation_id") not in operation_ids:
                    operation_ids.append(step.get("operation_id"))
            operation_outcomes = [{"operation_id": operation_id, "result": "plan_validated"} for operation_id in operation_ids]
        semantic_plan_outcomes = [
            {"operation_id": item["operation_id"], "status": "plan_validated"}
            for item in operation_outcomes
        ]
        goal_conditions = []
        goal_condition_keys = set()
        if isinstance(result.task_intent, dict):
            for relation in result.task_intent.get("spatial_relations", []):
                if relation.get("scope") == "goal":
                    condition = GoalCondition(
                        operation_id=None,
                        relation=relation["relation"],
                        subject=relation["subject"],
                        reference=relation.get("reference"),
                        source="explicit_goal",
                        verification_mode="geometry",
                    ).model_dump(mode="json")
                    goal_conditions.append(condition)
                    goal_condition_keys.add((condition["relation"], condition["subject"], condition["reference"]))
            for operation in result.task_intent.get("operations", []):
                task_type = operation.get("task_type")
                subject = operation.get("source") or operation.get("target")
                reference = operation.get("destination") or operation.get("reference")
                inferred = {"pick_and_place": "inside", "grasp": "held", "release": "not_held", "open": "open", "close": "closed"}.get(task_type)
                if inferred and subject:
                    key = (inferred, subject, reference)
                    if key in goal_condition_keys:
                        continue
                    mode = "geometry" if inferred == "inside" else "articulation" if inferred in {"open", "closed"} else "world_state"
                    condition = GoalCondition(operation_id=operation.get("operation_id"), relation=inferred, subject=subject, reference=reference, source="operation_inferred", verification_mode=mode).model_dump(mode="json")
                    goal_conditions.append(condition)
                    goal_condition_keys.add(key)
        semantic_validation = {
            "task_intent_validation": {"status": "accepted" if result.task_intent else "missing"},
            "grounding_validation": {"status": "accepted" if result.grounded_task else (result.status if "grounding" in result.status else "not_run")},
            "skill_plan_validation": {
                "status": "accepted" if result.skill_plan else (result.status if result.planner == "qwen" else "not_run"),
                # Keep operation_outcomes for compatibility; the explicit
                # semantic_plan_outcomes field avoids implying physical
                # execution success.
                "operation_outcomes": operation_outcomes,
                "semantic_plan_outcomes": semantic_plan_outcomes,
                "execution_goal_status": "not_verified",
            },
            "semantic_repairs": result.task_intent.get("semantic_repairs", []) if isinstance(result.task_intent, dict) else [],
            "status": result.status,
            "error": result.error,
        }
        payloads = {"task_intent.json": result.task_intent, "scene_registry.json": result.scene_registry, "grounded_task.json": result.grounded_task, "visual_grounding.json": result.visual_grounding, "skill_plan.json": result.skill_plan, "goal_conditions.json": goal_conditions, "semantic_validation.json": semantic_validation, "model_usage.json": result.model_usage, "summary.json": {"status": result.status, "route": result.route, "planner": result.planner, "model_call_count": result.model_call_count, "model_usage": result.model_usage, "planning_provenance": provenance, "error": result.error, "source_scene": result.source_scene, "interaction_registry": result.interaction_registry}}
        for name, payload in payloads.items():
            path = out / name
            if payload is None:
                if path.exists(): path.unlink()
            else:
                path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"); result.artifacts[name] = str(path)
        PipelineEngine._write_grounding_trace(result, out)
        return result

    @staticmethod
    def _write_grounding_trace(result, out):
        """Persist the evidence used to make (or fail) a grounding decision.

        ``visual_grounding.json`` remains the compatibility artifact.  These
        smaller files make it possible to inspect candidate evidence and the
        final identity decision independently, including on failed runs.
        """
        visual = result.visual_grounding or {}
        grounded_entities = (result.grounded_task or {}).get("entities", [])
        candidates = {
            "status": result.status,
            "method": visual.get("method"),
            "detections": visual.get("detections", []),
            "instance_candidates": visual.get("truth", []),
            "matches": visual.get("matches", []),
            "unmatched": visual.get("unmatched", []),
            "ambiguous": visual.get("ambiguous", []),
            "candidate_map": visual.get("candidates", {}),
            "providers": visual.get("providers", {}),
        }
        decisions = {
            "status": result.status,
            "error": result.error,
            "entities": [
                {
                    key: entity.get(key)
                    for key in (
                        "entity_id", "semantic_name", "category", "color",
                        "object_id", "body_name", "grounding_method", "bbox_iou",
                    )
                }
                for entity in grounded_entities
            ],
        }
        for name, payload in {
            "grounding_candidates.json": candidates,
            "grounding_decision.json": decisions,
        }.items():
            path = out / name
            path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            result.artifacts[name] = str(path)
        if visual.get("detections") is not None and "detections" in visual:
            path = out / "raw_vision_grounding.json"
            path.write_text(
                json.dumps({"detections": visual["detections"]}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            result.artifacts["raw_vision_grounding.json"] = str(path)

    @staticmethod
    def _add_observation_artifacts(artifacts, observation):
        for name, path in {"rgb.png": observation.rgb_path, "segmentation.npy": observation.segmentation_path, "segmentation.png": observation.segmentation_visualization_path, "instances.json": observation.instance_index_path}.items(): artifacts[name] = str(path)


def _semantic_cache_candidates(entities, semantic_map: dict[str, Any] | None, registry, excluded_object_ids: set[str] | None = None) -> dict[str, list[dict[str, str]]]:
    if not semantic_map:
        return {}
    objects = semantic_map.get("objects", semantic_map)
    available = {item.object_id for item in registry.objects}
    result: dict[str, list[dict[str, str]]] = {}
    for entity in entities:
        query = {str(entity.semantic_name).casefold(), *(str(alias).casefold() for alias in entity.aliases)}
        matches = []
        for object_id, value in objects.items():
            if object_id not in available or object_id in (excluded_object_ids or set()):
                continue
            if value.get("category") and value.get("category") != entity.category:
                continue
            if entity.color and value.get("attributes", {}).get("color") and value.get("attributes", {}).get("color") != entity.color:
                continue
            labels = {str(object_id).casefold(), *(str(label).casefold() for label in value.get("labels", []))}
            if any(exact_name_match(token, label) for token in query for label in labels):
                matches.append({"object_id": object_id})
        result[entity.entity_id] = matches
    return result


def _geometry_candidates(entities, registry, excluded_object_ids: set[str] | None = None) -> dict[str, list[dict[str, str]]]:
    """Use stable authored body/object names before paying for vision."""
    result: dict[str, list[dict[str, str]]] = {}
    for entity in entities:
        queries = {str(entity.semantic_name).casefold(), *(str(alias).casefold() for alias in entity.aliases)}
        values = []
        for item in registry.objects:
            if item.object_id in (excluded_object_ids or set()):
                continue
            if item.object_id.startswith("scene_object_") and str(item.semantic_name).casefold() == item.body_name.casefold():
                continue
            names = {item.object_id.casefold(), item.body_name.casefold(), str(item.semantic_name).casefold()}
            if any(exact_name_match(query, name) for query in queries for name in names):
                values.append({"object_id": item.object_id})
        result[entity.entity_id] = values
    return result


def _registry_bounds(registry, positions):
    bounds = {}
    for item in registry.objects:
        dimensions = item.dimensions_m
        if not dimensions or item.object_id not in positions:
            continue
        center = positions[item.object_id]
        half = tuple(float(value) / 2 for value in dimensions)
        bounds[item.object_id] = (
            tuple(center[index] - half[index] for index in range(3)),
            tuple(center[index] + half[index] for index in range(3)),
        )
    return bounds
