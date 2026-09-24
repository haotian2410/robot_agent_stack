import json

import pytest
from pydantic import ValidationError

from robot_agent_sim.contracts.task_intent import (
    Operation,
    SpatialRelation,
    SpatialRelationType,
    TaskEntity,
    TaskIntent,
    TaskStatus,
    TaskType,
    QuantityMode,
)
from robot_agent_sim.models.task_understanding import ParseEntity, ParseOperation, ParseRelation, TaskParseLLMOutput, enrich_task
from robot_agent_sim.models.motion_policy import MotionPolicy
from robot_agent_sim.models.fake import FakeTaskUnderstandingProvider
from robot_agent_sim.models.qwen_http import QwenProviderError, _extract_json
from robot_agent_sim.pipeline.engine import PipelineEngine, _semantic_cache_candidates
from robot_agent_sim.grounding.world_relation import RelationAmbiguous, WorldRelationResolver
from robot_agent_sim.grounding.name_matching import exact_name_match
from robot_agent_sim.contracts.grounded_task import GroundedEntity, GroundedTask
from robot_agent_sim.contracts.skill_plan import SkillPlan, SkillStep
from robot_agent_sim.planning.context_builder import PlannerInitialState, build_planner_context
from robot_agent_sim.planning.semantic_validator import validate_semantic_plan
from robot_agent_sim.assets.registry import AssetRegistry
from robot_agent_sim.scene.registry import SceneObject, SceneRegistry
from robot_agent_sim.scene.constraints import SceneConstraintError, validate_generated_scene


def _entities():
    return [
        TaskEntity(entity_id="apple", semantic_name="apple", category="fruit"),
        TaskEntity(entity_id="basket", semantic_name="basket", category="container"),
    ]


def test_operation_contract_rejects_missing_pick_destination():
    with pytest.raises(ValidationError, match="task_semantic_invalid"):
        TaskIntent(
            status=TaskStatus.ACCEPTED, instruction="put apple", task_types=[TaskType.PICK_AND_PLACE],
            entities=_entities(), operations=[Operation(operation_id="op-1", task_type=TaskType.PICK_AND_PLACE, source="apple")],
        )


def test_operation_contract_rejects_open_without_handle():
    with pytest.raises(ValidationError, match="task_semantic_invalid"):
        TaskIntent(
            status=TaskStatus.ACCEPTED, instruction="open door", task_types=[TaskType.OPEN],
            entities=_entities(), operations=[Operation(operation_id="op-1", task_type=TaskType.OPEN, target="basket")],
        )


def test_binary_relation_requires_reference_and_unary_forbids_it():
    with pytest.raises(ValidationError, match="requires reference"):
        SpatialRelation(subject="apple", relation=SpatialRelationType.INSIDE)
    with pytest.raises(ValidationError, match="forbids reference"):
        SpatialRelation(subject="apple", relation=SpatialRelationType.LEFT, reference="basket")


def test_conflicting_relations_are_clarification_required(tmp_path):
    class ConflictingProvider:
        def understand(self, request):
            return TaskParseLLMOutput(
                status="accepted",
                entities=[ParseEntity(id="apple", name="apple", category="fruit")],
                operations=[ParseOperation(type="grasp", target="apple")],
                relations=[
                    ParseRelation(subject="apple", relation=SpatialRelationType.LEFT),
                    ParseRelation(subject="apple", relation=SpatialRelationType.RIGHT),
                ],
            )

    result = PipelineEngine(understanding=ConflictingProvider()).plan("抓苹果", output_dir=tmp_path)
    assert result.status == "clarification_required"


def test_explicit_motion_text_overrides_conflicting_model_and_records_repair():
    parsed = TaskParseLLMOutput(
        status="accepted",
        entities=[ParseEntity(id="apple", name="apple", category="fruit")],
        operations=[ParseOperation(type="move", target="apple", motion_direction="left", distance_m=0.5)],
        raw_direction="left", distance_m=0.5,
    )
    intent = enrich_task(parsed, "把苹果向右移动5厘米", MotionPolicy(default_relative_distance_m=0.2))
    operation = intent.operations[0]
    assert operation.motion_direction.value == "right"
    assert operation.distance_m == 0.05
    assert len(intent.semantic_repairs) == 2


def test_multiple_moves_cannot_broadcast_top_level_direction():
    parsed = TaskParseLLMOutput(
        status="accepted",
        entities=[
            ParseEntity(id="apple", name="apple", category="fruit"),
            ParseEntity(id="banana", name="banana", category="fruit"),
        ],
        operations=[
            ParseOperation(type="move", target="apple"),
            ParseOperation(type="move", target="banana"),
        ],
        raw_direction="right",
    )
    with pytest.raises(ValueError, match="every move operation requires operation-local"):
        enrich_task(parsed, "先移动苹果，再移动香蕉")


def test_all_quantity_is_rejected_instead_of_silently_executing_one():
    request = type("Request", (), {"instruction": "把三个苹果都放进篮子"})()
    parsed = FakeTaskUnderstandingProvider().understand(request)
    intent = enrich_task(parsed, request.instruction)
    assert intent.status.value == "unsupported_multi_object_execution"
    assert intent.operations == []


@pytest.mark.parametrize(
    ("instruction", "operations"),
    [
        (
            "苹果左移5厘米，然后香蕉左移10厘米",
            [ParseOperation(type="move", target="apple", motion_direction="left", distance_m=0.05), ParseOperation(type="move", target="banana", motion_direction="left", distance_m=0.10)],
        ),
        (
            "苹果左移5厘米，然后香蕉右移10厘米",
            [ParseOperation(type="move", target="apple", motion_direction="left", distance_m=0.05), ParseOperation(type="move", target="banana", motion_direction="right", distance_m=0.10)],
        ),
        (
            "苹果左移5厘米，然后香蕉左移10厘米",
            [ParseOperation(type="move", target="apple", motion_direction="left", distance_m=0.05)],
        ),
    ],
)
def test_multiple_directional_moves_require_clarification(instruction, operations):
    parsed = TaskParseLLMOutput(
        status="accepted",
        entities=[
            ParseEntity(id="apple", name="apple", category="fruit"),
            ParseEntity(id="banana", name="banana", category="fruit"),
        ],
        operations=operations,
    )
    intent = enrich_task(parsed, instruction)
    assert intent.status == TaskStatus.CLARIFICATION_REQUIRED
    assert intent.operations == []


def test_single_directional_move_keeps_deterministic_text_repair():
    parsed = TaskParseLLMOutput(
        status="accepted",
        entities=[ParseEntity(id="apple", name="apple", category="fruit")],
        operations=[ParseOperation(type="move", target="apple", motion_direction="right", distance_m=0.5)],
    )
    intent = enrich_task(parsed, "苹果左移5厘米")
    assert intent.status == TaskStatus.ACCEPTED
    assert intent.operations[0].motion_direction.value == "left"
    assert intent.operations[0].distance_m == pytest.approx(0.05)


def test_candidate_pool_quantity_remains_supported():
    request = type("Request", (), {"instruction": "把三个苹果中靠近篮子的苹果放进篮子"})()
    parsed = FakeTaskUnderstandingProvider().understand(request)
    assert parsed.entities[0].quantity_mode == QuantityMode.CANDIDATE_POOL
    assert enrich_task(parsed, request.instruction).status == TaskStatus.ACCEPTED


def test_goal_conditions_are_structured_and_execution_remains_unverified(tmp_path):
    result = PipelineEngine().plan("把苹果放进篮子", output_dir=tmp_path)
    assert result.status == "accepted"
    goals = json.loads((tmp_path / "goal_conditions.json").read_text())
    assert goals == [{
        "operation_id": None,
        "relation": "inside",
        "subject": "apple_01",
        "reference": "basket_01",
        "source": "explicit_goal",
        "verification_mode": "geometry",
    }]
    validation = json.loads((tmp_path / "semantic_validation.json").read_text())
    assert validation["skill_plan_validation"]["execution_goal_status"] == "not_verified"
    assert validation["skill_plan_validation"]["operation_outcomes"] == [{"operation_id": "op-1", "result": "plan_validated"}]


def test_candidate_pool_without_relation_still_generates_minimum_pool(tmp_path):
    class PoolProvider:
        def understand(self, request):
            return TaskParseLLMOutput(
                status="accepted",
                entities=[ParseEntity(
                    id="apple", name="apple", category="fruit",
                    count=1, quantity_mode=QuantityMode.CANDIDATE_POOL,
                )],
                operations=[ParseOperation(type="grasp", target="apple")],
            )

    result = PipelineEngine(understanding=PoolProvider()).plan("抓候选苹果", output_dir=tmp_path)
    assert result.status == "accepted"
    candidates = [item for item in result.scene_registry["objects"] if item.get("candidate_for") == "apple"]
    assert len(candidates) == 2


def test_route_a_candidate_pool_generates_count_and_leftmost_binding(tmp_path):
    result = PipelineEngine().plan("把三个苹果中最左边的苹果抓起来", output_dir=tmp_path)
    assert result.status == "accepted"
    candidates = [item for item in result.scene_registry["objects"] if item.get("candidate_for") == "apple_01"]
    assert len(candidates) == 3
    selected = next(item for item in candidates if item["object_id"] == result.scene_registry["bindings"]["apple_01"])
    assert selected["position"][0] == min(item["position"][0] for item in candidates)


@pytest.mark.parametrize("instruction", ["把三个苹果中最高的苹果抓起来", "把三个苹果中最低的苹果抓起来"])
def test_route_a_vertical_ranking_requires_supported_height_levels(tmp_path, instruction):
    result = PipelineEngine().plan(instruction, output_dir=tmp_path)
    assert result.status == "scene_generation_constraint_failed"
    assert "supported height levels" in (result.error or "")


@pytest.mark.parametrize(
    ("instruction", "selector"),
    [
        ("把三个苹果中最右边的苹果抓起来", lambda item, reference: item["position"][0]),
        ("把三个苹果中最前面的苹果抓起来", lambda item, reference: item["position"][1]),
        ("把三个苹果中最后面的苹果抓起来", lambda item, reference: item["position"][1]),
        ("把三个苹果中靠近篮子的苹果抓起来", lambda item, reference: (
            (item["position"][0] - reference["position"][0]) ** 2
            + (item["position"][1] - reference["position"][1]) ** 2
        )),
    ],
)
def test_route_a_candidate_pool_preserves_count_for_other_rankings(tmp_path, instruction, selector):
    result = PipelineEngine().plan(instruction, output_dir=tmp_path)
    assert result.status == "accepted"
    candidates = [item for item in result.scene_registry["objects"] if item.get("candidate_for") == "apple_01"]
    assert len(candidates) == 3
    selected = next(item for item in candidates if item["object_id"] == result.scene_registry["bindings"]["apple_01"])
    basket = next((item for item in result.scene_registry["objects"] if item["semantic_name"] == "basket"), None)
    scores = [selector(item, basket) for item in candidates]
    if "最右边" in instruction or "最前面" in instruction:
        assert selector(selected, basket) == max(scores)
    elif "最后面" in instruction:
        assert selector(selected, basket) == min(scores)
    else:
        assert selector(selected, basket) == min(scores)


def test_held_pick_and_place_does_not_require_regrasp():
    task = GroundedTask(
        instruction="place held apple",
        task_types=[TaskType.PICK_AND_PLACE],
        entities=[
            GroundedEntity(entity_id="apple", semantic_name="apple", object_id="apple-01", category="fruit", grounding_method="asset_scene_binding"),
            GroundedEntity(entity_id="basket", semantic_name="basket", object_id="basket-01", category="container", grounding_method="asset_scene_binding"),
        ],
        operations=[Operation(operation_id="op-1", task_type=TaskType.PICK_AND_PLACE, source="apple", destination="basket")],
        spatial_relations=[], scene_id="scene",
    )
    context = build_planner_context(task, initial_state=PlannerInitialState(held_entity="apple"))
    plan = SkillPlan(task_types=[TaskType.PICK_AND_PLACE], steps=[
        SkillStep(step_id="step-1", operation_id="op-1", skill_name="locate", target_object="basket-01"),
        SkillStep(step_id="step-2", operation_id="op-1", skill_name="move", target_object="basket-01", semantic_target="container_interior"),
        SkillStep(step_id="step-3", operation_id="op-1", skill_name="release", target_object="apple-01", reference_object="basket-01", semantic_target="container_interior"),
    ])
    validate_semantic_plan(plan, task, context)


def test_held_pick_and_place_rejects_wrong_held_object():
    task = GroundedTask(
        instruction="place apple",
        task_types=[TaskType.PICK_AND_PLACE],
        entities=[
            GroundedEntity(entity_id="apple", semantic_name="apple", object_id="apple-01", category="fruit", grounding_method="asset_scene_binding"),
            GroundedEntity(entity_id="basket", semantic_name="basket", object_id="basket-01", category="container", grounding_method="asset_scene_binding"),
        ], operations=[Operation(operation_id="op-1", task_type=TaskType.PICK_AND_PLACE, source="apple", destination="basket")], spatial_relations=[], scene_id="scene",
    )
    context = build_planner_context(task, initial_state=PlannerInitialState(held_entity="banana"))
    plan = SkillPlan(task_types=[TaskType.PICK_AND_PLACE], steps=[
        SkillStep(step_id="step-1", operation_id="op-1", skill_name="locate", target_object="basket-01"),
        SkillStep(step_id="step-2", operation_id="op-1", skill_name="move", target_object="basket-01", semantic_target="container_interior"),
        SkillStep(step_id="step-3", operation_id="op-1", skill_name="release", target_object="apple-01", reference_object="basket-01", semantic_target="container_interior"),
    ])
    with pytest.raises(ValueError, match="release requires held target"):
        validate_semantic_plan(plan, task, context)


def test_pick_and_place_rejects_leaving_destination_before_release():
    task = GroundedTask(
        instruction="place apple in basket", task_types=[TaskType.PICK_AND_PLACE],
        entities=[
            GroundedEntity(entity_id="apple", semantic_name="apple", object_id="apple-01", category="fruit", grounding_method="asset_scene_binding"),
            GroundedEntity(entity_id="basket", semantic_name="basket", object_id="basket-01", category="container", grounding_method="asset_scene_binding"),
        ],
        operations=[Operation(operation_id="op-1", task_type=TaskType.PICK_AND_PLACE, source="apple", destination="basket")],
        spatial_relations=[], scene_id="scene",
    )
    context = build_planner_context(task, initial_state=PlannerInitialState(held_entity="apple"))
    plan = SkillPlan(task_types=[TaskType.PICK_AND_PLACE], steps=[
        SkillStep(step_id="step-1", operation_id="op-1", skill_name="locate", target_object="basket-01"),
        SkillStep(step_id="step-2", operation_id="op-1", skill_name="move", target_object="basket-01", semantic_target="container_interior"),
        SkillStep(step_id="step-3", operation_id="op-1", skill_name="locate", target_object="apple-01"),
        SkillStep(step_id="step-4", operation_id="op-1", skill_name="move", target_object="apple-01", semantic_target="relative_motion"),
        SkillStep(step_id="step-5", operation_id="op-1", skill_name="release", target_object="apple-01", reference_object="basket-01", semantic_target="container_interior"),
    ])
    with pytest.raises(ValueError, match="semantic_plan_invalid"):
        validate_semantic_plan(plan, task, context)


def test_pick_and_place_rejects_release_with_wrong_reference():
    task = GroundedTask(
        instruction="place apple in basket", task_types=[TaskType.PICK_AND_PLACE],
        entities=[
            GroundedEntity(entity_id="apple", semantic_name="apple", object_id="apple-01", category="fruit", grounding_method="asset_scene_binding"),
            GroundedEntity(entity_id="basket", semantic_name="basket", object_id="basket-01", category="container", grounding_method="asset_scene_binding"),
            GroundedEntity(entity_id="box", semantic_name="box", object_id="box-01", category="container", grounding_method="asset_scene_binding"),
        ],
        operations=[Operation(operation_id="op-1", task_type=TaskType.PICK_AND_PLACE, source="apple", destination="basket")],
        spatial_relations=[], scene_id="scene",
    )
    context = build_planner_context(task, initial_state=PlannerInitialState(held_entity="apple"))
    plan = SkillPlan(task_types=[TaskType.PICK_AND_PLACE], steps=[
        SkillStep(step_id="step-1", operation_id="op-1", skill_name="locate", target_object="box-01"),
        SkillStep(step_id="step-2", operation_id="op-1", skill_name="move", target_object="box-01", semantic_target="container_interior"),
        SkillStep(step_id="step-3", operation_id="op-1", skill_name="release", target_object="apple-01", reference_object="box-01", semantic_target="container_interior"),
    ])
    with pytest.raises(ValueError, match="semantic_plan_invalid"):
        validate_semantic_plan(plan, task, context)


def test_release_and_move_operations_require_their_terminal_effects():
    apple = GroundedEntity(entity_id="apple", semantic_name="apple", object_id="apple-01", category="fruit", grounding_method="asset_scene_binding")
    release_task = GroundedTask(
        instruction="release apple", task_types=[TaskType.RELEASE], entities=[apple],
        operations=[Operation(operation_id="op-1", task_type=TaskType.RELEASE, target="apple")], spatial_relations=[], scene_id="scene",
    )
    release_context = build_planner_context(release_task, initial_state=PlannerInitialState(held_entity="apple"))
    locate_only = SkillPlan(task_types=[TaskType.RELEASE], steps=[
        SkillStep(step_id="step-1", operation_id="op-1", skill_name="locate", target_object="apple-01"),
    ])
    with pytest.raises(ValueError, match="release operation did not execute release"):
        validate_semantic_plan(locate_only, release_task, release_context)
    release_then_regrasp = SkillPlan(task_types=[TaskType.RELEASE], steps=[
        SkillStep(step_id="step-1", operation_id="op-1", skill_name="release", target_object="apple-01"),
        SkillStep(step_id="step-2", operation_id="op-1", skill_name="locate", target_object="apple-01"),
        SkillStep(step_id="step-3", operation_id="op-1", skill_name="move", target_object="apple-01", semantic_target="grasp_region"),
        SkillStep(step_id="step-4", operation_id="op-1", skill_name="grasp", target_object="apple-01"),
    ])
    with pytest.raises(ValueError, match="release operation ended holding released target"):
        validate_semantic_plan(release_then_regrasp, release_task, release_context)

    move_task = GroundedTask(
        instruction="move apple", task_types=[TaskType.MOVE], entities=[apple],
        operations=[Operation(operation_id="op-1", task_type=TaskType.MOVE, target="apple")], spatial_relations=[], scene_id="scene",
    )
    move_context = build_planner_context(move_task)
    move_locate_only = SkillPlan(task_types=[TaskType.MOVE], steps=[
        SkillStep(step_id="step-1", operation_id="op-1", skill_name="locate", target_object="apple-01"),
        SkillStep(step_id="step-2", operation_id="op-1", skill_name="move", target_object="apple-01", semantic_target="grasp_region"),
    ])
    with pytest.raises(ValueError, match="move operation did not execute matching semantic move"):
        validate_semantic_plan(move_locate_only, move_task, move_context)


@pytest.mark.parametrize(
    ("task_type", "skills", "message"),
    [
        (TaskType.OPEN, ("pull", "push"), "open operation did not end with matching pull"),
        (TaskType.CLOSE, ("push", "pull"), "close operation did not end with matching push"),
    ],
)
def test_mechanism_operation_requires_matching_final_effect(task_type, skills, message):
    task = GroundedTask(
        instruction=task_type.value, task_types=[task_type],
        entities=[
            GroundedEntity(entity_id="door", semantic_name="door", object_id="door-01", category="door", grounding_method="interaction_registry"),
            GroundedEntity(entity_id="handle", semantic_name="handle", object_id="handle-01", category="handle", grounding_method="interaction_registry"),
        ],
        operations=[Operation(operation_id="op-1", task_type=task_type, target="door", reference="handle")],
        spatial_relations=[], scene_id="scene",
    )
    registry = {"objects": {
        "door-01": {"action_requests": {"pull": {}, "push": {}}},
        "handle-01": {"spatial": {"anchors": {"grasp": {}}}, "affordances": {
            "mechanism": {"acting_target": "door-01", "contact_target": "handle-01"},
        }},
    }}
    context = build_planner_context(task, registry, initial_state=PlannerInitialState(held_entity="handle"))
    plan = SkillPlan(task_types=[task_type], steps=[
        SkillStep(step_id=f"step-{index}", operation_id="op-1", skill_name=skill, target_object="door-01", reference_object="handle-01")
        for index, skill in enumerate(skills, 1)
    ])
    with pytest.raises(ValueError, match=message):
        validate_semantic_plan(plan, task, context)


def test_specific_missing_asset_does_not_fallback_by_category():
    with pytest.raises(KeyError, match="asset_missing"):
        AssetRegistry().resolve("container", "cup")
    with pytest.raises(KeyError, match="asset_missing"):
        AssetRegistry().resolve("container", "cup", aliases=["container"])


def test_generic_ball_can_use_category_asset_but_baseball_stays_exact():
    registry = AssetRegistry()
    generic = registry.resolve("ball", "球")
    specific = registry.resolve("ball", "棒球")
    assert generic.model_name == "baseball"
    assert specific.model_name == "baseball"


def test_world_relations_apply_hard_filter_then_nearest_ranking():
    intent = TaskIntent(
        status=TaskStatus.ACCEPTED, instruction="select", task_types=[TaskType.GRASP],
        entities=_entities(), operations=[Operation(operation_id="op-1", task_type=TaskType.GRASP, target="apple")],
        spatial_relations=[
            SpatialRelation(subject="apple", relation=SpatialRelationType.LEFT_OF, reference="basket"),
            SpatialRelation(subject="apple", relation=SpatialRelationType.NEAREST, reference="basket"),
        ],
    )
    selected = WorldRelationResolver().resolve(
        intent,
        {"apple": [{"object_id": "a-left-near"}, {"object_id": "a-right-near"}], "basket": [{"object_id": "basket-1"}]},
        {"a-left-near": (-0.2, 0.0, 0.0), "a-right-near": (0.2, 0.0, 0.0), "basket-1": (0.0, 0.0, 0.0)},
    )
    assert selected["apple"]["object_id"] == "a-left-near"


def test_nearest_distance_tie_is_ambiguous():
    intent = TaskIntent(
        status=TaskStatus.ACCEPTED, instruction="select", task_types=[TaskType.GRASP],
        entities=_entities(), operations=[Operation(operation_id="op-1", task_type=TaskType.GRASP, target="apple")],
        spatial_relations=[SpatialRelation(subject="apple", relation=SpatialRelationType.NEAREST, reference="basket")],
    )
    with pytest.raises(RelationAmbiguous, match="distance tie"):
        WorldRelationResolver().resolve(
            intent,
            {"apple": [{"object_id": "a1"}, {"object_id": "a2"}], "basket": [{"object_id": "basket-1"}]},
            {"a1": (-0.1, 0.0, 0.0), "a2": (0.1, 0.0, 0.0), "basket-1": (0.0, 0.0, 0.0)},
        )


def test_rightmost_tie_is_ambiguous():
    intent = TaskIntent(
        status=TaskStatus.ACCEPTED, instruction="select", task_types=[TaskType.GRASP],
        entities=[TaskEntity(entity_id="apple", semantic_name="apple", category="fruit")],
        operations=[Operation(operation_id="op-1", task_type=TaskType.GRASP, target="apple")],
        spatial_relations=[SpatialRelation(subject="apple", relation=SpatialRelationType.RIGHTMOST)],
    )
    with pytest.raises(RelationAmbiguous, match="distance tie"):
        WorldRelationResolver().resolve(
            intent,
            {"apple": [{"object_id": "a1"}, {"object_id": "a2"}]},
            {"a1": (0.2, 0.0, 0.0), "a2": (0.2, 0.1, 0.0)},
        )


@pytest.mark.parametrize(
    "relations",
    [
        [SpatialRelation(subject="apple", relation=SpatialRelationType.LEFTMOST), SpatialRelation(subject="apple", relation=SpatialRelationType.RIGHTMOST)],
        [SpatialRelation(subject="apple", relation=SpatialRelationType.FRONTMOST), SpatialRelation(subject="apple", relation=SpatialRelationType.BACKMOST)],
        [SpatialRelation(subject="apple", relation=SpatialRelationType.HIGHEST), SpatialRelation(subject="apple", relation=SpatialRelationType.LOWEST)],
    ],
)
def test_opposite_ranking_relations_are_semantic_conflicts(relations):
    with pytest.raises(ValueError, match="semantic_conflict"):
        TaskIntent(
            status=TaskStatus.ACCEPTED,
            instruction="conflicting ranking",
            task_types=[TaskType.GRASP],
            entities=[TaskEntity(entity_id="apple", semantic_name="apple", category="fruit")],
            operations=[Operation(operation_id="op-1", task_type=TaskType.GRASP, target="apple")],
            spatial_relations=relations,
        )


def test_resolver_rejects_colliding_bindings_for_distinct_entities():
    intent = TaskIntent(
        status=TaskStatus.ACCEPTED, instruction="select two apples", task_types=[TaskType.GRASP],
        entities=[
            TaskEntity(entity_id="apple_a", semantic_name="apple", category="fruit"),
            TaskEntity(entity_id="apple_b", semantic_name="apple", category="fruit"),
        ],
        operations=[
            Operation(operation_id="op-1", task_type=TaskType.GRASP, target="apple_a"),
            Operation(operation_id="op-2", task_type=TaskType.GRASP, target="apple_b", depends_on=["op-1"]),
        ],
    )
    with pytest.raises(RelationAmbiguous, match="distinct object assignment"):
        WorldRelationResolver().resolve(
            intent,
            {"apple_a": [{"object_id": "apple-01"}], "apple_b": [{"object_id": "apple-01"}]},
            {"apple-01": (0.0, 0.0, 0.0)},
        )


def test_inside_selection_requires_and_uses_container_bounds():
    intent = TaskIntent(
        status=TaskStatus.ACCEPTED, instruction="select", task_types=[TaskType.GRASP],
        entities=_entities(), operations=[Operation(operation_id="op-1", task_type=TaskType.GRASP, target="apple")],
        spatial_relations=[SpatialRelation(subject="apple", relation=SpatialRelationType.INSIDE, reference="basket")],
    )
    selected = WorldRelationResolver().resolve(
        intent,
        {"apple": [{"object_id": "a1"}, {"object_id": "a2"}], "basket": [{"object_id": "basket-1"}]},
        {"a1": (0.0, 0.0, 0.02), "a2": (0.3, 0.0, 0.0), "basket-1": (0.0, 0.0, 0.0)},
        bounds={"basket-1": ((-0.1, -0.1, -0.1), (0.1, 0.1, 0.1))},
    )
    assert selected["apple"]["object_id"] == "a1"


def test_inside_selection_requires_full_subject_aabb_containment():
    intent = TaskIntent(
        status=TaskStatus.ACCEPTED, instruction="select", task_types=[TaskType.GRASP],
        entities=_entities(), operations=[Operation(operation_id="op-1", task_type=TaskType.GRASP, target="apple")],
        spatial_relations=[SpatialRelation(subject="apple", relation=SpatialRelationType.INSIDE, reference="basket")],
    )
    selected = WorldRelationResolver().resolve(
        intent,
        {"apple": [{"object_id": "center-only"}, {"object_id": "contained"}], "basket": [{"object_id": "basket-1"}]},
        {"center-only": (0.08, 0.0, 0.0), "contained": (0.0, 0.0, 0.0), "basket-1": (0.0, 0.0, 0.0)},
        bounds={
            "basket-1": ((-0.1, -0.1, -0.1), (0.1, 0.1, 0.1)),
            "center-only": ((0.03, -0.05, -0.05), (0.13, 0.05, 0.05)),
            "contained": ((-0.05, -0.05, -0.05), (0.05, 0.05, 0.05)),
        },
    )
    assert selected["apple"]["object_id"] == "contained"


def test_generated_scene_inside_constraint_rejects_partial_aabb_overlap():
    intent = TaskIntent(
        status=TaskStatus.ACCEPTED, instruction="inside", task_types=[TaskType.GRASP],
        entities=_entities(), operations=[Operation(operation_id="op-1", task_type=TaskType.GRASP, target="apple")],
        spatial_relations=[SpatialRelation(subject="apple", relation=SpatialRelationType.INSIDE, reference="basket")],
    )
    registry = SceneRegistry(
        scene_id="inside-aabb", robot="ur5e",
        objects=[
            SceneObject(object_id="apple-1", body_name="apple-1", role="target", semantic_name="apple", position=(0.08, 0.0, 0.0), dimensions_m=(0.1, 0.1, 0.1)),
            SceneObject(object_id="basket-1", body_name="basket-1", role="target", semantic_name="basket", position=(0.0, 0.0, 0.0), dimensions_m=(0.2, 0.2, 0.2)),
        ],
        bindings={"apple": "apple-1", "basket": "basket-1"},
    )
    with pytest.raises(SceneConstraintError, match="inside relation not satisfied"):
        validate_generated_scene(intent, registry)


def test_name_matching_does_not_use_dangerous_substrings():
    assert exact_name_match("apple", "apple_01")
    assert not exact_name_match("apple", "pineapple")
    assert not exact_name_match("ball", "baseball")
    assert not exact_name_match("cup", "cupcake")


@pytest.mark.parametrize(
    ("query", "existing"),
    [("apple", "pineapple"), ("ball", "baseball"), ("cup", "cupcake")],
)
def test_grounding_semantic_cache_rejects_substring_matches(query, existing):
    object_id = f"{existing}_01"
    registry = SceneRegistry(
        scene_id="safe-matching", robot="ur5e",
        objects=[SceneObject(
            object_id=object_id, body_name=object_id, role="target",
            semantic_name=existing, source="uploaded",
        )],
    )
    entity = TaskEntity(entity_id="query", semantic_name=query, category="object")
    candidates = _semantic_cache_candidates(
        [entity], {"objects": {object_id: {"labels": [existing]}}}, registry,
    )
    assert candidates == {"query": []}


@pytest.mark.parametrize("goal_relation", [SpatialRelationType.RIGHT, SpatialRelationType.RIGHTMOST])
def test_goal_unary_relation_does_not_change_initial_scene_placement(goal_relation):
    class GoalProvider:
        def __init__(self, with_goal):
            self.with_goal = with_goal

        def understand(self, request):
            return TaskParseLLMOutput(
                status="accepted",
                entities=[ParseEntity(id="apple", name="apple", category="fruit")],
                operations=[ParseOperation(type="grasp", target="apple")],
                relations=[ParseRelation(scope="goal", subject="apple", relation=goal_relation)] if self.with_goal else [],
            )

    baseline = PipelineEngine(understanding=GoalProvider(False)).plan("抓苹果", seed=17)
    goal = PipelineEngine(understanding=GoalProvider(True)).plan("抓苹果", seed=17)
    baseline_position = next(item["position"] for item in baseline.scene_registry["objects"] if item["semantic_name"] == "apple")
    goal_position = next(item["position"] for item in goal.scene_registry["objects"] if item["semantic_name"] == "apple")
    assert goal_position == baseline_position
    assert len(goal.scene_registry["objects"]) == 1


def test_generated_scene_preserves_inside_selection_relation(tmp_path):
    class InsideProvider:
        def understand(self, request):
            return TaskParseLLMOutput(
                status="accepted",
                entities=[
                    ParseEntity(id="apple", name="apple", category="fruit"),
                    ParseEntity(id="basket", name="basket", category="container"),
                ],
                operations=[ParseOperation(type="grasp", target="apple")],
                relations=[ParseRelation(subject="apple", relation=SpatialRelationType.INSIDE, reference="basket")],
            )
    result = PipelineEngine(understanding=InsideProvider()).plan("抓篮子里的苹果", output_dir=tmp_path)
    assert result.status == "accepted"
    apple = next(item for item in result.scene_registry["objects"] if item["semantic_name"] == "apple")
    basket = next(item for item in result.scene_registry["objects"] if item["semantic_name"] == "basket")
    assert abs(apple["position"][0] - basket["position"][0]) <= basket["dimensions_m"][0] / 2


def test_model_output_must_be_one_json_object():
    assert _extract_json('{"ok":true}') == '{"ok": true}'
    assert _extract_json('```json\n{"ok":true}\n```') == '{"ok": true}'
    with pytest.raises(QwenProviderError):
        _extract_json('说明文字 {"example":true} 最终结果 {"ok":true}')
