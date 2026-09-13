import json

import pytest

from robot_agent_sim.contracts.grounded_task import GroundedEntity, GroundedTask
from robot_agent_sim.contracts.skill_plan import SkillPlan, SkillStep
from robot_agent_sim.contracts.task_intent import Operation, SpatialRelation, SpatialRelationType, TaskType
from robot_agent_sim.models.prompts import SKILL_PLANNING_PROMPT
from robot_agent_sim.planning.context_builder import build_planner_context
from robot_agent_sim.planning.recipe_planner import RecipePlanner
from robot_agent_sim.planning.semantic_validator import validate_semantic_plan
from robot_agent_sim.skills.registry import REGISTRY


SIDECAR = "packages/robot_agent_control/demo/common/scenes/scene_001.interactions.json"


def cabinet_task():
    entities = [
        GroundedEntity(entity_id="cabinet_door_01", semantic_name="blue cabinet door", object_id="blue_cabinet_door", grounding_method="interaction_registry"),
        GroundedEntity(entity_id="cabinet_handle_01", semantic_name="blue cabinet handle", object_id="blue_cabinet_handle", grounding_method="interaction_registry"),
        GroundedEntity(entity_id="red_ball_01", semantic_name="red ball", object_id="red_ball", grounding_method="interaction_registry"),
        GroundedEntity(entity_id="upper_compartment_01", semantic_name="blue cabinet upper compartment", object_id="blue_cabinet_upper_compartment", grounding_method="interaction_registry"),
    ]
    operations = [
        Operation(operation_id="op-1", task_type=TaskType.OPEN, target="cabinet_door_01", reference="cabinet_handle_01"),
        Operation(operation_id="op-2", task_type=TaskType.PICK_AND_PLACE, source="red_ball_01", destination="upper_compartment_01", depends_on=["op-1"]),
        Operation(operation_id="op-3", task_type=TaskType.CLOSE, target="cabinet_door_01", reference="cabinet_handle_01", depends_on=["op-2"]),
    ]
    return GroundedTask(instruction="cabinet", task_types=[TaskType.OPEN, TaskType.PICK_AND_PLACE, TaskType.CLOSE], entities=entities, operations=operations, spatial_relations=[SpatialRelation(scope="goal", subject="red_ball_01", relation=SpatialRelationType.INSIDE, reference="upper_compartment_01")], scene_id="scene_001")


def test_context_projects_semantics_and_excludes_physics():
    context = build_planner_context(cabinet_task(), SIDECAR)
    payload = json.dumps(context.model_dump(mode="json"), ensure_ascii=False)
    assert context.operations[0].target == "cabinet_door_01"
    by_id = {item.id: item for item in context.entities}
    assert by_id["cabinet_door_01"].category == "door"
    assert "pullable" in by_id["cabinet_door_01"].affordances
    assert "graspable" in by_id["cabinet_handle_01"].affordances
    assert "part_of:cabinet_door_01" in by_id["cabinet_handle_01"].relations
    assert "placeable" in by_id["upper_compartment_01"].affordances
    for forbidden in ("position", "quaternion", "joint", "trajectory", "bbox", "body_name"):
        assert forbidden not in payload


def test_catalog_and_prompt_do_not_leak_recipes():
    catalog = REGISTRY.prompt_catalog()
    assert "description:" in catalog and "requires: pressable" in catalog
    assert "preconditions:" in catalog and "effects:" in catalog
    assert "open ->" not in catalog and "close ->" not in catalog
    assert "locate -> move -> grasp" not in SKILL_PLANNING_PROMPT


def test_semantic_validator_accepts_noncanonical_extra_locate():
    task = cabinet_task()
    context = build_planner_context(task, SIDECAR)
    plan = RecipePlanner().plan(task)
    # A repeated locate is semantically harmless and must not be exact-matched.
    plan.steps.insert(1, SkillStep(step_id="step-99", operation_id="op-1", skill_name="locate", target_object="blue_cabinet_handle"))
    plan = SkillPlan(task_types=plan.task_types, steps=plan.steps)
    validate_semantic_plan(plan, task, context)


def test_semantic_validator_rejects_press_on_handle():
    task = cabinet_task().model_copy(update={
        "task_types": [TaskType.PRESS],
        "operations": [Operation(operation_id="op-1", task_type=TaskType.PRESS, target="cabinet_handle_01")],
    })
    context = build_planner_context(task, SIDECAR)
    plan = SkillPlan(task_types=[TaskType.PRESS], steps=[
        SkillStep(step_id="step-1", operation_id="op-1", skill_name="locate", target_object="blue_cabinet_handle"),
        SkillStep(step_id="step-2", operation_id="op-1", skill_name="move", target_object="blue_cabinet_handle", semantic_target="grasp_region"),
        SkillStep(step_id="step-3", operation_id="op-1", skill_name="press", target_object="blue_cabinet_handle"),
    ])
    with pytest.raises(ValueError, match="pressable"):
        validate_semantic_plan(plan, task, context)


def test_semantic_validator_rejects_pull_without_contact():
    task = cabinet_task().model_copy(update={"task_types": [TaskType.OPEN], "operations": [Operation(operation_id="op-1", task_type=TaskType.OPEN, target="cabinet_door_01", reference="cabinet_handle_01")]})
    context = build_planner_context(task, SIDECAR)
    plan = SkillPlan(task_types=[TaskType.OPEN], steps=[SkillStep(step_id="step-1", operation_id="op-1", skill_name="pull", target_object="blue_cabinet_door", reference_object="blue_cabinet_handle")])
    with pytest.raises(ValueError, match="grasp/contact"):
        validate_semantic_plan(plan, task, context)
