import json

import pytest
from pydantic import ValidationError

from robot_agent_sim.contracts.grounded_task import GroundedEntity, GroundedTask
from robot_agent_sim.contracts.skill_plan import SkillPlan, SkillStep
from robot_agent_sim.contracts.task_intent import Operation, SpatialRelation, SpatialRelationType, TaskType
from robot_agent_sim.models.prompts import SKILL_PLANNING_PROMPT
from robot_agent_sim.models.skill_planning import SkillPlanLLMOutput
from robot_agent_sim.models.qwen_http import QwenHTTPProvider
from robot_agent_sim.planning.context_builder import build_planner_context
from robot_agent_sim.planning.recipe_planner import RecipePlanner
from robot_agent_sim.planning.semantic_validator import validate_semantic_plan
from robot_agent_sim.skills.registry import REGISTRY
from robot_agent_sim.execution.compiler import compile_execution_bundle


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


def test_root_single_operation_is_rejected_and_raw_is_retained(monkeypatch):
    class Response:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return {"choices": [{"message": {"content": '{"id":"op-1","steps":[]}'}}], "usage": {"prompt_tokens": 17, "completion_tokens": 9}}

    monkeypatch.setattr("robot_agent_sim.models.qwen_http.httpx.post", lambda *args, **kwargs: Response())
    provider = QwenHTTPProvider("http://localhost/v1", "Qwen", use_structured_output="off")
    from robot_agent_sim.models.skill_planning import SkillPlanningRequest
    context = build_planner_context(cabinet_task(), SIDECAR)
    with pytest.raises(ValidationError):
        provider.plan(SkillPlanningRequest(context=context, skill_catalog=REGISTRY.prompt_catalog()))
    assert provider.last_raw_values["skill_planning"] == {"id": "op-1", "steps": []}
    assert provider.calls[-1]["prompt_tokens"] == 17


def test_qwen_response_finish_reason_is_recorded(monkeypatch):
    class Response:
        status_code = 200
        def raise_for_status(self): return None
        def json(self):
            return {"choices": [{"message": {"content": '{"id":"op-1","steps":[]}'}, "finish_reason": "length"}], "usage": {"prompt_tokens": 4, "completion_tokens": 8}}
    monkeypatch.setattr("robot_agent_sim.models.qwen_http.httpx.post", lambda *args, **kwargs: Response())
    provider = QwenHTTPProvider("http://localhost/v1", "Qwen", use_structured_output="off")
    from robot_agent_sim.models.skill_planning import SkillPlanningRequest
    with pytest.raises(ValidationError):
        provider.plan(SkillPlanningRequest(context=build_planner_context(cabinet_task(), SIDECAR), skill_catalog=REGISTRY.prompt_catalog()))
    assert provider.calls[-1]["finish_reason"] == "length"


def test_pipeline_failure_writes_raw_plan_and_usage(tmp_path):
    class BrokenPlanner:
        calls = [{"stage": "skill_planning", "status": "succeeded", "prompt_tokens": 23, "completion_tokens": 11, "finish_reason": "length"}]
        last_raw_values = {"skill_planning": {"id": "op-1", "steps": []}}

        def plan(self, request):
            SkillPlanLLMOutput.model_validate(self.last_raw_values["skill_planning"])

    from robot_agent_sim.pipeline.engine import PipelineEngine
    result = PipelineEngine(planner=BrokenPlanner()).plan("抓取红方块", planner="qwen", output_dir=tmp_path)
    assert result.status == "planning_failed"
    assert (tmp_path / "raw_skill_plan.json").is_file()
    assert result.model_usage["stages"][-1]["prompt_tokens"] == 23
    assert result.model_usage["stages"][-1]["completion_tokens"] == 11
    assert result.model_usage["stages"][-1]["total_tokens"] == 34
    assert result.model_usage["stages"][-1]["finish_reason"] == "length"
    assert result.source_scene is not None
    assert result.interaction_registry is not None


def test_compiler_trace_records_semantic_anchor_mapping(tmp_path):
    task = cabinet_task()
    plan = RecipePlanner().plan(task)
    bundle = compile_execution_bundle(
        plan, task,
        scene_path="packages/robot_agent_control/world_model/robotsim/scene_001.xml",
        interaction_registry_path=SIDECAR,
        output_dir=tmp_path,
        route="B",
    )
    assert bundle.commands
    commands = json.loads((tmp_path / "commands.json").read_text())["commands"]
    trace = json.loads((tmp_path / "compiled_step_trace.json").read_text())
    assert len(trace) == len(plan.steps)
    assert len(commands) == len(plan.steps)
    step10 = next(item for item in trace if item["skill_step_id"] == "step-10")
    assert step10["resolved_anchor"] == "interior"
    assert any(item["parameters"].get("anchor") == "interior" for item in step10["generated_commands"])


def test_compiler_preserves_one_command_per_semantic_skill(tmp_path):
    task = GroundedTask(
        instruction="put the red ball in the upper compartment",
        task_types=[TaskType.PICK_AND_PLACE],
        entities=[
            GroundedEntity(entity_id="red_ball_01", semantic_name="red ball", object_id="red_ball", grounding_method="interaction_registry"),
            GroundedEntity(entity_id="upper_compartment_01", semantic_name="upper compartment", object_id="blue_cabinet_upper_compartment", grounding_method="interaction_registry"),
        ],
        operations=[Operation(operation_id="op-1", task_type=TaskType.PICK_AND_PLACE, source="red_ball_01", destination="upper_compartment_01")],
        spatial_relations=[SpatialRelation(scope="goal", subject="red_ball_01", relation=SpatialRelationType.INSIDE, reference="upper_compartment_01")],
        scene_id="scene_001",
    )
    plan = RecipePlanner().plan(task)
    compile_execution_bundle(
        plan, task,
        scene_path="packages/robot_agent_control/world_model/robotsim/scene_001.xml",
        interaction_registry_path=SIDECAR,
        output_dir=tmp_path,
        route="B",
    )

    commands = json.loads((tmp_path / "commands.json").read_text())["commands"]
    trace = json.loads((tmp_path / "compiled_step_trace.json").read_text())
    assert len(plan.steps) == len(commands) == len(trace) == 6
    assert [command["skill_name"] for command in commands] == [step.skill_name for step in plan.steps]
    assert all(len(item["generated_commands"]) == 1 for item in trace)
    assert not any(command["parameters"].get("target") == "home" for command in commands)
    assert not any(
        command["parameters"].get("target") == "end_effector"
        and command["parameters"].get("relation") in {"above", "behind"}
        for command in commands
    )

    pick_grasp = next(item for item in trace if item["semantic_skill"] == "grasp")
    assert [item["skill_name"] for item in pick_grasp["generated_commands"]] == ["grasp"]
    assert pick_grasp["generated_commands"][0]["parameters"] == {"target": "red_ball"}

    pick_release = next(item for item in trace if item["semantic_skill"] == "release")
    assert [item["skill_name"] for item in pick_release["generated_commands"]] == ["release"]
    assert pick_release["generated_commands"][0]["parameters"] == {
        "target": "red_ball",
        "reference": "blue_cabinet_upper_compartment",
    }
