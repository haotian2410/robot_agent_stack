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
)
from robot_agent_sim.models.task_understanding import ParseEntity, ParseOperation, ParseRelation, TaskParseLLMOutput, enrich_task
from robot_agent_sim.models.motion_policy import MotionPolicy
from robot_agent_sim.pipeline.engine import PipelineEngine


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
