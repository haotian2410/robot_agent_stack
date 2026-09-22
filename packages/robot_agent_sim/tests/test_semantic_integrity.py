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
from robot_agent_sim.pipeline.engine import PipelineEngine
from robot_agent_sim.grounding.world_relation import RelationAmbiguous, WorldRelationResolver
from robot_agent_sim.grounding.name_matching import exact_name_match


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


def test_candidate_pool_quantity_remains_supported():
    request = type("Request", (), {"instruction": "把三个苹果中靠近篮子的苹果放进篮子"})()
    parsed = FakeTaskUnderstandingProvider().understand(request)
    assert parsed.entities[0].quantity_mode == QuantityMode.CANDIDATE_POOL
    assert enrich_task(parsed, request.instruction).status == TaskStatus.ACCEPTED


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


def test_name_matching_does_not_use_dangerous_substrings():
    assert exact_name_match("apple", "apple_01")
    assert not exact_name_match("apple", "pineapple")
    assert not exact_name_match("ball", "baseball")


def test_model_output_must_be_one_json_object():
    assert _extract_json('{"ok":true}') == '{"ok": true}'
    assert _extract_json('```json\n{"ok":true}\n```') == '{"ok": true}'
    with pytest.raises(QwenProviderError):
        _extract_json('说明文字 {"example":true} 最终结果 {"ok":true}')
