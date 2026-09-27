from types import SimpleNamespace

import pytest

from robot_agent_sim.contracts.placement import PlacementTargetKind, PlacementTargetSpec
from robot_agent_sim.contracts.task_intent import SpatialRelationType
from robot_agent_sim.execution.placement.resolver import PlacementResolutionError, PlacementResolver
from robot_agent_sim.models.fake import FakeTaskUnderstandingProvider
from robot_agent_sim.models.task_understanding import TaskUnderstandingRequest, enrich_task
from robot_agent_sim.session.contracts import ObjectWorldState, WorldState
from robot_agent_sim.scene.registry import SceneObject, SceneRegistry


def _intent(text: str):
    parsed = FakeTaskUnderstandingProvider().understand(TaskUnderstandingRequest(instruction=text))
    return enrich_task(parsed, text)


def test_takeout_keeps_source_selection_independent_from_destination():
    intent = _intent("把盒子里的棒球拿出来放到苹果右边")
    operation = intent.operations[0]
    assert operation.source == "baseball_01"
    assert operation.destination == "apple_01"
    assert operation.placement_target.kind == PlacementTargetKind.RELATIVE_OBJECT
    assert operation.placement_target.relation == SpatialRelationType.RIGHT_OF
    assert any(
        relation.scope == "selection"
        and relation.subject == "baseball_01"
        and relation.relation == SpatialRelationType.INSIDE
        and relation.reference == "open_box_01"
        for relation in intent.spatial_relations
    )


def test_takeout_without_endpoint_uses_explicit_free_space_policy():
    intent = _intent("把盒子里的棒球拿出来")
    operation = intent.operations[0]
    assert operation.placement_target.kind == PlacementTargetKind.FREE_SPACE
    assert operation.placement_target.reference == "__table__"
    assert any(repair["type"] == "implicit_free_space_placement" for repair in intent.semantic_repairs)


def test_container_top_is_clarification_not_interior_fallback():
    intent = _intent("把苹果放到盒子上面")
    assert intent.status.value == "clarification_required"
    assert "顶部" in intent.explanation


def test_free_space_requires_a_unique_support_when_reference_is_omitted():
    registry = SceneRegistry(
        scene_id="test",
        robot="ur5e",
        objects=[
            SceneObject(object_id="apple", body_name="apple", role="target", semantic_name="apple", dimensions_m=(.06, .06, .06)),
        ],
    )
    world = WorldState(
        world_version=0,
        scene_version=1,
        turn_index=0,
        sim_time=0,
        objects={"apple": ObjectWorldState(object_id="apple", body_name="apple", position=(0, 0, 0))},
    )
    metadata = {
        "objects": {
            "table": {"category": "support_surface", "spatial": {"regions": {"support_surface": {"local_min": [-.3, -.3, 0], "local_max": [.3, .3, 0]}}}},
            "shelf": {"category": "support_surface", "spatial": {"regions": {"support_surface": {"local_min": [-.3, -.3, 0], "local_max": [.3, .3, 0]}}}},
        }
    }
    with pytest.raises(PlacementResolutionError, match="multiple support surfaces"):
        PlacementResolver().resolve(
            PlacementTargetSpec(kind=PlacementTargetKind.FREE_SPACE),
            source_object_id="apple",
            world_state=world,
            scene_registry=registry,
            interaction_registry=metadata,
            source_dimensions=(.06, .06, .06),
            world_version=0,
        )


def test_feasibility_checker_can_reject_all_geometric_candidates():
    registry = SceneRegistry(
        scene_id="test",
        robot="ur5e",
        objects=[SceneObject(object_id="apple", body_name="apple", role="target", semantic_name="apple", dimensions_m=(.06, .06, .06))],
    )
    world = WorldState(
        world_version=0,
        scene_version=1,
        turn_index=0,
        sim_time=0,
        objects={"apple": ObjectWorldState(object_id="apple", body_name="apple", position=(0, 0, 0))},
    )
    metadata = {"objects": {"table": {"category": "support_surface", "spatial": {"regions": {"support_surface": {"local_min": [-.3, -.3, 0], "local_max": [.3, .3, 0]}}}}}}
    checker = SimpleNamespace(check_placement_pose=lambda source_object_id, candidate_pose: False)
    with pytest.raises(PlacementResolutionError, match="placement_capacity_exceeded"):
        PlacementResolver().resolve(
            PlacementTargetSpec(kind=PlacementTargetKind.FREE_SPACE, reference="table"),
            source_object_id="apple",
            world_state=world,
            scene_registry=registry,
            interaction_registry=metadata,
            source_dimensions=(.06, .06, .06),
            world_version=0,
            feasibility_checker=checker,
        )
