from __future__ import annotations

from ...contracts.placement import PlacementTargetKind, ResolvedPlacement
from ...contracts.task_intent import SpatialRelationType
from .geometry import aabb, overlap
from .feasibility import PlacementFeasibilityChecker, is_feasible


def resolve_relative(*, source_id, reference_id, relation, source_dimensions, world_state, registry, world_version, feasibility_checker: PlacementFeasibilityChecker | None = None):
    reference_state = world_state.objects.get(reference_id)
    if reference_state is None:
        raise ValueError(f"placement_reference_missing: {reference_id}")
    source_item = next((item for item in registry.objects if item.object_id == source_id), None)
    reference_item = next((item for item in registry.objects if item.object_id == reference_id), None)
    reference_dimensions = reference_item.dimensions_m if reference_item else (0.06, 0.06, 0.06)
    gap = 0.01
    offsets = {
        SpatialRelationType.RIGHT_OF: (reference_dimensions[0] / 2 + source_dimensions[0] / 2 + gap, 0.0, 0.0),
        SpatialRelationType.LEFT_OF: (-(reference_dimensions[0] / 2 + source_dimensions[0] / 2 + gap), 0.0, 0.0),
        SpatialRelationType.FRONT_OF: (0.0, reference_dimensions[1] / 2 + source_dimensions[1] / 2 + gap, 0.0),
        SpatialRelationType.BEHIND: (0.0, -(reference_dimensions[1] / 2 + source_dimensions[1] / 2 + gap), 0.0),
        SpatialRelationType.ABOVE: (0.0, 0.0, reference_dimensions[2] / 2 + source_dimensions[2] / 2 + gap),
        SpatialRelationType.BELOW: (0.0, 0.0, -(reference_dimensions[2] / 2 + source_dimensions[2] / 2 + gap)),
    }
    if relation == SpatialRelationType.NEAR:
        offsets_list = [offsets[relation] for relation in (SpatialRelationType.RIGHT_OF, SpatialRelationType.LEFT_OF, SpatialRelationType.FRONT_OF, SpatialRelationType.BEHIND)]
    else:
        offsets_list = [offsets[relation]]
    candidates = []
    for offset in offsets_list:
        position = tuple(reference_state.position[index] + offset[index] for index in range(3))
        if relation == SpatialRelationType.BELOW and position[2] < source_dimensions[2] / 2:
            continue
        candidate_box = aabb(position, source_dimensions)
        if any(
            item.object_id not in {source_id, reference_id}
            and item.object_id in world_state.objects
            and overlap(candidate_box, aabb(world_state.objects[item.object_id].position, item.dimensions_m or (0.06, 0.06, 0.06)))
            for item in registry.objects
        ):
            continue
        if not is_feasible(feasibility_checker, source_id, position):
            continue
        candidates.append((position, relation.value if relation != SpatialRelationType.NEAR else "near_candidate"))
    if not candidates:
        raise ValueError(f"placement_no_feasible_candidate: {relation.value}")
    position, chosen = candidates[0]
    return ResolvedPlacement(
        kind=PlacementTargetKind.RELATIVE_OBJECT,
        source_object_id=source_id,
        reference_object_id=reference_id,
        host_object_id=reference_id,
        relation=relation,
        world_position=position,
        world_version=world_version,
        resolver="relative_placement",
        candidate_count=len(candidates),
        chosen_candidate=chosen,
    )
