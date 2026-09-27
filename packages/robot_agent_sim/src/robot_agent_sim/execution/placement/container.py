from __future__ import annotations

from ...contracts.placement import PlacementTargetSpec, PlacementTargetKind, ResolvedPlacement
from ...contracts.task_intent import SpatialRelationType
from ..placement_allocator import allocate_interior_slots
from .geometry import add, quaternion_rotate, quaternion_inverse_rotate
from .feasibility import PlacementFeasibilityChecker, is_feasible


def resolve_container(spec: PlacementTargetSpec, *, source_id, reference_id, source_dimensions, world_state, registry, metadata, world_version, feasibility_checker: PlacementFeasibilityChecker | None = None):
    target = (metadata.get("objects", {}).get(reference_id) or {}) if metadata else {}
    spatial = target.get("spatial", {})
    region = (spatial.get("regions", {}) or {}).get("interior")
    anchor = (spatial.get("anchors", {}) or {}).get("interior")
    if not isinstance(region, dict) or not isinstance(anchor, dict):
        raise ValueError(f"placement_region_metadata_required: {reference_id}")
    local_min = tuple(float(value) for value in region.get("local_min", ()))
    local_max = tuple(float(value) for value in region.get("local_max", ()))
    if len(local_min) != 3 or len(local_max) != 3:
        raise ValueError(f"placement_region_metadata_required: {reference_id}")
    container_state = world_state.objects.get(reference_id)
    if container_state is None:
        raise ValueError(f"placement_reference_missing: {reference_id}")
    occupied = []
    container_item = next((item for item in registry.objects if item.object_id == reference_id), None)
    for item in registry.objects:
        if item.object_id in {reference_id, source_id} or item.object_id not in world_state.objects:
            continue
        state = world_state.objects[item.object_id]
        relative_world = tuple(state.position[index] - container_state.position[index] for index in range(3))
        relative = quaternion_inverse_rotate(relative_world, container_state.quaternion)
        dims = item.dimensions_m or (0.06, 0.06, 0.06)
        if (local_min[0] - dims[0] / 2 <= relative[0] <= local_max[0] + dims[0] / 2
                and local_min[1] - dims[1] / 2 <= relative[1] <= local_max[1] + dims[1] / 2
                and local_min[2] - dims[2] / 2 <= relative[2] <= local_max[2] + dims[2] / 2):
            occupied.append(((relative[0] - dims[0] / 2, relative[0] + dims[0] / 2),
                             (relative[1] - dims[1] / 2, relative[1] + dims[1] / 2)))
    policy = spatial.get("placement_policy", {}) if isinstance(spatial, dict) else {}
    max_overhang = float(policy.get("max_overhang_m", 0.0) or 0.0)
    slots = allocate_interior_slots(
        local_min,
        local_max,
        source_dimensions,
        occupied=occupied,
        max_overhang=max_overhang,
    )
    if not slots:
        raise ValueError(f"placement_capacity_exceeded: {reference_id}")
    feasible_slots = []
    for candidate in slots:
        candidate_local = (candidate.local_position[0], candidate.local_position[1], local_min[2])
        candidate_world = add(container_state.position, quaternion_rotate(candidate_local, container_state.quaternion))
        if is_feasible(feasibility_checker, source_id, candidate_world):
            feasible_slots.append(candidate)
    if not feasible_slots:
        raise ValueError(f"placement_no_feasible_candidate: {reference_id}")
    slot = feasible_slots[0]
    # The allocator stores a centre-height for geometric packing, while the
    # generated MuJoCo body frame is translated to the object's contact
    # (lowest-vertex) plane.  Persist the physical body pose in that frame;
    # the session will use the authored interior approach height for the
    # gripper anchor.
    local = (slot.local_position[0], slot.local_position[1], local_min[2])
    world = add(container_state.position, quaternion_rotate(local, container_state.quaternion))
    return ResolvedPlacement(
        kind=PlacementTargetKind.CONTAINER_INTERIOR,
        source_object_id=source_id,
        reference_object_id=reference_id,
        host_object_id=reference_id,
        relation=SpatialRelationType.INSIDE,
        local_position=local,
        world_position=world,
        world_version=world_version,
        resolver="container_placement",
        candidate_count=len(slots),
        chosen_candidate=slot.slot_id,
    )
