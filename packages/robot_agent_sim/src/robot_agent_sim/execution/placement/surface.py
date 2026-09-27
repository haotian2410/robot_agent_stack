from __future__ import annotations

from ...contracts.placement import PlacementTargetKind, ResolvedPlacement
from ...contracts.task_intent import SpatialRelationType
from ...scene.support_surfaces import WORKSPACE_X, WORKSPACE_Y
from .geometry import add, aabb, overlap, quaternion_rotate


def resolve_surface(*, source_id, reference_id, source_dimensions, world_state, registry, metadata, world_version, resolver_name="surface_placement"):
    target = (metadata.get("objects", {}).get(reference_id) or {}) if metadata else {}
    spatial = target.get("spatial", {})
    region = (spatial.get("regions", {}) or {}).get("support_surface") or {
        "local_min": [WORKSPACE_X[0], WORKSPACE_Y[0], 0.0],
        "local_max": [WORKSPACE_X[1], WORKSPACE_Y[1], 0.0],
    }
    host = world_state.objects.get(reference_id)
    if reference_id == "__table__":
        # The synthetic table's public interaction frame is the workspace
        # origin.  The physical UR5e base XML may place its visual table body
        # below that frame (for example at z=-0.6); using that body pose here
        # would shift every semantic "on the table" target after turn one.
        host_position = (0.0, 0.0, 0.0)
        host_quaternion = (1.0, 0.0, 0.0, 0.0)
    elif host is None:
        host_position = (0.0, 0.0, 0.0)
        host_quaternion = (1.0, 0.0, 0.0, 0.0)
    else:
        host_position = host.position
        host_quaternion = host.quaternion
    low = tuple(float(value) for value in region["local_min"])
    high = tuple(float(value) for value in region["local_max"])
    step = max(min(float(source_dimensions[0]), float(source_dimensions[1])) + 0.01, 0.08)
    occupied = []
    for item in registry.objects:
        if item.object_id in {source_id, reference_id} or item.object_id not in world_state.objects:
            continue
        state = world_state.objects[item.object_id]
        relative = tuple(state.position[index] - host_position[index] for index in range(3))
        dims = item.dimensions_m or (0.06, 0.06, 0.06)
        if low[0] - dims[0] <= relative[0] <= high[0] + dims[0] and low[1] - dims[1] <= relative[1] <= high[1] + dims[1]:
            occupied.append(aabb(relative, dims))
    candidates = []
    x = low[0] + source_dimensions[0] / 2
    while x <= high[0] - source_dimensions[0] / 2 + 1e-9:
        y = low[1] + source_dimensions[1] / 2
        while y <= high[1] - source_dimensions[1] / 2 + 1e-9:
            # Scene object poses are contact/body poses (mesh assets are
            # translated so their lowest vertex is local z=0).  The physical
            # placement therefore uses the support plane itself; the live
            # session creates a separate elevated end-effector anchor.
            local = (round(x, 6), round(y, 6), round(max(0.0, high[2]), 6))
            if not any(overlap(aabb(local, source_dimensions), box) for box in occupied):
                candidates.append(local)
            y += step
        x += step
    if not candidates:
        raise ValueError(f"placement_capacity_exceeded: {reference_id}")
    local = candidates[0]
    return ResolvedPlacement(
        kind=PlacementTargetKind.SUPPORT_SURFACE,
        source_object_id=source_id,
        support_object_id=reference_id,
        host_object_id=reference_id,
        relation=SpatialRelationType.ON,
        local_position=local,
        world_position=add(host_position, quaternion_rotate(local, host_quaternion)),
        world_version=world_version,
        resolver=resolver_name,
        candidate_count=len(candidates),
        chosen_candidate="surface_grid_1",
    )
