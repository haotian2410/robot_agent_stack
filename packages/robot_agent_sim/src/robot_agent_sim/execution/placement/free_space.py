from __future__ import annotations

from ...contracts.placement import PlacementTargetKind, ResolvedPlacement
from ...contracts.task_intent import SpatialRelationType
from .surface import resolve_surface
from .feasibility import PlacementFeasibilityChecker


def resolve_free_space(*, source_id, reference_id, source_dimensions, world_state, registry, metadata, world_version, feasibility_checker: PlacementFeasibilityChecker | None = None):
    resolved = resolve_surface(
        source_id=source_id,
        reference_id=reference_id,
        source_dimensions=source_dimensions,
        world_state=world_state,
        registry=registry,
        metadata=metadata,
        world_version=world_version,
        resolver_name="free_space_placement",
        feasibility_checker=feasibility_checker,
    )
    return resolved.model_copy(update={"kind": PlacementTargetKind.FREE_SPACE, "relation": None})
