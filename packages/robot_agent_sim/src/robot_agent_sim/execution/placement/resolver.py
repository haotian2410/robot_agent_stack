from __future__ import annotations

import json
from pathlib import Path

from ...contracts.placement import PlacementTargetKind
from .container import resolve_container
from .free_space import resolve_free_space
from .relative import resolve_relative
from .surface import resolve_surface


class PlacementResolutionError(ValueError):
    pass


class PlacementResolver:
    """Dispatch semantic placement to a deterministic live-world resolver."""

    def resolve(self, placement_spec, *, source_object_id, world_state, scene_registry, interaction_registry, source_dimensions, world_version):
        spec = placement_spec
        reference_id = spec.reference or "__table__"
        metadata = interaction_registry
        if isinstance(metadata, (str, Path)):
            metadata = json.loads(Path(metadata).read_text(encoding="utf-8"))
        try:
            if spec.kind == PlacementTargetKind.CONTAINER_INTERIOR:
                return resolve_container(spec, source_id=source_object_id, reference_id=reference_id, source_dimensions=source_dimensions, world_state=world_state, registry=scene_registry, metadata=metadata, world_version=world_version)
            if spec.kind == PlacementTargetKind.SUPPORT_SURFACE:
                return resolve_surface(source_id=source_object_id, reference_id=reference_id, source_dimensions=source_dimensions, world_state=world_state, registry=scene_registry, metadata=metadata, world_version=world_version)
            if spec.kind == PlacementTargetKind.RELATIVE_OBJECT:
                return resolve_relative(source_id=source_object_id, reference_id=reference_id, relation=spec.relation, source_dimensions=source_dimensions, world_state=world_state, registry=scene_registry, world_version=world_version)
            return resolve_free_space(source_id=source_object_id, reference_id=reference_id, source_dimensions=source_dimensions, world_state=world_state, registry=scene_registry, metadata=metadata, world_version=world_version)
        except ValueError as exc:
            raise PlacementResolutionError(str(exc)) from exc
