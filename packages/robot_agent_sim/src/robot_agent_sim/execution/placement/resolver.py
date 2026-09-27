from __future__ import annotations

import json
from pathlib import Path

from ...contracts.placement import PlacementTargetKind
from .container import resolve_container
from .free_space import resolve_free_space
from .relative import resolve_relative
from .surface import resolve_surface
from .feasibility import PlacementFeasibilityChecker


class PlacementResolutionError(ValueError):
    pass


class PlacementResolver:
    """Dispatch semantic placement to a deterministic live-world resolver."""

    def resolve(self, placement_spec, *, source_object_id, world_state, scene_registry, interaction_registry, source_dimensions, world_version, feasibility_checker: PlacementFeasibilityChecker | None = None):
        spec = placement_spec
        metadata = interaction_registry
        if isinstance(metadata, (str, Path)):
            metadata = json.loads(Path(metadata).read_text(encoding="utf-8"))
        reference_id = self._resolve_support_reference(spec, metadata)
        try:
            if spec.kind == PlacementTargetKind.CONTAINER_INTERIOR:
                return resolve_container(spec, source_id=source_object_id, reference_id=reference_id, source_dimensions=source_dimensions, world_state=world_state, registry=scene_registry, metadata=metadata, world_version=world_version, feasibility_checker=feasibility_checker)
            if spec.kind == PlacementTargetKind.SUPPORT_SURFACE:
                return resolve_surface(source_id=source_object_id, reference_id=reference_id, source_dimensions=source_dimensions, world_state=world_state, registry=scene_registry, metadata=metadata, world_version=world_version, feasibility_checker=feasibility_checker)
            if spec.kind == PlacementTargetKind.RELATIVE_OBJECT:
                return resolve_relative(source_id=source_object_id, reference_id=reference_id, relation=spec.relation, source_dimensions=source_dimensions, world_state=world_state, registry=scene_registry, world_version=world_version, feasibility_checker=feasibility_checker)
            return resolve_free_space(source_id=source_object_id, reference_id=reference_id, source_dimensions=source_dimensions, world_state=world_state, registry=scene_registry, metadata=metadata, world_version=world_version, feasibility_checker=feasibility_checker)
        except ValueError as exc:
            raise PlacementResolutionError(str(exc)) from exc

    @staticmethod
    def _resolve_support_reference(spec, metadata) -> str:
        """Resolve an omitted free-space support without choosing a random anchor.

        A scene may expose more than one legitimate support surface.  In that
        case an omitted support is a language ambiguity and must be surfaced to
        the user; only an explicitly named support (or a scene with exactly
        one primary support) is executable.
        """
        explicit = getattr(spec, "reference", None) or getattr(spec, "support", None)
        if explicit:
            return str(explicit)
        objects = (metadata or {}).get("objects", {}) if isinstance(metadata, dict) else {}
        support_ids = []
        for object_id, item in objects.items():
            if not isinstance(item, dict):
                continue
            category = str(item.get("category", "")).casefold()
            spatial = item.get("spatial", {}) if isinstance(item.get("spatial", {}), dict) else {}
            regions = spatial.get("regions", {}) if isinstance(spatial.get("regions", {}), dict) else {}
            if category in {"support_surface", "surface", "table", "shelf"} or "support_surface" in regions:
                support_ids.append(str(object_id))
        support_ids = sorted(set(support_ids))
        if len(support_ids) == 1:
            return support_ids[0]
        if not support_ids:
            raise PlacementResolutionError("placement_clarification_required: no support surface is available")
        raise PlacementResolutionError(
            "placement_clarification_required: multiple support surfaces are available "
            f"({', '.join(support_ids)}); specify the table or shelf"
        )
