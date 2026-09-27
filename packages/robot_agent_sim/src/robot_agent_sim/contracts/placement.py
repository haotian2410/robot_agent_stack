"""Semantic placement targets shared by understanding, planning and execution."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .spatial import SpatialRelationType


class PlacementTargetKind(StrEnum):
    CONTAINER_INTERIOR = "container_interior"
    SUPPORT_SURFACE = "support_surface"
    RELATIVE_OBJECT = "relative_object"
    FREE_SPACE = "free_space"


class PlacementTargetSpec(BaseModel):
    """Semantic destination only; it intentionally contains no coordinates."""

    model_config = ConfigDict(extra="forbid")

    kind: PlacementTargetKind
    reference: str | None = None
    relation: SpatialRelationType | None = None
    support: str | None = None

    @model_validator(mode="after")
    def validate_shape(self):
        if self.kind in {PlacementTargetKind.CONTAINER_INTERIOR, PlacementTargetKind.SUPPORT_SURFACE} and not self.reference:
            raise ValueError(f"placement_target {self.kind.value} requires reference")
        if self.kind == PlacementTargetKind.RELATIVE_OBJECT:
            if not self.reference:
                raise ValueError("relative_object placement requires reference")
            if self.relation not in {
                SpatialRelationType.LEFT_OF, SpatialRelationType.RIGHT_OF,
                SpatialRelationType.FRONT_OF, SpatialRelationType.BEHIND,
                SpatialRelationType.NEAR, SpatialRelationType.ABOVE,
                SpatialRelationType.BELOW,
            }:
                raise ValueError("relative_object placement requires a supported relation")
        if self.kind == PlacementTargetKind.CONTAINER_INTERIOR and self.relation not in {None, SpatialRelationType.INSIDE}:
            raise ValueError("container_interior placement relation must be inside or null")
        if self.kind == PlacementTargetKind.SUPPORT_SURFACE and self.relation not in {None, SpatialRelationType.ON}:
            raise ValueError("support_surface placement relation must be on or null")
        if self.kind == PlacementTargetKind.FREE_SPACE and self.relation is not None:
            raise ValueError("free_space placement does not accept a relation")
        return self


class ResolvedPlacement(BaseModel):
    """Python-owned physical placement result, persisted per concrete subtask."""

    model_config = ConfigDict(extra="forbid")

    kind: PlacementTargetKind
    source_object_id: str
    reference_object_id: str | None = None
    support_object_id: str | None = None
    relation: SpatialRelationType | None = None
    host_object_id: str
    anchor_name: str = "resolved_placement"
    local_position: tuple[float, float, float] | None = None
    world_position: tuple[float, float, float]
    world_version: int
    resolver: str
    candidate_count: int = Field(default=1, ge=1)
    chosen_candidate: str | None = None


__all__ = ["PlacementTargetKind", "PlacementTargetSpec", "ResolvedPlacement"]
