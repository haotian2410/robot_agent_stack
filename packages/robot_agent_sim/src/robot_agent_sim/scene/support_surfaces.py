"""Shared generated-scene support-surface geometry."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SupportSurface:
    surface_id: str
    body_name: str
    position: tuple[float, float, float]
    dimensions_m: tuple[float, float, float]
    thickness_m: float = 0.03


WORK_TABLE = SupportSurface(
    surface_id="__table__",
    body_name="work_table",
    position=(0.0, 0.0, -0.6),
    dimensions_m=(0.75, 1.50, 0.03),
)

# Public tabletop-centered object placement frame.  Keep a safety margin from
# the physical table edge for object geometry and robot reachability.
WORKSPACE_X = (-0.31, 0.31)
WORKSPACE_Y = (-0.66, 0.66)


__all__ = ["SupportSurface", "WORK_TABLE", "WORKSPACE_X", "WORKSPACE_Y"]
