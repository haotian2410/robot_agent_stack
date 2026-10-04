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


@dataclass(frozen=True)
class TableCameraSpec:
    """Camera placement expressed as ratios of the support surface."""
    xy_ratio: tuple[float, float] = (0.5, 0.5)
    height_ratio: float = 1.0
    fovy_deg: float = 45.0

    def world_position(self, surface: SupportSurface) -> tuple[float, float, float]:
        half_x, half_y = surface.dimensions_m[0] / 2.0, surface.dimensions_m[1] / 2.0
        x = surface.position[0] - half_x + surface.dimensions_m[0] * self.xy_ratio[0]
        y = surface.position[1] - half_y + surface.dimensions_m[1] * self.xy_ratio[1]
        z = surface.position[2] + max(surface.dimensions_m[0], surface.dimensions_m[1]) * self.height_ratio
        return (x, y, z)


WORK_TABLE = SupportSurface(
    surface_id="__table__",
    body_name="work_table",
    # Public support-contact frame: object positions at z=0 touch the table.
    position=(0.0, 0.0, 0.0),
    dimensions_m=(0.75, 1.50, 0.03),
)
TABLE_CAMERA = TableCameraSpec()

# Public tabletop-centered object placement frame.  Keep a safety margin from
# the physical table edge for object geometry and robot reachability.
WORKSPACE_X = (-0.31, 0.31)
WORKSPACE_Y = (-0.66, 0.66)


__all__ = ["SupportSurface", "TableCameraSpec", "TABLE_CAMERA", "WORK_TABLE", "WORKSPACE_X", "WORKSPACE_Y"]
