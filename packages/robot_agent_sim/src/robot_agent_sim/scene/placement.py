"""Shared deterministic placement helpers for composition and scene patches."""

from __future__ import annotations

from .composer import MIN_GAP_M, WORKSPACE_X, WORKSPACE_Y, _overlap


class PlacementSolver:
    def place_relative(self, relation: str, reference_position, subject_dimensions, reference_dimensions, existing) -> tuple[float, float, float]:
        x, y, z = (float(v) for v in reference_position)
        gap = MIN_GAP_M + 0.02
        if relation in {"right", "right_of"}:
            x += reference_dimensions[0] / 2 + subject_dimensions[0] / 2 + gap
        elif relation in {"left", "left_of"}:
            x -= reference_dimensions[0] / 2 + subject_dimensions[0] / 2 + gap
        elif relation in {"front", "front_of"}:
            y += reference_dimensions[1] / 2 + subject_dimensions[1] / 2 + gap
        elif relation in {"back", "behind"}:
            y -= reference_dimensions[1] / 2 + subject_dimensions[1] / 2 + gap
        elif relation in {"above", "up"}:
            z += reference_dimensions[2] + gap
        x_limit = WORKSPACE_X
        candidate = (
            round(max(x_limit[0] + subject_dimensions[0] / 2, min(x_limit[1] - subject_dimensions[0] / 2, x)), 6),
            round(max(WORKSPACE_Y[0] + subject_dimensions[1] / 2, min(WORKSPACE_Y[1] - subject_dimensions[1] / 2, y)), 6),
            round(z, 6),
        )
        if any(_overlap(candidate, subject_dimensions, item.position, item.dimensions_m or (0.05, 0.05, 0.05)) for item in existing):
            # Keep the requested side while trying small orthogonal offsets.
            # Prefer the robot-facing half of the table (negative y) so a
            # long mesh such as a banana remains reachable.
            for offset in (-0.12, -0.24, -0.36, 0.12, 0.24, 0.36):
                trial = (candidate[0], round(candidate[1] + offset, 6), candidate[2]) if relation in {"right", "right_of", "left", "left_of"} else (round(candidate[0] + offset, 6), candidate[1], candidate[2])
                trial = (trial[0], max(WORKSPACE_Y[0] + subject_dimensions[1] / 2, min(WORKSPACE_Y[1] - subject_dimensions[1] / 2, trial[1])), trial[2])
                if all(not _overlap(trial, subject_dimensions, item.position, item.dimensions_m or (0.05, 0.05, 0.05)) for item in existing):
                    return trial
            raise ValueError("unable to place scene patch without overlap")
        return candidate
