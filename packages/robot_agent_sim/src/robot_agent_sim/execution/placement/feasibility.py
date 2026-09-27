"""Optional final-stage placement feasibility hook.

Placement resolvers own semantic geometry.  They must not duplicate the
control package's IK/collision/path planner, so callers may inject a checker
that reuses those services to reject an otherwise geometric candidate.
"""

from __future__ import annotations

from typing import Any, Protocol


class PlacementFeasibilityChecker(Protocol):
    """Check a concrete candidate pose with the existing motion stack."""

    def check_placement_pose(self, source_object_id: str, candidate_pose: Any) -> bool:
        """Return true only when IK, collision and path checks all succeed."""


def is_feasible(checker: PlacementFeasibilityChecker | None, source_object_id: str, candidate_pose: Any) -> bool:
    """Normalize boolean or structured checker results for resolver use."""
    if checker is None:
        return True
    result = checker.check_placement_pose(source_object_id, candidate_pose)
    if isinstance(result, dict):
        return bool(result.get("feasible", result.get("success", False)))
    return bool(result)


__all__ = ["PlacementFeasibilityChecker", "is_feasible"]
