"""Shared candidate evidence used by all grounding providers."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class GroundingCandidate(BaseModel):
    """A provider observation, never a final semantic binding."""

    model_config = ConfigDict(extra="forbid")
    object_id: str
    sources: set[str] = Field(default_factory=set)
    body_name: str | None = None
    world_position: tuple[float, float, float] | None = None
    detection_bbox: tuple[int, int, int, int] | None = None
    instance_bbox: tuple[int, int, int, int] | None = None
    bbox_iou: float | None = None


def merge_candidate(values: list[dict], candidate: GroundingCandidate) -> None:
    """Merge provider evidence for one object without duplicating candidates."""
    for value in values:
        if value.get("object_id") != candidate.object_id:
            continue
        value.setdefault("sources", [])
        value["sources"] = sorted(set(value["sources"]) | candidate.sources)
        for key in ("body_name", "world_position", "detection_bbox", "instance_bbox", "bbox_iou"):
            if getattr(candidate, key) is not None:
                value[key] = getattr(candidate, key)
        return
    values.append(candidate.model_dump(mode="json"))


__all__ = ["GroundingCandidate", "merge_candidate"]
