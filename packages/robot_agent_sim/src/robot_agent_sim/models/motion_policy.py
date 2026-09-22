"""Numeric policy for underspecified directional motion."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class MotionPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")
    default_relative_distance_m: float = Field(default=0.10, gt=0, le=2)
