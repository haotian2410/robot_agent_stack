"""Numeric policy for underspecified directional motion."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from ..contracts.task_intent import MotionScale


class MotionPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")
    default_relative_distance_m: float = Field(default=0.10, gt=0, le=2)
    small_scale_factor: float = Field(default=0.10, gt=0, le=10)
    medium_scale_factor: float = Field(default=0.50, gt=0, le=10)
    large_scale_factor: float = Field(default=2.00, gt=0, le=10)

    def scale_factor(self, scale: MotionScale) -> float:
        return {
            MotionScale.SMALL: self.small_scale_factor,
            MotionScale.MEDIUM: self.medium_scale_factor,
            MotionScale.LARGE: self.large_scale_factor,
        }[scale]
