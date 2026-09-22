from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class ModelCallMode(StrEnum):
    GENERATED_INITIAL = "generated_initial"
    UPLOADED_INITIAL = "uploaded_initial"
    CURRENT_SCENE = "current_scene"


@dataclass(frozen=True)
class StageGenerationConfig:
    """Per-stage decoding limits; keeping these explicit prevents an
    accidentally unbounded shared generation setting."""

    temperature: float = 0.0
    max_completion_tokens: int = 256


class ModelCallBudgetExceeded(RuntimeError):
    pass


@dataclass
class ModelCallBudget:
    max_calls: int
    allowed_stages: tuple[str, ...]
    stage_limits: dict[str, int] = field(default_factory=dict)
    calls: int = 0
    stages: list[dict] = field(default_factory=list)

    @classmethod
    def for_mode(cls, mode: ModelCallMode, planner: str = "qwen"):
        if mode == ModelCallMode.GENERATED_INITIAL:
            stages = ("task_understanding",) if planner == "recipe" else ("task_understanding", "skill_planning")
        else:
            stages = ("task_understanding", "vision_grounding") if planner == "recipe" else ("task_understanding", "vision_grounding", "skill_planning")
        return cls(len(stages), stages, {stage: 1 for stage in stages})

    @classmethod
    def for_route(cls, route_b: bool, planner: str = "qwen"):
        """Compatibility shim for callers outside the session pipeline."""
        return cls.for_mode(ModelCallMode.UPLOADED_INITIAL if route_b else ModelCallMode.GENERATED_INITIAL, planner)

    def consume(self, stage: str):
        used = sum(1 for item in self.stages if item["stage"] == stage)
        if stage not in self.allowed_stages or self.calls >= self.max_calls or (stage in self.stage_limits and used >= self.stage_limits[stage]):
            raise ModelCallBudgetExceeded(f"model call budget exceeded at {stage}")
        self.calls += 1
        self.stages.append({"stage": stage, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "finish_reason": None})

    def update(self, stage: str, record: dict | None):
        if not record: return
        item = self.stages[-1]
        for key in ("prompt_tokens", "completion_tokens"):
            value = record.get(key)
            if isinstance(value, int): item[key] = value
        if record.get("finish_reason") is not None:
            item["finish_reason"] = record["finish_reason"]
        item["total_tokens"] = item["prompt_tokens"] + item["completion_tokens"]

    def summary(self):
        return {"calls": self.calls, "prompt_tokens": sum(x["prompt_tokens"] for x in self.stages), "completion_tokens": sum(x["completion_tokens"] for x in self.stages), "total_tokens": sum(x["total_tokens"] for x in self.stages), "stages": self.stages}
