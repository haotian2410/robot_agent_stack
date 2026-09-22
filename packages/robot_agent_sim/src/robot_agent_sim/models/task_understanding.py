"""Minimal task-parser contract and deterministic internal enrichment."""
from __future__ import annotations

import re
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..contracts.task_intent import Operation, SpatialRelation, SpatialRelationType, TaskEntity, TaskIntent, TaskType
from ..contracts.turn import SceneEditIntent, SceneQueryIntent, TurnKind


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ParseEntity(StrictModel):
    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    category: str = Field(min_length=1)
    color: str | None = None
    count: int = Field(default=1, ge=1, le=100)


class ParseOperation(StrictModel):
    type: Literal[
        "locate",
        "search",
        "move",
        "grasp",
        "release",
        "pick_and_place",
        "press",
        "open",
        "close",
    ]
    source: str | None = None
    destination: str | None = None
    target: str | None = None
    reference: str | None = None
    motion_direction: str | None = None
    distance_m: float | None = Field(default=None, gt=0, le=2)


class ParseRelation(StrictModel):
    scope: Literal["scene", "selection", "goal"] = "selection"
    subject: str
    relation: SpatialRelationType
    reference: str | None = None


class TaskParseLLMOutput(StrictModel):
    status: Literal["accepted", "unsupported_task", "direction_clarification_required"]
    turn_kind: TurnKind = TurnKind.ROBOT_TASK
    scene_edit: SceneEditIntent | None = None
    scene_query: SceneQueryIntent | None = None
    entities: list[ParseEntity] = Field(default_factory=list)
    operations: list[ParseOperation] = Field(default_factory=list)
    relations: list[ParseRelation] = Field(default_factory=list)
    raw_direction: str | None = None
    distance_m: float | None = Field(default=None, gt=0, le=2)
    raw_task: str | None = None

    @model_validator(mode="after")
    def valid_status(self):
        if self.status == "accepted" and self.turn_kind == TurnKind.ROBOT_TASK and (not self.entities or not self.operations):
            raise ValueError("accepted parse requires entities and operations")
        if self.status == "accepted" and self.turn_kind == TurnKind.SCENE_EDIT and self.scene_edit is None:
            raise ValueError("scene_edit turn requires scene_edit intent")
        if self.status == "accepted" and self.turn_kind == TurnKind.SCENE_QUERY and self.scene_query is None:
            raise ValueError("scene_query turn requires scene_query intent")
        if self.turn_kind != TurnKind.ROBOT_TASK and (self.entities or self.operations or self.relations):
            raise ValueError("non-robot turn must not contain a robot plan")
        if self.turn_kind != TurnKind.SCENE_EDIT and self.scene_edit is not None:
            raise ValueError("scene_edit intent is only valid for scene_edit turns")
        if self.turn_kind != TurnKind.SCENE_QUERY and self.scene_query is not None:
            raise ValueError("scene_query intent is only valid for scene_query turns")
        if self.status != "accepted" and (self.entities or self.operations or self.relations):
            raise ValueError("rejected parse must not include a plan")
        if self.status == "accepted" and self.raw_direction is not None and self.raw_direction not in {"left", "right", "front", "back", "up", "down"}:
            raise ValueError("accepted motion direction must be one of left/right/front/back/up/down")
        return self


class TaskUnderstandingRequest(StrictModel):
    instruction: str = Field(min_length=1, max_length=2000)


class TaskUnderstandingProvider(Protocol):
    def understand(self, request: TaskUnderstandingRequest) -> TaskParseLLMOutput: ...


_EXPLICIT_MOTION_DIRECTIONS = (
    (r"(?:向|往|朝)\s*左\s*(?:移(?:动)?|挪)|左移", "left"),
    (r"(?:向|往|朝)\s*右\s*(?:移(?:动)?|挪)|右移", "right"),
    (r"(?:向|往|朝)\s*前\s*(?:移(?:动)?|挪)|前移", "front"),
    (r"(?:向|往|朝)\s*后\s*(?:移(?:动)?|挪)|后移", "back"),
    (r"(?:向|往|朝)\s*上\s*(?:移(?:动)?|挪)|上移", "up"),
    (r"(?:向|往|朝)\s*下\s*(?:移(?:动)?|挪)|下移", "down"),
)


def _instruction_motion(instruction: str) -> tuple[str | None, float | None]:
    """Recover an explicit directional displacement from the user's text.

    Local language models occasionally classify “把苹果往右移动一点” as a
    plain grasp and omit ``raw_direction``.  Directional displacement is a
    small, closed grammar, so normalize that part deterministically instead of
    allowing a structurally valid but incomplete model response to execute.
    The patterns deliberately require a movement verb, avoiding selectors such
    as “右边的苹果”.
    """

    direction = next(
        (value for pattern, value in _EXPLICIT_MOTION_DIRECTIONS if re.search(pattern, instruction)),
        None,
    )
    if direction is None:
        return None, None
    match = re.search(r"(\d+(?:\.\d+)?)\s*(厘米|cm|米|m)", instruction, re.IGNORECASE)
    if match is None:
        return direction, 0.10
    distance = float(match.group(1))
    if match.group(2).casefold() in {"厘米", "cm"}:
        distance /= 100.0
    return direction, distance


def enrich_task(parsed: TaskParseLLMOutput, instruction: str) -> TaskIntent:
    text_direction, text_distance = _instruction_motion(instruction)
    has_model_move = any(operation.type == "move" for operation in parsed.operations)
    operations = []
    for index, op in enumerate(parsed.operations):
        # A single grasp/locate result for an explicit displacement is a common
        # Qwen schema-level under-parse.  Treat the displacement as the semantic
        # operation; its deterministic recipe already includes grasp/release.
        promote_to_move = (
            text_direction is not None
            and not has_model_move
            and len(parsed.operations) == 1
            and op.type in {"grasp", "locate"}
        )
        operation_type = "move" if promote_to_move else op.type
        direction = op.motion_direction or (
            parsed.raw_direction if operation_type == "move" else None
        ) or (text_direction if operation_type == "move" else None)
        distance_m = op.distance_m or parsed.distance_m or (
            text_distance if operation_type == "move" else None
        )
        if distance_m is None and operation_type == "move" and direction:
            distance_m = 0.10
        target = op.target
        source = op.source
        if promote_to_move:
            target = target or source or parsed.entities[0].id
            source = None
        operations.append(Operation(
            operation_id=f"op-{index + 1}", task_type=operation_type, source=source,
            destination=op.destination, target=target, reference=op.reference,
            description=operation_type.replace("_", " "),
            depends_on=[f"op-{index}"] if index else [],
            motion_direction=direction,
            distance_m=distance_m,
        ))
    task_types = list(dict.fromkeys(operation.task_type for operation in operations))
    if len(task_types) > 1:
        task_types.append(TaskType.MIXED)
    messages = {
        "direction_clarification_required": "当前仅支持上下、左右、前后，请明确选择其中一个方向。",
        "unsupported_task": "当前不支持该任务类型。",
    }
    return TaskIntent(
        status=parsed.status, instruction=instruction, task_types=task_types,
        entities=[TaskEntity(entity_id=e.id, semantic_name=e.name, category=e.category, color=e.color, count=e.count) for e in parsed.entities],
        operations=operations,
        spatial_relations=[SpatialRelation(scope=r.scope, subject=r.subject, relation=r.relation, reference=r.reference) for r in parsed.relations],
        raw_direction=(parsed.raw_direction or text_direction) if parsed.status == "accepted" else None,
        explanation=messages.get(parsed.status, ""),
    )
