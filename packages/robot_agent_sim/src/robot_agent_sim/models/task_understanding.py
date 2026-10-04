"""Minimal task-parser contract and deterministic internal enrichment."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..contracts.task_intent import Direction, MotionScale, Operation, QuantityMode, SpatialRelation, SpatialRelationType, TaskEntity, TaskIntent, TaskType, TaskStatus
from ..contracts.turn import SceneEditIntent, SceneQueryIntent, SessionControlIntent, TurnKind
from ..contracts.placement import PlacementTargetSpec
from .motion_policy import MotionPolicy
from ..semantics.evidence import extract_semantic_evidence
from ..semantics.entity_normalizer import same_semantic_role
from ..semantics.operation_normalizer import collapse_duplicate_operations


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ParseEntity(StrictModel):
    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    category: str = Field(min_length=1)
    dialogue_ref: bool = False
    dialogue_ref_set: bool = False
    color: str | None = None
    count: int = Field(default=1, ge=1, le=100)
    quantity_mode: QuantityMode = QuantityMode.SINGLE


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
    motion_scale: MotionScale | None = None
    placement_target: PlacementTargetSpec | None = None


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
    session_control: SessionControlIntent | None = None
    entities: list[ParseEntity] = Field(default_factory=list)
    operations: list[ParseOperation] = Field(default_factory=list)
    relations: list[ParseRelation] = Field(default_factory=list)
    raw_direction: str | None = None
    distance_m: float | None = Field(default=None, gt=0, le=2)
    raw_task: str | None = None

    @model_validator(mode="after")
    def valid_status(self):
        if self.status == "accepted" and self.turn_kind == TurnKind.ROBOT_TASK and not self.entities:
            raise ValueError("accepted parse requires entities")
        if self.status == "accepted" and self.turn_kind == TurnKind.SCENE_EDIT and self.scene_edit is None:
            raise ValueError("scene_edit turn requires scene_edit intent")
        if self.status == "accepted" and self.turn_kind == TurnKind.SCENE_QUERY and self.scene_query is None:
            raise ValueError("scene_query turn requires scene_query intent")
        if self.status == "accepted" and self.turn_kind == TurnKind.SESSION_CONTROL and self.session_control is None:
            raise ValueError("session_control turn requires session_control intent")
        if self.turn_kind != TurnKind.ROBOT_TASK and (self.entities or self.operations or self.relations):
            raise ValueError("non-robot turn must not contain a robot plan")
        if self.turn_kind != TurnKind.SCENE_EDIT and self.scene_edit is not None:
            raise ValueError("scene_edit intent is only valid for scene_edit turns")
        if self.turn_kind != TurnKind.SCENE_QUERY and self.scene_query is not None:
            raise ValueError("scene_query intent is only valid for scene_query turns")
        if self.turn_kind != TurnKind.SESSION_CONTROL and self.session_control is not None:
            raise ValueError("session_control intent is only valid for session_control turns")
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

_DISTANCE_PATTERN = re.compile(r"(\d+(?:\.\d+)?|[零一二两三四五六七八九十百]+)\s*(厘米|cm|米|m)", re.IGNORECASE)


def _parse_distance_number(value: str) -> float:
    if value.isdigit() or re.fullmatch(r"\d+(?:\.\d+)?", value):
        return float(value)
    digits = {"零": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4,
              "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
    if value == "十":
        return 10.0
    if "百" in value:
        left, right = value.split("百", 1)
        return float((digits.get(left, 1) if left else 1) * 100 + _parse_distance_number(right) if right else (digits.get(left, 1) if left else 1) * 100)
    if "十" in value:
        left, right = value.split("十", 1)
        tens = digits.get(left, 1) if left else 1
        return float(tens * 10 + (digits.get(right, 0) if right else 0))
    return float(digits[value])


@dataclass(frozen=True)
class ExplicitMotionSpan:
    direction: Direction
    distance_m: float | None
    motion_scale: MotionScale | None
    start: int
    end: int


def _extract_explicit_motion_spans(instruction: str) -> list[ExplicitMotionSpan]:
    """Return every explicit directional motion phrase in text order."""
    matches: list[tuple[int, int, Direction]] = []
    for pattern, direction in _EXPLICIT_MOTION_DIRECTIONS:
        for match in re.finditer(pattern, instruction):
            matches.append((match.start(), match.end(), Direction(direction)))
    matches.sort(key=lambda item: (item[0], -(item[1] - item[0])))
    unique: list[tuple[int, int, Direction]] = []
    for item in matches:
        if any(item[0] < end and item[1] > start for start, end, _ in unique):
            continue
        unique.append(item)
    unique.sort(key=lambda item: item[0])

    spans = []
    for index, (start, end, direction) in enumerate(unique):
        next_start = unique[index + 1][0] if index + 1 < len(unique) else len(instruction)
        window_end = min(next_start, end + 20)
        distance_match = _DISTANCE_PATTERN.search(instruction, end, window_end)
        distance_m = None
        if distance_match is not None:
            distance_m = _parse_distance_number(distance_match.group(1))
            if distance_match.group(2).casefold() in {"厘米", "cm"}:
                distance_m /= 100.0
        phrase = instruction[end:window_end]
        if distance_m is not None:
            motion_scale = None
        elif any(token in phrase for token in ("大幅", "很多", "很远")):
            motion_scale = MotionScale.LARGE
        elif any(token in phrase for token in ("一些", "一段", "适中", "中等", "不少", "远一些")):
            motion_scale = MotionScale.MEDIUM
        else:
            motion_scale = MotionScale.SMALL
        spans.append(ExplicitMotionSpan(
            direction=direction,
            distance_m=distance_m,
            motion_scale=motion_scale,
            start=start,
            end=distance_match.end() if distance_match is not None else end,
        ))
    return spans


def _instruction_motion(instruction: str) -> tuple[str | None, float | None, MotionScale | None]:
    """Recover an explicit directional displacement from the user's text.

    Local language models occasionally classify “把苹果往右移动一点” as a
    plain grasp and omit ``raw_direction``.  Directional displacement is a
    small, closed grammar, so normalize that part deterministically instead of
    allowing a structurally valid but incomplete model response to execute.
    The patterns deliberately require a movement verb, avoiding selectors such
    as “右边的苹果”.
    """

    spans = _extract_explicit_motion_spans(instruction)
    if len(spans) != 1:
        return None, None, None
    span = spans[0]
    return span.direction.value, span.distance_m, span.motion_scale


def _normalize_cardinality(parsed: TaskParseLLMOutput, instruction: str) -> tuple[TaskParseLLMOutput, list[dict[str, object]]]:
    """Make explicit user quantities authoritative over a model proposal.

    In particular, Qwen sometimes represents “两个棒球” as two identical
    candidate entities.  That is a decomposition error, not two semantic
    roles, so collapse it before grounding and planning.
    """
    if parsed.turn_kind != TurnKind.ROBOT_TASK:
        return parsed, []
    evidence = extract_semantic_evidence(instruction)
    result = parsed.model_copy(deep=True)
    repairs: list[dict[str, object]] = []
    aliases = {"苹果": "apple", "香蕉": "banana", "棒球": "baseball", "球": "ball", "盒子": "box", "篮子": "basket"}
    for quantity in evidence.quantities:
        qname = aliases.get(quantity.semantic_name, quantity.semantic_name).casefold()
        matches = [e for e in result.entities if e.name.casefold() == qname
                   or qname in e.name.casefold() or e.name.casefold() in qname]
        if not matches:
            continue
        # Selection language makes the quantity a candidate population;
        # otherwise every explicitly counted member is actionable.
        mode = QuantityMode.CANDIDATE_POOL if not quantity.explicit_all else QuantityMode.ALL
        primary = matches[0]
        old_count = primary.count
        if len(matches) > 1:
            same_roles = all(same_semantic_role(matches[0], candidate) for candidate in matches[1:])
            destinations = {op.destination for op in result.operations if op.source in {e.id for e in matches}}
            if len(destinations) > 1:
                same_roles = False
            if same_roles:
                duplicate_ids = [e.id for e in matches[1:]]
                for op in result.operations:
                    if op.source in duplicate_ids: op.source = primary.id
                    if op.target in duplicate_ids: op.target = primary.id
                    if op.reference in duplicate_ids: op.reference = primary.id
                for rel in result.relations:
                    if rel.subject in duplicate_ids: rel.subject = primary.id
                    if rel.reference in duplicate_ids: rel.reference = primary.id
                result.entities = [e for e in result.entities if e.id not in duplicate_ids]
                repairs.append({"type": "duplicate_entity_merge", "from": duplicate_ids, "to": primary.id, "count": quantity.count})
                # Collapse operations which only differed by the duplicate
                # entity decomposition.
                result.operations, collapsed = collapse_duplicate_operations(result.operations)
                repairs.extend({"type": "duplicate_operation_collapse", "operation": op.type} for op in collapsed)
        if primary.count != quantity.count or primary.quantity_mode != mode:
            repairs.append({"type": "explicit_quantity_repair", "entity": primary.id, "from_count": old_count, "to_count": quantity.count, "quantity_mode": mode.value})
        primary.count = quantity.count
        primary.quantity_mode = mode
    return result, repairs


def _normalize_support_relative_relations(
    parsed: TaskParseLLMOutput,
) -> tuple[TaskParseLLMOutput, list[dict[str, object]]]:
    """Turn table-relative unary directions into binary scene constraints.

    The compact ``left``/``back`` relations are valid selectors without a
    reference (for example “the left apple”).  Qwen also uses that vocabulary
    for phrases such as “the apple in the table's lower-left corner”, where a
    support surface is explicitly the frame of reference.  Keep the strict
    contract for arbitrary references, but canonicalize support-frame uses to
    ``left_of``/``behind`` so they can be grounded and composed safely.
    """
    unary_to_binary = {
        SpatialRelationType.LEFT: SpatialRelationType.LEFT_OF,
        SpatialRelationType.RIGHT: SpatialRelationType.RIGHT_OF,
        SpatialRelationType.FRONT: SpatialRelationType.FRONT_OF,
        SpatialRelationType.BACK: SpatialRelationType.BEHIND,
        SpatialRelationType.UP: SpatialRelationType.ABOVE,
        SpatialRelationType.DOWN: SpatialRelationType.BELOW,
    }
    support_categories = {"support_surface", "surface", "table", "shelf", "furniture"}
    entities = {entity.id: entity for entity in parsed.entities}
    repairs: list[dict[str, object]] = []
    relations = []
    for relation in parsed.relations:
        reference = entities.get(relation.reference) if relation.reference else None
        is_support_reference = reference is not None and (
            reference.category.casefold() in support_categories
            or reference.name.casefold() in {"table", "桌子", "桌面", "台面", "shelf", "架子"}
        )
        replacement = unary_to_binary.get(relation.relation)
        if relation.reference and replacement is not None and is_support_reference:
            relations.append(relation.model_copy(update={"relation": replacement}))
            repairs.append({
                "type": "support_frame_relation_repair",
                "subject": relation.subject,
                "reference": relation.reference,
                "from": relation.relation.value,
                "to": replacement.value,
            })
        else:
            relations.append(relation)
    if relations == parsed.relations:
        return parsed, repairs
    return parsed.model_copy(update={"relations": relations}), repairs


def enrich_task(parsed: TaskParseLLMOutput, instruction: str, motion_policy: MotionPolicy | None = None) -> TaskIntent:
    policy = motion_policy or MotionPolicy()
    parsed, quantity_repairs = _normalize_cardinality(parsed, instruction)
    # A local model can preserve every entity/relation yet omit the obvious
    # high-level operation.  Repair only the unambiguous transitive placement
    # form; never invent an action for arbitrary entity-only proposals.
    missing_operation_repairs: list[dict[str, object]] = []
    if parsed.status == "accepted" and parsed.turn_kind == TurnKind.ROBOT_TASK and not parsed.operations:
        placement_language = any(token in instruction.casefold() for token in ("放进", "放入", "放到", "put into", "put in"))
        container_categories = {"container", "box", "basket", "location"}
        sources = [entity for entity in parsed.entities if entity.category.casefold() not in container_categories]
        destinations = [entity for entity in parsed.entities if entity.category.casefold() in container_categories]
        if placement_language and len(sources) == 1 and len(destinations) == 1:
            parsed.operations.append(ParseOperation(type="pick_and_place", source=sources[0].id, destination=destinations[0].id))
            missing_operation_repairs.append({
                "type": "missing_operation_repair",
                "operation": "pick_and_place",
                "source": sources[0].id,
                "destination": destinations[0].id,
                "reason": "explicit placement language with unique source and destination",
            })
        else:
            return TaskIntent(
                status=TaskStatus.INVALID,
                instruction=instruction,
                explanation="模型未返回可安全恢复的高层操作。",
            )
    from ..semantics.placement_normalizer import normalize_placement_operations
    parsed, placement_repairs = normalize_placement_operations(parsed, instruction)
    parsed, support_relation_repairs = _normalize_support_relative_relations(parsed)
    clarification = next(
        (repair for repair in placement_repairs if repair.get("type") == "placement_clarification_required"),
        None,
    )
    if clarification is not None:
        reason = clarification.get("reason")
        explanation = (
            "当前指令要求放到容器上方，但该容器没有声明可支撑的顶部平面；"
            "请明确改为放进容器、放到桌面，或指定可放置的顶部。"
            if reason == "container_has_no_supportable_top_surface"
            else "放置目标的语义不明确或当前场景没有该支撑面，请明确桌面、架子或其他可用支撑面。"
        )
        return TaskIntent(
            status=TaskStatus.CLARIFICATION_REQUIRED,
            instruction=instruction,
            semantic_repairs=[*quantity_repairs, *placement_repairs],
            explanation=explanation,
        )
    explicit_evidence = extract_semantic_evidence(instruction)
    semantic_aliases = {"苹果": "apple", "香蕉": "banana", "棒球": "baseball", "球": "ball", "盒子": "box", "篮子": "basket"}
    motion_spans = _extract_explicit_motion_spans(instruction)
    # Multiple direction spans are valid when the model preserved them as
    # operation-local moves (for example, “先右移，再前移”).  Clarify only
    # when the proposal collapsed several user motions into an ambiguous
    # single operation.
    if len(motion_spans) > 1 and not (
        len(parsed.operations) >= len(motion_spans)
        and all(operation.type == "move" and operation.motion_direction for operation in parsed.operations[:len(motion_spans)])
        and len({operation.target or operation.source for operation in parsed.operations[:len(motion_spans)]}) == 1
    ):
        return TaskIntent(
            status=TaskStatus.CLARIFICATION_REQUIRED,
            instruction=instruction,
            explanation="当前一次任务只支持一个明确的方向移动，请拆成多个连续指令。",
        )
    text_direction, text_distance, text_motion_scale = _instruction_motion(instruction)
    has_model_move = any(operation.type == "move" for operation in parsed.operations)
    parsed_move_count = sum(1 for operation in parsed.operations if operation.type == "move")
    repairs: list[dict[str, object]] = [*quantity_repairs, *missing_operation_repairs, *placement_repairs, *support_relation_repairs]
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
        model_direction = op.motion_direction or (parsed.raw_direction if operation_type == "move" and parsed_move_count == 1 else None)
        model_distance = op.distance_m or (parsed.distance_m if operation_type == "move" and parsed_move_count == 1 else None)
        model_scale = op.motion_scale
        direction = model_direction
        distance_m = model_distance
        motion_scale = model_scale
        if operation_type == "move" and text_direction is not None and len(motion_spans) == 1:
            if model_direction != text_direction and model_direction is not None:
                repairs.append({"field": f"op-{index + 1}.motion_direction", "model_value": model_direction, "text_value": text_direction, "chosen": text_direction, "reason": "explicit_user_constraint"})
            direction = text_direction
            if text_distance is not None and model_distance != text_distance:
                repairs.append({"field": f"op-{index + 1}.distance_m", "model_value": model_distance, "text_value": text_distance, "chosen": text_distance, "reason": "explicit_user_constraint"})
            distance_m = text_distance
            motion_scale = text_motion_scale if text_distance is None else None
        elif operation_type == "move" and index < len(motion_spans):
            span = motion_spans[index]
            if distance_m is None:
                motion_scale = span.motion_scale
        if operation_type == "move" and direction is None and parsed_move_count > 1:
            raise ValueError("task_semantic_invalid: every move operation requires operation-local motion_direction")
        if operation_type == "move" and direction is not None and distance_m is None and motion_scale is None:
            motion_scale = MotionScale.SMALL
            repairs.append({"field": f"op-{index + 1}.motion_scale", "model_value": None, "chosen": motion_scale.value, "reason": "default_small_motion_scale"})
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
            motion_scale=motion_scale,
            placement_target=op.placement_target,
        ))
    task_types = list(dict.fromkeys(operation.task_type for operation in operations))
    if len(task_types) > 1:
        task_types.append(TaskType.MIXED)
    messages = {
        "direction_clarification_required": "当前仅支持上下、左右、前后，请明确选择其中一个方向。",
        "unsupported_task": "当前不支持该任务类型。",
    }
    multi_hold = any(
        operation.task_type == TaskType.GRASP
        and any(entity.id in {operation.source, operation.target} and entity.quantity_mode == QuantityMode.ALL and entity.count > 1 for entity in parsed.entities)
        for operation in operations
    )
    if multi_hold:
        return TaskIntent(
            status=TaskStatus.PHYSICAL_CAPABILITY_CONFLICT,
            instruction=instruction,
            task_types=task_types,
            entities=[TaskEntity(entity_id=e.id, semantic_name=e.name, category=_canonical_category(e.category), color=e.color, count=e.count, quantity_mode=e.quantity_mode) for e in parsed.entities],
            operations=operations,
            explanation="单夹爪不能同时保持多个对象，请改为逐个抓取并放置。",
        )
    return TaskIntent(
        status=parsed.status, instruction=instruction, task_types=task_types,
        entities=[TaskEntity(entity_id=e.id, semantic_name=e.name, category=_canonical_category(e.category), color=e.color, count=e.count, quantity_mode=e.quantity_mode,
                             count_explicit=any(q.semantic_name and (semantic_aliases.get(q.semantic_name, q.semantic_name).casefold() in e.name.casefold() or e.name.casefold() in q.semantic_name.casefold()) for q in explicit_evidence.quantities),
                             all_available=e.quantity_mode != QuantityMode.SINGLE) for e in parsed.entities],
        operations=operations,
        spatial_relations=[SpatialRelation(scope=r.scope, subject=r.subject, relation=r.relation, reference=r.reference) for r in parsed.relations],
        raw_direction=text_direction or (parsed.raw_direction if parsed_move_count <= 1 else None) if parsed.status == "accepted" else None,
        semantic_repairs=repairs,
        explanation=messages.get(parsed.status, ""),
    )


def _canonical_category(category: str) -> str:
    value = category.casefold().strip()
    return {
        "box": "container",
        "basket": "container",
        "apple": "fruit",
        "banana": "fruit",
        "baseball": "ball",
    }.get(value, value)
