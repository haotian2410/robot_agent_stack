"""High-level operation and relation contracts independent of model syntax."""

from __future__ import annotations


_ROLE_CONTRACTS = {
    "locate": ({"target"}, {"target"}),
    "search": ({"target"}, {"target"}),
    "grasp": ({"target"}, {"target"}),
    "press": ({"target"}, {"target"}),
    "pick_and_place": ({"source", "destination"}, {"source", "destination"}),
    "open": ({"target", "reference"}, {"target", "reference"}),
    "close": ({"target", "reference"}, {"target", "reference"}),
    "move": (set(), {"target", "source", "reference"}),
    "release": (set(), {"target", "source", "reference"}),
}


def validate_operation_contract(operation) -> None:
    task_type = operation.task_type.value
    required, allowed = _ROLE_CONTRACTS[task_type]
    roles = {
        "source": operation.source,
        "destination": operation.destination,
        "target": operation.target,
        "reference": operation.reference,
    }
    missing = sorted(role for role in required if not roles[role])
    extra = sorted(role for role, value in roles.items() if value and role not in allowed)
    if missing:
        raise ValueError(f"task_semantic_invalid: {task_type} requires roles {missing}")
    if extra:
        raise ValueError(f"task_semantic_invalid: {task_type} forbids roles {extra}")
    if task_type in {"move", "release"} and bool(operation.target) == bool(operation.source):
        raise ValueError(f"task_semantic_invalid: {task_type} requires exactly one of target or source")
    if task_type == "pick_and_place" and operation.source == operation.destination:
        raise ValueError("task_semantic_invalid: pick_and_place source and destination must differ")
    if task_type in {"open", "close"} and operation.target == operation.reference:
        raise ValueError(f"task_semantic_invalid: {task_type} target and reference must differ")
    if task_type != "move" and (operation.motion_direction is not None or operation.distance_m is not None):
        raise ValueError(f"task_semantic_invalid: motion fields are only valid for move")
    if operation.distance_m is not None and operation.motion_direction is None:
        raise ValueError("task_semantic_invalid: distance_m requires motion_direction")


_CONFLICTS = {
    "left": "right", "right": "left",
    "front": "back", "back": "front",
    "up": "down", "down": "up",
    "left_of": "right_of", "right_of": "left_of",
    "front_of": "behind", "behind": "front_of",
    "above": "below", "below": "above",
    "nearest": "farthest", "farthest": "nearest",
    "leftmost": "rightmost", "rightmost": "leftmost",
    "frontmost": "backmost", "backmost": "frontmost",
    "highest": "lowest", "lowest": "highest",
}


def validate_relation_consistency(relations) -> None:
    seen: set[tuple[str, str | None, str, str]] = set()
    for relation in relations:
        key = (relation.subject, relation.reference, relation.scope, relation.relation.value)
        opposite = _CONFLICTS.get(relation.relation.value)
        if opposite and (relation.subject, relation.reference, relation.scope, opposite) in seen:
            raise ValueError(
                "semantic_conflict: contradictory spatial relations "
                f"for {relation.subject}: {opposite} and {relation.relation.value}"
            )
        seen.add(key)
