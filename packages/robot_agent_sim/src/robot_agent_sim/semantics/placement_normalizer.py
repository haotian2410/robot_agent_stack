"""Canonicalize semantic placement without generating physical coordinates.

The language model is allowed to propose a destination, but destination roles
are checked against the user's words.  This repairs common model mistakes such
as treating ``拿出来`` as a bare grasp and silently turning ``盒子上面`` into
the box interior.  Ambiguous placement is marked for clarification instead of
falling back to a default anchor.
"""

from __future__ import annotations

import re

from ..contracts.placement import PlacementTargetKind, PlacementTargetSpec
from ..contracts.task_intent import SpatialRelationType
from ..models.task_understanding import ParseEntity, ParseOperation, ParseRelation


TABLE_ENTITY = "__table__"
_CONTAINER_CATEGORIES = {"container", "box", "basket", "location"}
_SUPPORT_CATEGORIES = {"support_surface", "surface", "table", "shelf"}
_TAKEOUT_TOKENS = ("拿出来", "取出来", "拿出", "取出", "take out", "remove from")
_PLACEMENT_MARKER = re.compile(
    r"(?:放到|放入|放进|放在|置于|put\s+(?:into|in|on|to)|place\s+(?:into|in|on|at))",
    re.IGNORECASE,
)


def _is_container(entity: ParseEntity | None) -> bool:
    return entity is not None and entity.category.casefold() in _CONTAINER_CATEGORIES


def _is_support(entity: ParseEntity | None) -> bool:
    return entity is not None and entity.category.casefold() in _SUPPORT_CATEGORIES


def _is_payload(entity: ParseEntity | None) -> bool:
    return entity is not None and not _is_container(entity) and not _is_support(entity)


def _entity_aliases(entity: ParseEntity) -> tuple[str, ...]:
    aliases = {entity.id, entity.name}
    aliases.update(part for part in entity.id.split("_") if len(part) > 1)
    aliases.update(part for part in entity.name.casefold().split() if len(part) > 1)
    semantic_aliases = {
        "apple": "苹果", "banana": "香蕉", "baseball": "棒球", "ball": "球",
        "basket": "篮子", "box": "盒子", "open box": "盒子", "table": "桌面",
        "shelf": "架子",
    }
    for key, alias in semantic_aliases.items():
        if key in entity.name.casefold() or key in entity.id.casefold():
            aliases.add(alias)
    return tuple(sorted((value.casefold() for value in aliases if value), key=len, reverse=True))


def _entity_mentions(text: str, entities: list[ParseEntity]) -> dict[str, list[tuple[int, int]]]:
    mentions: dict[str, list[tuple[int, int]]] = {entity.id: [] for entity in entities}
    folded = text.casefold()
    for entity in entities:
        for alias in _entity_aliases(entity):
            start = folded.find(alias)
            while start >= 0:
                mentions[entity.id].append((start, start + len(alias)))
                start = folded.find(alias, start + 1)
        mentions[entity.id] = sorted(set(mentions[entity.id]))
    return mentions


def _explicit_roles(
    text: str,
    entities: list[ParseEntity],
    *,
    mentions: dict[str, list[tuple[int, int]]] | None = None,
) -> tuple[str | None, str | None]:
    """Infer source/destination from an explicit ``把 A 放到 B`` phrase."""
    marker = _PLACEMENT_MARKER.search(text)
    if marker is None:
        return None, None
    mentions = mentions or _entity_mentions(text, entities)
    before: list[tuple[int, ParseEntity]] = []
    after: list[tuple[int, ParseEntity]] = []
    for entity in entities:
        for start, end in mentions.get(entity.id, []):
            if end <= marker.start():
                before.append((start, entity))
            elif start >= marker.end():
                after.append((start, entity))
    source_candidates = [(start, entity) for start, entity in before if _is_payload(entity)]
    source = max(source_candidates, key=lambda item: item[0])[1] if source_candidates else None
    destination_candidates = [(start, entity) for start, entity in after if entity.id != (source.id if source else None)]
    destination = min(destination_candidates, key=lambda item: item[0])[1] if destination_candidates else None
    return (source.id if source else None), (destination.id if destination else None)


def _text_relation(text: str) -> SpatialRelationType | None:
    patterns = (
        (("右边", "右侧", "右面", "right of", "to the right"), SpatialRelationType.RIGHT_OF),
        (("左边", "左侧", "左面", "left of", "to the left"), SpatialRelationType.LEFT_OF),
        (("前面", "前方", "front of", "in front"), SpatialRelationType.FRONT_OF),
        (("后面", "后方", "behind", "back of"), SpatialRelationType.BEHIND),
        (("旁边", "附近", "near", "next to"), SpatialRelationType.NEAR),
        (("下方", "下面", "下边", "below", "under"), SpatialRelationType.BELOW),
        (("上方", "上面", "上边", "above", "over"), SpatialRelationType.ABOVE),
    )
    marker = _PLACEMENT_MARKER.search(text)
    suffix = text[marker.end():] if marker else text
    for tokens, relation in patterns:
        if any(token.casefold() in suffix.casefold() for token in tokens):
            return relation
    return None


def _has_top_surface_word(text: str) -> bool:
    marker = _PLACEMENT_MARKER.search(text)
    suffix = text[marker.end():] if marker else text
    if any(token in suffix.casefold() for token in ("上面", "上方", "顶部", "顶上", "上边", "on top", "above")):
        return True
    return bool(re.search(r"(?:盒子|箱子|容器|篮子|box|container|basket)\s*上(?:面)?", suffix, re.IGNORECASE))


def normalize_placement_operations(parsed, instruction: str):
    """Return ``(parse, repairs)`` with explicit placement semantics."""
    if parsed.turn_kind.value != "robot_task" or parsed.status != "accepted":
        return parsed, []
    result = parsed.model_copy(deep=True)
    repairs: list[dict[str, object]] = []
    text = instruction.casefold()
    entities = {entity.id: entity for entity in result.entities}
    mentions = _entity_mentions(instruction, result.entities)
    explicit_source, explicit_destination = _explicit_roles(instruction, result.entities, mentions=mentions)
    if explicit_destination and re.search(r"另一个\s*(?:的)?\s*盒子|another\s+box", instruction, re.IGNORECASE):
        # Generic aliases such as “盒子” occur at the same character span for
        # both duplicated container entities.  The phrase “另一个盒子” is an
        # explicit request for the second deterministic container role.
        containers = [entity for entity in result.entities if _is_container(entity)]
        if len(containers) > 1:
            explicit_destination = containers[1].id
    relation_from_text = _text_relation(instruction)
    has_takeout = any(token in text for token in _TAKEOUT_TOKENS)
    table_language = any(token in text for token in ("桌面", "桌上", "台面", "桌子", "table"))
    shelf_language = any(token in text for token in ("架子", "架面", "shelf"))
    surface_language = table_language or shelf_language
    free_language = any(token in text for token in ("空位置", "空地方", "空着的位置", "空位", "free space", "empty place"))
    placement_marker = _PLACEMENT_MARKER.search(instruction)
    source_phrase = instruction[:placement_marker.start()] if placement_marker is not None else instruction
    # ``篮子里`` after the placement verb names the destination, not an
    # INSIDE source selector.  Only inspect the source phrase for this repair.
    inside_language = any(token in source_phrase.casefold() for token in ("里面", "里", "inside", "from"))

    def ensure_table() -> str:
        candidate = next(
            (
                entity for entity in result.entities
                if entity.id != TABLE_ENTITY
                and (
                    entity.name.casefold() in {"table", "桌子", "桌面", "台面", "shelf", "架子"}
                    or entity.category.casefold() in _SUPPORT_CATEGORIES
                )
            ),
            None,
        )
        if candidate is not None:
            old_id = candidate.id
            for operation in result.operations:
                for field in ("source", "destination", "target", "reference"):
                    if getattr(operation, field) == old_id:
                        setattr(operation, field, TABLE_ENTITY)
                if operation.placement_target is not None:
                    placement = operation.placement_target
                    if placement.reference == old_id or placement.support == old_id:
                        operation.placement_target = placement.model_copy(update={
                            "reference": TABLE_ENTITY if placement.reference == old_id else placement.reference,
                            "support": TABLE_ENTITY if placement.support == old_id else placement.support,
                        })
            for relation in result.relations:
                if relation.subject == old_id:
                    relation.subject = TABLE_ENTITY
                if relation.reference == old_id:
                    relation.reference = TABLE_ENTITY
            result.entities = [entity for entity in result.entities if entity.id != old_id]
            entities.pop(old_id, None)
            mentions.pop(old_id, None)
        if TABLE_ENTITY not in entities:
            table = ParseEntity(id=TABLE_ENTITY, name="table", category="support_surface")
            result.entities.append(table)
            entities[TABLE_ENTITY] = table
            mentions[TABLE_ENTITY] = []
        return TABLE_ENTITY

    def relation_for(source: str, destination: str):
        for relation in result.relations:
            if relation.subject == source and relation.reference == destination and relation.relation in {
                SpatialRelationType.LEFT_OF, SpatialRelationType.RIGHT_OF,
                SpatialRelationType.FRONT_OF, SpatialRelationType.BEHIND,
                SpatialRelationType.NEAR, SpatialRelationType.ABOVE,
                SpatialRelationType.BELOW,
            }:
                return relation.relation
        return relation_from_text

    def add_inside_selection(source_id: str, container_id: str | None) -> None:
        if not container_id or source_id == container_id or not _is_container(entities.get(container_id)):
            return
        if not any(
            rel.scope == "selection" and rel.subject == source_id
            and rel.reference == container_id and rel.relation == SpatialRelationType.INSIDE
            for rel in result.relations
        ):
            result.relations.append(ParseRelation(scope="selection", subject=source_id, relation=SpatialRelationType.INSIDE, reference=container_id))
            repairs.append({"type": "inside_selection_repair", "source": source_id, "reference": container_id})

    # Make an explicit ``把 A 放到 B`` role ordering authoritative before
    # classifying the destination.
    if explicit_source and explicit_destination:
        for operation in result.operations:
            if operation.type not in {"pick_and_place", "grasp"}:
                continue
            old_roles = (operation.source, operation.destination, operation.type)
            operation.source = explicit_source
            operation.destination = explicit_destination
            if operation.type == "grasp":
                operation.type = "pick_and_place"
                operation.target = None
                operation.reference = None
            if old_roles != (operation.source, operation.destination, operation.type):
                repairs.append({
                    "type": "explicit_placement_role_repair",
                    "from": {"source": old_roles[0], "destination": old_roles[1], "operation": old_roles[2]},
                    "to": {"source": explicit_source, "destination": explicit_destination, "operation": operation.type},
                })

    # Recover the source selector from ``盒子里的球``.
    inside_pair = next((rel for rel in result.relations if rel.relation == SpatialRelationType.INSIDE and rel.scope == "selection"), None)
    if inside_pair is None and inside_language:
        source_guess = explicit_source or next((op.source for op in result.operations if op.source and _is_payload(entities.get(op.source))), None)
        source_guess = source_guess or next((op.target for op in result.operations if op.target and _is_payload(entities.get(op.target))), None)
        source_guess = source_guess or next((entity.id for entity in result.entities if _is_payload(entity)), None)
        container_guess = next((entity.id for entity in result.entities if _is_container(entity) and entity.id != source_guess), None)
        if source_guess and container_guess:
            add_inside_selection(source_guess, container_guess)
            inside_pair = next((rel for rel in result.relations if rel.relation == SpatialRelationType.INSIDE and rel.scope == "selection"), None)

    # ``拿出来`` is not a new atomic skill.  Normalize it to ordinary
    # pick-and-place, with a policy free-space destination when no endpoint is
    # stated.
    if has_takeout:
        source_id = inside_pair.subject if inside_pair is not None else explicit_source
        if source_id is None:
            source_id = next((entity.id for entity in result.entities if _is_payload(entity)), None)
        if source_id is not None:
            destination_id = explicit_destination
            if free_language or destination_id is None and not surface_language:
                destination_id = ensure_table()
                chosen_kind = PlacementTargetKind.FREE_SPACE
                repairs.append({
                    "type": "implicit_free_space_placement",
                    "support": destination_id,
                    "reason": "take_out_without_explicit_destination" if not free_language else "explicit_free_space_language",
                })
            elif table_language:
                destination_id = ensure_table()
                chosen_kind = PlacementTargetKind.SUPPORT_SURFACE
            elif shelf_language:
                destination_entity = entities.get(destination_id)
                if not _is_support(destination_entity) or destination_id == TABLE_ENTITY:
                    repairs.append({"type": "placement_clarification_required", "reason": "named_support_surface_missing", "reference": "shelf"})
                    chosen_kind = PlacementTargetKind.SUPPORT_SURFACE
                else:
                    chosen_kind = PlacementTargetKind.SUPPORT_SURFACE
            else:
                chosen_kind = None
            operation = next((item for item in result.operations if item.type in {"pick_and_place", "grasp"}), None)
            if operation is None:
                operation = ParseOperation(type="pick_and_place", source=source_id, destination=destination_id)
                result.operations.append(operation)
            operation.type = "pick_and_place"
            operation.source = source_id
            operation.destination = destination_id
            operation.target = None
            operation.reference = None
            add_inside_selection(source_id, inside_pair.reference if inside_pair is not None else None)
            if chosen_kind is None:
                destination_entity = entities.get(destination_id)
                if _is_container(destination_entity):
                    chosen_kind = PlacementTargetKind.CONTAINER_INTERIOR
                elif _is_support(destination_entity):
                    chosen_kind = PlacementTargetKind.SUPPORT_SURFACE
                else:
                    chosen_kind = PlacementTargetKind.RELATIVE_OBJECT
            if chosen_kind == PlacementTargetKind.CONTAINER_INTERIOR:
                spec = PlacementTargetSpec(kind=chosen_kind, reference=destination_id, relation=SpatialRelationType.INSIDE)
            elif chosen_kind == PlacementTargetKind.SUPPORT_SURFACE:
                spec = PlacementTargetSpec(kind=chosen_kind, reference=destination_id, relation=SpatialRelationType.ON)
            elif chosen_kind == PlacementTargetKind.FREE_SPACE:
                spec = PlacementTargetSpec(kind=chosen_kind, reference=destination_id)
            else:
                relation = relation_for(source_id, destination_id)
                if relation is None:
                    repairs.append({"type": "placement_clarification_required", "reason": "relative_destination_relation_missing"})
                    relation = SpatialRelationType.NEAR
                spec = PlacementTargetSpec(kind=chosen_kind, reference=destination_id, relation=relation)
            operation.placement_target = spec

    if not result.operations and free_language:
        source = explicit_source or next((entity.id for entity in result.entities if _is_payload(entity)), None)
        if source:
            destination = ensure_table()
            result.operations.append(ParseOperation(type="pick_and_place", source=source, destination=destination, placement_target=PlacementTargetSpec(kind=PlacementTargetKind.FREE_SPACE, reference=destination)))
            repairs.append({"type": "implicit_free_space_placement", "support": destination, "reason": "explicit_free_space_language"})

    for index, operation in enumerate(result.operations):
        if operation.type != "pick_and_place" or not operation.source:
            continue
        destination_id = operation.destination
        destination = entities.get(destination_id) if destination_id else None
        if free_language and (destination is None or not _is_container(destination) or table_language):
            destination_id = ensure_table()
            operation.destination = destination_id
            spec = PlacementTargetSpec(kind=PlacementTargetKind.FREE_SPACE, reference=destination_id)
        elif table_language:
            destination_id = ensure_table()
            operation.destination = destination_id
            spec = PlacementTargetSpec(kind=PlacementTargetKind.SUPPORT_SURFACE, reference=destination_id, relation=SpatialRelationType.ON)
        elif shelf_language:
            if _is_support(destination) and destination_id != TABLE_ENTITY:
                spec = PlacementTargetSpec(kind=PlacementTargetKind.SUPPORT_SURFACE, reference=destination_id, relation=SpatialRelationType.ON)
            else:
                repairs.append({"type": "placement_clarification_required", "operation": f"op-{index + 1}", "reason": "named_support_surface_missing", "reference": "shelf"})
                operation.placement_target = None
                continue
        elif _is_container(destination):
            if _has_top_surface_word(instruction):
                repairs.append({
                    "type": "placement_clarification_required",
                    "operation": f"op-{index + 1}",
                    "reason": "container_has_no_supportable_top_surface",
                    "reference": destination_id,
                })
                operation.placement_target = None
                continue
            spec = PlacementTargetSpec(kind=PlacementTargetKind.CONTAINER_INTERIOR, reference=destination_id, relation=SpatialRelationType.INSIDE)
        elif destination is not None:
            relation = relation_for(operation.source, destination_id)
            if relation is None:
                continue
            spec = PlacementTargetSpec(kind=PlacementTargetKind.RELATIVE_OBJECT, reference=destination_id, relation=relation)
        else:
            continue

        if operation.placement_target != spec:
            if operation.placement_target is not None:
                repairs.append({"type": "placement_semantic_repair", "operation": f"op-{index + 1}", "from": operation.placement_target.model_dump(mode="json"), "to": spec.model_dump(mode="json")})
            operation.placement_target = spec
        if operation.placement_target.kind in {PlacementTargetKind.SUPPORT_SURFACE, PlacementTargetKind.FREE_SPACE}:
            result.relations = [
                relation for relation in result.relations
                if not (relation.scope == "selection" and relation.relation in {SpatialRelationType.UP, SpatialRelationType.DOWN})
            ]
            if not any(relation.subject == operation.source and relation.reference == operation.destination and relation.relation == SpatialRelationType.ON for relation in result.relations):
                result.relations.append(ParseRelation(scope="goal", subject=operation.source, relation=SpatialRelationType.ON, reference=operation.destination))
        elif operation.placement_target.kind == PlacementTargetKind.RELATIVE_OBJECT:
            result.relations = [
                relation for relation in result.relations
                if not (relation.subject in {operation.source, operation.destination} and relation.scope == "selection" and relation.relation in {
                    SpatialRelationType.LEFT, SpatialRelationType.RIGHT,
                    SpatialRelationType.FRONT, SpatialRelationType.BACK,
                })
            ]
            if not any(relation.subject == operation.source and relation.reference == operation.destination and relation.relation == operation.placement_target.relation for relation in result.relations):
                result.relations.append(ParseRelation(scope="goal", subject=operation.source, relation=operation.placement_target.relation, reference=operation.destination))

    # A provider can leave behind a goal for its pre-repair roles (for
    # example ``apple INSIDE box`` after it confused the source with the
    # baseball selected from that box).  Keep only goals represented by the
    # canonical placement operations; selection relations remain untouched.
    canonical_goals = set()
    for operation in result.operations:
        if operation.type != "pick_and_place" or operation.placement_target is None:
            continue
        relation = {
            PlacementTargetKind.CONTAINER_INTERIOR: SpatialRelationType.INSIDE,
            PlacementTargetKind.SUPPORT_SURFACE: SpatialRelationType.ON,
            PlacementTargetKind.FREE_SPACE: SpatialRelationType.ON,
            PlacementTargetKind.RELATIVE_OBJECT: operation.placement_target.relation,
        }[operation.placement_target.kind]
        canonical_goals.add((operation.source, relation, operation.destination))
    if canonical_goals:
        result.relations = [
            relation for relation in result.relations
            if relation.scope != "goal"
            or (relation.subject, relation.relation, relation.reference) in canonical_goals
        ]

    return result, repairs


__all__ = ["TABLE_ENTITY", "normalize_placement_operations"]
