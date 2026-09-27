"""Canonicalize semantic placement without generating physical coordinates."""

from __future__ import annotations

from ..contracts.placement import PlacementTargetKind, PlacementTargetSpec
from ..contracts.task_intent import SpatialRelationType
from ..models.task_understanding import ParseEntity, ParseOperation, ParseRelation


TABLE_ENTITY = "__table__"


def normalize_placement_operations(parsed, instruction: str):
    """Return ``(parse, repairs)`` with explicit placement semantics.

    This function only interprets language and entity categories.  It never
    computes a pose or inserts a distance.  In particular, a destination
    object is not assumed to be a container unless its category says so.
    """
    if parsed.turn_kind.value != "robot_task" or parsed.status != "accepted":
        return parsed, []
    result = parsed.model_copy(deep=True)
    repairs: list[dict[str, object]] = []
    text = instruction.casefold()
    entities = {entity.id: entity for entity in result.entities}

    def ensure_table() -> str:
        # Qwen may recognise the tabletop as a generic ``furniture`` entity
        # (for example ``table``) rather than the support-surface category.
        # Rebind that semantic entity to the generated scene's synthetic
        # ``__table__`` instead of sending it through AssetResolver, where it
        # would be mistaken for a missing furniture asset.
        candidate = next(
            (
                entity for entity in result.entities
                if entity.id != TABLE_ENTITY
                and (
                    entity.name.casefold() in {"table", "桌子", "桌面", "台面", "shelf", "架子"}
                    or entity.category.casefold() == "support_surface"
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
                        operation.placement_target = placement.model_copy(
                            update={
                                "reference": TABLE_ENTITY if placement.reference == old_id else placement.reference,
                                "support": TABLE_ENTITY if placement.support == old_id else placement.support,
                            }
                        )
            for relation in result.relations:
                if relation.subject == old_id:
                    relation.subject = TABLE_ENTITY
                if relation.reference == old_id:
                    relation.reference = TABLE_ENTITY
            result.entities = [entity for entity in result.entities if entity.id != old_id]
            entities.pop(old_id, None)
        if TABLE_ENTITY not in entities:
            table = ParseEntity(id=TABLE_ENTITY, name="table", category="support_surface")
            result.entities.append(table)
            entities[TABLE_ENTITY] = table
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
        if any(token in text for token in ("旁边", "附近", "near", "next to")):
            return SpatialRelationType.NEAR
        return None

    surface_language = any(token in text for token in ("桌面", "桌上", "台面", "table", "架子上", "架面", "shelf"))
    free_language = any(token in text for token in ("空位置", "空地方", "空着的位置", "空位", "free space", "empty place"))
    inside_language = any(token in text for token in ("里面", "里", "inside", "from"))

    # Preserve/augment source selection semantics for “盒子里的球”.
    for operation in result.operations:
        if operation.type != "pick_and_place" or not operation.source:
            continue
        source = entities.get(operation.source)
        container = next((item for item in result.entities if item.category.casefold() in {"container", "box", "basket", "location"} and item.id != operation.source), None)
        if source and container and inside_language and not any(
            rel.subject == operation.source and rel.reference == container.id and rel.relation == SpatialRelationType.INSIDE
            for rel in result.relations
        ):
            result.relations.append(ParseRelation(scope="selection", subject=operation.source, relation=SpatialRelationType.INSIDE, reference=container.id))
            repairs.append({"type": "inside_selection_repair", "source": operation.source, "reference": container.id})

    # “拿出/取出” has a source selector but no explicit destination.  The
    # policy target is the single primary tabletop, represented synthetically.
    if not result.operations and any(token in text for token in ("拿出来", "取出来", "拿出", "取出", "take out", "remove from")):
        source = next((entity for entity in result.entities if entity.category.casefold() not in {"container", "box", "basket", "location", "support_surface"}), None)
        if source:
            destination = ensure_table()
            result.operations.append(ParseOperation(type="pick_and_place", source=source.id, destination=destination))
            repairs.append({"type": "implicit_free_space_placement", "support": destination, "reason": "take_out_without_explicit_destination"})

    if not result.operations and free_language:
        source = next((entity for entity in result.entities if entity.category.casefold() not in {"container", "box", "basket", "location", "support_surface"}), None)
        if source:
            destination = ensure_table()
            result.operations.append(ParseOperation(type="pick_and_place", source=source.id, destination=destination))
            repairs.append({"type": "implicit_free_space_placement", "support": destination, "reason": "explicit_free_space_language"})

    for index, operation in enumerate(result.operations):
        if operation.type != "pick_and_place" or not operation.source:
            continue
        destination = operation.destination
        if free_language and (destination is None or destination not in entities or entities[destination].category.casefold() not in {"container", "box", "basket", "location"}):
            destination = ensure_table()
            operation.destination = destination
            spec = PlacementTargetSpec(kind=PlacementTargetKind.FREE_SPACE, reference=destination)
        elif surface_language and (destination is None or destination not in entities or entities[destination].category.casefold() not in {"container", "box", "basket", "location"}):
            destination = ensure_table()
            operation.destination = destination
            spec = PlacementTargetSpec(kind=PlacementTargetKind.SUPPORT_SURFACE, reference=destination, relation=SpatialRelationType.ON)
        elif destination is not None and destination in entities and entities[destination].category.casefold() in {"container", "box", "basket", "location"}:
            spec = PlacementTargetSpec(kind=PlacementTargetKind.CONTAINER_INTERIOR, reference=destination, relation=SpatialRelationType.INSIDE)
        elif destination is not None:
            relation = relation_for(operation.source, destination)
            if relation is None:
                # A non-container destination without an explicit relation is
                # ambiguous and must not silently become a container target.
                continue
            spec = PlacementTargetSpec(kind=PlacementTargetKind.RELATIVE_OBJECT, reference=destination, relation=relation)
        else:
            continue
        if operation.placement_target != spec:
            if operation.placement_target is not None:
                repairs.append({"type": "placement_semantic_repair", "operation": f"op-{index + 1}", "from": operation.placement_target.model_dump(mode="json"), "to": spec.model_dump(mode="json")})
            operation.placement_target = spec
        if operation.placement_target is not None and operation.placement_target.kind == PlacementTargetKind.SUPPORT_SURFACE:
            # “桌面上” is a placement goal, not a source selector.
            result.relations = [
                relation for relation in result.relations
                if not (relation.scope == "selection" and relation.relation in {
                    SpatialRelationType.UP, SpatialRelationType.DOWN,
                })
            ]
            if not any(relation.subject == operation.source and relation.reference == operation.destination and relation.relation == SpatialRelationType.ON for relation in result.relations):
                result.relations.append(ParseRelation(scope="goal", subject=operation.source, relation=SpatialRelationType.ON, reference=operation.destination))
        if operation.placement_target is not None and operation.placement_target.kind == PlacementTargetKind.RELATIVE_OBJECT:
            # A phrase such as “放到棒球右边” must not turn the reference
            # object itself into a selected “right” candidate.
            result.relations = [
                relation for relation in result.relations
                if not (relation.subject == operation.destination and relation.scope == "selection" and relation.relation in {
                    SpatialRelationType.LEFT, SpatialRelationType.RIGHT,
                    SpatialRelationType.FRONT, SpatialRelationType.BACK,
                })
            ]
    return result, repairs


__all__ = ["TABLE_ENTITY", "normalize_placement_operations"]
