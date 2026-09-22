"""Deterministic semantic grounding for authored interaction registries."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import mujoco

from ..contracts.grounded_task import GroundedEntity
from .world_relation import WorldRelationResolver
from .name_matching import exact_name_match, normalize_name


def ground_with_interaction_registry(
    entities,
    registry_path: str | Path,
    scene_path: str | Path,
    *,
    intent=None,
    positions: dict[str, tuple[float, float, float] | list[float]] | None = None,
    bounds: dict[str, tuple[tuple[float, float, float], tuple[float, float, float]]] | None = None,
    excluded_object_ids: set[str] | None = None,
) -> list[GroundedEntity]:
    grounded, missing = ground_partial_with_interaction_registry(
        entities, registry_path, scene_path, intent=intent, positions=positions, bounds=bounds, excluded_object_ids=excluded_object_ids
    )
    if missing:
        names = ", ".join(entity.semantic_name for entity in missing)
        raise ValueError(f"interaction registry grounding is missing: {names}")
    return grounded


def ground_partial_with_interaction_registry(
    entities,
    registry_path: str | Path,
    scene_path: str | Path,
    *,
    intent=None,
    positions: dict[str, tuple[float, float, float] | list[float]] | None = None,
    bounds: dict[str, tuple[tuple[float, float, float], tuple[float, float, float]]] | None = None,
    excluded_object_ids: set[str] | None = None,
) -> tuple[list[GroundedEntity], list[Any]]:
    """Ground the subset covered by an authored registry.

    Uploaded scenes often have a deliberately small interaction sidecar.  A
    missing entry is not the same as an invalid scene: callers can send only
    the unresolved entities through geometry/vision grounding while preserving
    authored execution metadata for the entries that are known.
    """
    registry = json.loads(Path(registry_path).read_text(encoding="utf-8"))
    objects = registry.get("objects")
    if not isinstance(objects, dict):
        raise ValueError("interaction registry must contain an objects mapping")
    model = mujoco.MjModel.from_xml_path(str(Path(scene_path).resolve()))

    candidates_by_entity: dict[str, list[tuple[str, dict[str, Any]]]] = {}
    missing_entities = []
    for entity in entities:
        query_values = {entity.semantic_name, *entity.aliases}
        exact: list[tuple[str, dict[str, Any]]] = []
        fuzzy: list[tuple[str, dict[str, Any]]] = []
        for object_id, item in objects.items():
            if object_id in (excluded_object_ids or set()):
                continue
            names = {object_id, str(item.get("object_id", object_id)), *(str(value) for value in item.get("aliases", []))}
            if any(exact_name_match(query, name) for query in query_values for name in names):
                exact.append((object_id, item))
        candidates = exact or fuzzy
        if not candidates:
            missing_entities.append(entity)
            continue
        candidates_by_entity[entity.entity_id] = candidates

    if not candidates_by_entity:
        return [], missing_entities

    selected: dict[str, str] = {}
    if intent is not None and positions is not None and all(
        entity.entity_id in candidates_by_entity for entity in intent.entities
    ):
        resolver_candidates = {
            entity_id: [{"object_id": object_id} for object_id, _ in values]
            for entity_id, values in candidates_by_entity.items()
        }
        selected = {
            entity_id: value["object_id"]
            for entity_id, value in WorldRelationResolver().resolve(intent, resolver_candidates, positions, bounds=bounds).items()
        }
    else:
        used: set[str] = set()
        for entity in entities:
            if entity.entity_id not in candidates_by_entity:
                continue
            available = [item for item in candidates_by_entity[entity.entity_id] if item[0] not in used]
            if len(available) != 1:
                raise ValueError(
                    f"interaction registry grounding for {entity.semantic_name!r} is ambiguous: "
                    f"{[item[0] for item in available]}"
                )
            selected[entity.entity_id] = available[0][0]
            used.add(available[0][0])

    result: list[GroundedEntity] = []
    for entity in entities:
        if entity.entity_id not in selected:
            continue
        object_id = selected[entity.entity_id]
        item = dict(objects[object_id])
        result.append(
            GroundedEntity(
                entity_id=entity.entity_id,
                semantic_name=entity.semantic_name,
                object_id=object_id,
                body_name=_source_body_name(model, item),
                category=entity.category,
                color=entity.color,
                aliases=entity.aliases,
                quantity_mode=entity.quantity_mode,
                grounding_method="interaction_registry",
            )
        )
    return result, missing_entities


def _normalize(value: str) -> str:
    return re.sub(r"[^\w\u4e00-\u9fff]+", "", value.casefold())


def _source_body_name(model: mujoco.MjModel, item: dict[str, Any]) -> str | None:
    source = item.get("spatial", {}).get("source", {})
    name = source.get("name")
    source_type = source.get("type")
    if not isinstance(name, str):
        return None
    if source_type == "body":
        return name
    kind = {
        "site": mujoco.mjtObj.mjOBJ_SITE,
        "geom": mujoco.mjtObj.mjOBJ_GEOM,
    }.get(source_type)
    if kind is None:
        return None
    object_id = mujoco.mj_name2id(model, kind, name)
    if object_id < 0:
        raise ValueError(f"interaction registry source not found in scene: {source_type} {name}")
    body_id = int(model.site_bodyid[object_id]) if source_type == "site" else int(model.geom_bodyid[object_id])
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
