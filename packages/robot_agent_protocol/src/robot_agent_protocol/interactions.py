from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

from pydantic import Field

from .commands import StrictModel


class InteractionRegistryDocument(StrictModel):
    version: int | None = None
    registry_version: int | None = None
    schema_version: str = "1.0"
    coordinate_frame: str | None = None
    scene: str
    scene_fingerprint: str | None = None
    move_defaults: dict[str, Any] = Field(default_factory=dict)
    named_targets: dict[str, Any] = Field(default_factory=dict)
    objects: dict[str, Any]


class InteractionRegistryError(ValueError):
    pass


class InteractionRegistry:
    """Validated read-only access to scene interaction metadata."""

    def __init__(self, path: str | Path, *, scene_path: str | Path | None = None) -> None:
        self.path = Path(path).expanduser().resolve()
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        self.document = InteractionRegistryDocument.model_validate(raw)
        self.data = self.document.model_dump(exclude_none=True)
        self._scene_path_override = Path(scene_path).resolve() if scene_path else None
        self._validate_semantics()

    @property
    def scene_path(self) -> Path:
        if self._scene_path_override is not None:
            return self._scene_path_override
        value = Path(self.document.scene)
        return (value if value.is_absolute() else self.path.parent / value).resolve()

    @property
    def objects(self) -> Mapping[str, Any]:
        return self.data["objects"]

    @property
    def named_targets(self) -> Mapping[str, Any]:
        return self.data.get("named_targets", {})

    def object(self, key: str) -> dict[str, Any]:
        try:
            return deepcopy(self.objects[key])
        except KeyError as exc:
            raise InteractionRegistryError(f"scene has no object {key!r}") from exc

    def action_request(self, key: str, action: str) -> dict[str, Any]:
        try:
            return deepcopy(self.objects[key]["action_requests"][action])
        except KeyError as exc:
            raise InteractionRegistryError(f"object {key!r} does not support {action!r}") from exc

    def _validate_semantics(self) -> None:
        for key, obj in self.objects.items():
            if not isinstance(obj, dict) or not obj.get("aliases"):
                raise InteractionRegistryError(f"object {key!r} requires aliases")
            spatial = obj.get("spatial")
            if not isinstance(spatial, dict) or not spatial.get("source") or not spatial.get("reference_pose"):
                raise InteractionRegistryError(f"object {key!r} requires spatial source and reference pose")
            anchors = spatial.get("anchors")
            if not isinstance(anchors, dict) or spatial.get("default_anchor") not in anchors:
                raise InteractionRegistryError(f"object {key!r} has invalid default anchor")
            if not spatial.get("tool_orientation"):
                raise InteractionRegistryError(f"object {key!r} requires tool orientation")
        for owner, obj in self.objects.items():
            for action, spec in obj.get("affordances", {}).items():
                acting = str(spec.get("acting_target", owner))
                contact = spec.get("contact_target")
                if acting not in self.objects or contact not in self.objects:
                    raise InteractionRegistryError(f"object {owner!r} affordance {action!r} references unknown object")
                if action not in self.objects[acting].get("action_requests", {}):
                    raise InteractionRegistryError(f"acting target {acting!r} lacks {action!r} request")


# Requested public name.
InteractionRegistryModel = InteractionRegistryDocument
