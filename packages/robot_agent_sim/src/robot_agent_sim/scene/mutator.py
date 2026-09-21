"""Topology-changing mutations for generated SceneSession scenes."""

from __future__ import annotations

from pathlib import Path

from ..assets.registry import AssetRegistry
from ..backends.mujoco.backend import MujocoSceneBackend
from ..execution.interaction_registry_builder import build_generated_registry
from .placement import PlacementSolver
from .registry import SceneObject, SceneRegistry


class SceneMutator:
    def __init__(self, assets: AssetRegistry | None = None) -> None:
        self.assets = assets or AssetRegistry()
        self.backend = MujocoSceneBackend()
        self.placement = PlacementSolver()

    def add(
        self,
        registry: SceneRegistry,
        *,
        semantic_name: str,
        category: str,
        object_id: str,
        relation: str,
        reference_object: str,
        current_positions: dict[str, tuple[float, float, float]],
        output_dir: str | Path,
    ) -> tuple[SceneRegistry, Path, Path]:
        existing = [item.model_copy(update={"position": current_positions.get(item.object_id, item.position)}) for item in registry.objects]
        reference = next((item for item in existing if item.object_id == reference_object), None)
        if reference is None:
            raise ValueError(f"scene edit reference not found: {reference_object}")
        asset = self.assets.resolve(category, semantic_name, [semantic_name])
        dimensions = asset.dimensions_m or (0.08, 0.04, 0.04)
        reference_dimensions = reference.dimensions_m or (0.18, 0.18, 0.10)
        position = self.placement.place_relative(relation, reference.position, dimensions, reference_dimensions, existing)
        existing.append(SceneObject(
            object_id=object_id,
            body_name=object_id,
            role="task_object",
            semantic_name=semantic_name,
            position=position,
            dimensions_m=dimensions,
            model_id=asset.model_id,
            model_name=asset.model_name,
            source="generated",
        ))
        updated = registry.model_copy(update={"objects": existing, "bindings": dict(registry.bindings)})
        records = {record.model_id: record for record in self.assets.records}
        object_assets = {item.object_id: records[item.model_id] for item in updated.objects if item.model_id in records}
        if len(object_assets) != len(updated.objects):
            missing = [item.object_id for item in updated.objects if item.object_id not in object_assets]
            raise ValueError(f"scene edit assets unavailable: {missing}")
        out = Path(output_dir).resolve()
        scene = self.backend.compose(updated, object_assets, out)
        interaction = build_generated_registry(scene, updated, out / "interaction_registry.json")
        (out / "scene_registry.json").write_text(updated.model_dump_json(indent=2), encoding="utf-8")
        return updated, scene, interaction

    def remove(self, registry: SceneRegistry, object_id: str, *, current_positions: dict[str, tuple[float, float, float]], output_dir: str | Path) -> tuple[SceneRegistry, Path, Path]:
        remaining = [item.model_copy(update={"position": current_positions.get(item.object_id, item.position)}) for item in registry.objects if item.object_id != object_id]
        if len(remaining) == len(registry.objects):
            raise ValueError(f"scene edit object not found: {object_id}")
        updated = registry.model_copy(update={"objects": remaining, "bindings": {k: v for k, v in registry.bindings.items() if v != object_id}})
        records = {record.model_id: record for record in self.assets.records}
        object_assets = {item.object_id: records[item.model_id] for item in updated.objects if item.model_id in records}
        out = Path(output_dir).resolve()
        scene = self.backend.compose(updated, object_assets, out)
        interaction = build_generated_registry(scene, updated, out / "interaction_registry.json")
        (out / "scene_registry.json").write_text(updated.model_dump_json(indent=2), encoding="utf-8")
        return updated, scene, interaction

