"""Compatibility imports for contracts now owned by robot_agent_protocol."""
from __future__ import annotations
import json
from pathlib import Path
from robot_agent_protocol import (CommandDocument, ExecutionFailure, ExecutionOptions,
    ExecutionReport, RuntimeStepReport, SkillCommand, ViewerMode, scene_sha256)
from robot_agent_protocol import load_command_document as load_v1_command_document
from robot_agent_protocol import load_legacy_command_document

def load_command_document(path: str | Path) -> CommandDocument:
    """Dispatch legacy fixtures explicitly; formal v1 remains strict."""
    command_path = Path(path).expanduser().resolve()
    raw = json.loads(command_path.read_text(encoding="utf-8"))
    if isinstance(raw, dict) and raw.get("schema_version") == "1.0":
        return load_v1_command_document(command_path)
    return load_legacy_command_document(command_path)

__all__ = ["CommandDocument", "ExecutionFailure", "ExecutionOptions", "ExecutionReport",
    "RuntimeStepReport", "SkillCommand", "ViewerMode", "load_command_document", "scene_sha256"]
