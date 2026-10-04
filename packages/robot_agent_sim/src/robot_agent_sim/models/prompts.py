from __future__ import annotations

import json
from importlib.resources import files


def _load_prompt(filename: str) -> str:
    """Load a checked-in prompt template from the installed package."""
    content = files("robot_agent_sim.models.prompt_templates").joinpath(filename).read_text(encoding="utf-8")
    if not content.strip():
        raise RuntimeError(f"prompt template is empty: {filename}")
    return content


# The text templates are the only runtime source of truth. Missing package
# data is an installation error rather than a silent fallback to stale text.
TASK_UNDERSTANDING_PROMPT = _load_prompt("task_understanding_v2.txt")
VISION_GROUNDING_PROMPT = _load_prompt("vision_grounding_v1.txt")
SKILL_PLANNING_PROMPT = _load_prompt("skill_planning_v2.txt")


def prompt_payload(value) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
