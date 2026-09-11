from __future__ import annotations

from pathlib import Path

from robot_agent_protocol import ExecutionBundle, validate_bundle_consistency


def execute_bundle(
    bundle: ExecutionBundle | str | Path,
    *,
    viewer_mode: str = "headless",
):
    from robot_agent_control import ControlExecutor

    if not isinstance(bundle, ExecutionBundle):
        bundle = ExecutionBundle.model_validate_json(
            Path(bundle).read_text(encoding="utf-8")
        )
    validate_bundle_consistency(bundle)
    return ControlExecutor().execute(
        bundle.commands,
        viewer_mode=viewer_mode,
        output_dir=bundle.task_dir,
    )
