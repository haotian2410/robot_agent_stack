"""Small process boundary for MuJoCo control execution."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from robot_agent_protocol import ErrorCode, ProcessFailure


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--viewer-mode", choices=("auto", "step", "headless"), default="auto")
    args = parser.parse_args()

    try:
        from .control_adapter import execute_bundle

        report = execute_bundle(args.bundle, viewer_mode=args.viewer_mode)
        if not hasattr(report, "commands_completed"):
            print(report.model_dump_json())
            return 1
    except Exception as exc:  # the child must return a stable process envelope
        print(ProcessFailure(error_code=ErrorCode.INVALID_REQUEST, error_message=str(exc)).model_dump_json())
        return 1
    task_dir = args.bundle.resolve().parent
    try:
        commands = json.loads((task_dir / "commands.json").read_text(encoding="utf-8")).get("commands", [])
        traces = {item.get("skill_step_id"): item.get("operation_id") for item in json.loads((task_dir / "compiled_step_trace.json").read_text(encoding="utf-8"))}
        operation_ids = []
        for command in commands:
            step_id = str(command.get("source_skill_step_id", ""))
            operation_id = traces.get(step_id, step_id)
            if operation_id and operation_id not in operation_ids:
                operation_ids.append(operation_id)
        batch = {
            "status": "success" if report.success else "partial_failure",
            "total": len(operation_ids), "completed": len(operation_ids) if report.success else 0,
            "failed": 0 if report.success else 1, "skipped": 0,
            "subtasks": [{"operation_id": operation_id, "status": "succeeded" if report.success else "failed"} for operation_id in operation_ids],
        }
        (task_dir / "batch_execution_report.json").write_text(json.dumps(batch, ensure_ascii=False, indent=2), encoding="utf-8")
    except (OSError, ValueError, json.JSONDecodeError):
        pass
    print(json.dumps({
        "success": report.success,
        "commands_completed": report.commands_completed,
        "commands_total": report.commands_total,
        "runtime_steps": len(report.steps),
        "report": str(task_dir / "execution_report.json"),
    }, ensure_ascii=False))
    return 0 if report.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
