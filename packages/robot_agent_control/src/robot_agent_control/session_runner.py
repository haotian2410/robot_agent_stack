"""JSONL subprocess entry point for a persistent ControlSession."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from robot_agent_protocol import ExecutionBundle, ProcessFailure, validate_bundle_consistency

from .session import ControlSession


def _reply(request_id, *, ok=True, **payload):
    result = {"id": request_id, "ok": ok, **payload}
    print(json.dumps(result, ensure_ascii=False), flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Persistent robot-agent MuJoCo session runner")
    parser.add_argument("--viewer-mode", choices=("auto", "step", "headless"), default="headless")
    args = parser.parse_args()
    session: ControlSession | None = None
    try:
        for line in sys.stdin:
            if not line.strip():
                continue
            request = json.loads(line)
            request_id = request.get("id")
            operation = request.get("op")
            try:
                if operation == "OPEN_SESSION":
                    if session is not None:
                        session.close()
                    bundle = ExecutionBundle.model_validate_json(Path(request["bundle"]).read_text(encoding="utf-8"))
                    validate_bundle_consistency(bundle)
                    session = ControlSession(bundle.commands, headless=args.viewer_mode == "headless", viewer_mode=args.viewer_mode)
                    _reply(request_id, snapshot=session.snapshot())
                elif operation == "EXECUTE_BUNDLE":
                    if session is None:
                        raise RuntimeError("session is not open")
                    bundle = ExecutionBundle.model_validate_json(Path(request["bundle"]).read_text(encoding="utf-8"))
                    validate_bundle_consistency(bundle)
                    report = session.execute(bundle.commands, output_dir=bundle.task_dir)
                    _reply(request_id, report=report.model_dump(mode="json"), snapshot=session.snapshot())
                elif operation == "SNAPSHOT":
                    if session is None:
                        raise RuntimeError("session is not open")
                    _reply(request_id, snapshot=session.snapshot())
                elif operation == "OBSERVE":
                    if session is None:
                        raise RuntimeError("session is not open")
                    _reply(request_id, observation=session.observe(request["output_dir"]), snapshot=session.snapshot())
                elif operation == "RELOAD_SCENE":
                    if session is None:
                        raise RuntimeError("session is not open")
                    bundle = ExecutionBundle.model_validate_json(Path(request["bundle"]).read_text(encoding="utf-8"))
                    validate_bundle_consistency(bundle)
                    _reply(request_id, snapshot=session.reload(bundle.commands), reloaded=True)
                elif operation == "CLOSE_SESSION":
                    if session is not None:
                        session.close()
                        session = None
                    _reply(request_id, closed=True)
                    return 0
                else:
                    raise ValueError(f"unsupported session operation: {operation!r}")
            except Exception as exc:
                _reply(request_id, ok=False, error={"error_code": "SESSION_ERROR", "error_message": str(exc)})
    except Exception as exc:
        print(ProcessFailure(error_code="SESSION_ERROR", error_message=str(exc)).model_dump_json(), file=sys.stderr, flush=True)
        return 1
    finally:
        if session is not None:
            session.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
