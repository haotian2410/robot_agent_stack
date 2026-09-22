"""Client for the persistent control subprocess."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


class SessionExecutorClient:
    def __init__(self, *, viewer_mode: str = "headless") -> None:
        env = os.environ.copy()
        if viewer_mode == "headless":
            env["MUJOCO_GL"] = "egl"
            env["PYOPENGL_PLATFORM"] = "egl"
        else:
            env.pop("MUJOCO_GL", None)
            env.pop("PYOPENGL_PLATFORM", None)
        self.process = subprocess.Popen(
            [sys.executable, "-m", "robot_agent_control.session_runner", "--viewer-mode", viewer_mode],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            text=True,
            bufsize=1,
            env=env,
        )
        self._next_id = 1
        self.closed = False

    def request(self, operation: str, **payload: Any) -> dict[str, Any]:
        if self.closed or self.process.stdin is None or self.process.stdout is None:
            raise RuntimeError("session executor is closed")
        request_id = self._next_id
        self._next_id += 1
        self.process.stdin.write(json.dumps({"id": request_id, "op": operation, **payload}, ensure_ascii=False) + "\n")
        self.process.stdin.flush()
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError(f"session runner exited with code {self.process.poll()}")
        response = json.loads(line)
        if not response.get("ok", False):
            error = response.get("error", {})
            raise RuntimeError(error.get("error_message", "session runner failed"))
        return response

    def open(self, bundle: str | Path) -> dict[str, Any]:
        return self.request("OPEN_SESSION", bundle=str(Path(bundle).resolve()))

    def execute(self, bundle: str | Path) -> dict[str, Any]:
        return self.request("EXECUTE_BUNDLE", bundle=str(Path(bundle).resolve()))

    def snapshot(self) -> dict[str, Any]:
        return self.request("SNAPSHOT")

    def observe(self, output_dir: str | Path) -> dict[str, Any]:
        return self.request("OBSERVE", output_dir=str(Path(output_dir).resolve()))

    def reload(self, bundle: str | Path) -> dict[str, Any]:
        return self.request("RELOAD_SCENE", bundle=str(Path(bundle).resolve()))

    def reload_scene(self, scene: str | Path, registry: str | Path, *, robot: str | None = None) -> dict[str, Any]:
        return self.request("RELOAD_SCENE_STATE", scene=str(Path(scene).resolve()), registry=str(Path(registry).resolve()), robot=robot)

    def close(self) -> None:
        if not self.closed:
            try:
                self.request("CLOSE_SESSION")
            finally:
                self.closed = True
                self.process.wait(timeout=10)


__all__ = ["SessionExecutorClient"]
