from __future__ import annotations

import threading
import time
from pathlib import Path

import mujoco.viewer

from robot_agent_control.contracts import load_command_document
from robot_agent_control.session import ControlSession


ROOT = Path(__file__).resolve().parents[1]
COMMANDS = ROOT / "demo" / "skill_command" / "config_001.commands.json"


class _FakeViewer:
    def __init__(self):
        self.cam = type("Camera", (), {"lookat": [0.0, 0.0, 0.0], "distance": 1.0, "azimuth": 0.0, "elevation": 0.0})()
        self._running = True
        self._sync_lock = threading.Lock()
        self.concurrent_sync = False
        self.sync_count = 0

    def is_running(self):
        return self._running

    def sync(self):
        if not self._sync_lock.acquire(blocking=False):
            self.concurrent_sync = True
            return
        try:
            self.sync_count += 1
            time.sleep(0.001)
        finally:
            self._sync_lock.release()

    def stop(self):
        self._running = False


class _FakeViewerContext:
    def __init__(self, viewer):
        self.viewer = viewer

    def __enter__(self):
        return self.viewer

    def __exit__(self, *_args):
        self.viewer.stop()


def test_auto_viewer_sync_does_not_race_execution(monkeypatch, tmp_path):
    viewer = _FakeViewer()
    monkeypatch.setattr(mujoco.viewer, "launch_passive", lambda *_args, **_kwargs: _FakeViewerContext(viewer))
    monkeypatch.setattr("robot_agent_control.session.ControlExecutor._configure_camera", lambda *_args, **_kwargs: None)
    document = load_command_document(COMMANDS)
    command = document.commands[-1].model_copy(update={"command_id": "home-command", "source_skill_step_id": "step-1"})
    session = ControlSession(document.model_copy(update={"commands": [command]}), headless=False, viewer_mode="auto")
    try:
        time.sleep(0.01)
        report = session.execute(session.document, output_dir=tmp_path)
        assert report.success
        assert viewer.sync_count > 0
        assert not viewer.concurrent_sync
    finally:
        session.close()
    assert session._viewer_thread is None

