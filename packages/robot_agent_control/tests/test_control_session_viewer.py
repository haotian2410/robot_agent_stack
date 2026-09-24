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


def test_step_viewer_continue_resumes_execution(monkeypatch, tmp_path):
    viewer = _FakeViewer()
    monkeypatch.setattr(mujoco.viewer, "launch_passive", lambda *_args, **_kwargs: _FakeViewerContext(viewer))
    monkeypatch.setattr("robot_agent_control.session.ControlExecutor._configure_camera", lambda *_args, **_kwargs: None)
    document = load_command_document(COMMANDS)
    command = document.commands[-1].model_copy(update={"command_id": "home-command", "source_skill_step_id": "step-1"})
    session = ControlSession(document.model_copy(update={"commands": [command]}), headless=False, viewer_mode="step")
    try:
        # The home move uses the runtime's minimum playback duration; set the
        # key event after the command has had time to reach the STEP wait.
        threading.Timer(6.0, session._continue_event.set).start()
        report = session.execute(session.document, output_dir=tmp_path)
        assert report.success
    finally:
        session.close()


def test_step_viewer_close_cancels_waiting_execution(monkeypatch, tmp_path):
    viewer = _FakeViewer()
    monkeypatch.setattr(mujoco.viewer, "launch_passive", lambda *_args, **_kwargs: _FakeViewerContext(viewer))
    monkeypatch.setattr("robot_agent_control.session.ControlExecutor._configure_camera", lambda *_args, **_kwargs: None)
    document = load_command_document(COMMANDS)
    command = document.commands[-1].model_copy(update={"command_id": "home-command", "source_skill_step_id": "step-1"})
    session = ControlSession(document.model_copy(update={"commands": [command]}), headless=False, viewer_mode="step")
    try:
        threading.Timer(0.5, viewer.stop).start()
        report = session.execute(session.document, output_dir=tmp_path)
        assert not report.success
        assert report.failure.error_code == "EXECUTION_CANCELLED_BY_USER"
    finally:
        session.close()


def test_viewer_reload_replaces_old_viewer_and_close_cleans_up(monkeypatch):
    viewers = []

    def launch(*_args, **_kwargs):
        viewer = _FakeViewer()
        viewers.append(viewer)
        return _FakeViewerContext(viewer)

    monkeypatch.setattr(mujoco.viewer, "launch_passive", launch)
    monkeypatch.setattr("robot_agent_control.session.ControlExecutor._configure_camera", lambda *_args, **_kwargs: None)
    document = load_command_document(COMMANDS)
    session = ControlSession(document, headless=False, viewer_mode="auto")
    try:
        session.reload_scene(document.scene, document.registry, robot=document.robot)
        assert len(viewers) == 2
        assert not viewers[0].is_running()
        assert viewers[1].sync_count > 0
    finally:
        session.close()
    assert not viewers[1].is_running()
    assert session._viewer is None
    assert session._viewer_thread is None
