from pathlib import Path

import pytest
from typer.testing import CliRunner

from robot_agent_sim import cli
from robot_agent_sim.pipeline.engine import PipelineResult

runner = CliRunner()


@pytest.fixture
def calls(monkeypatch):
    received = []

    class FakeEngine:
        def plan(self, instruction, **kwargs):
            received.append((instruction, kwargs))
            return PipelineResult(task_intent={"status": "accepted"})

    monkeypatch.setattr(cli, "_new_engine", lambda *args, **kwargs: FakeEngine())
    return received


def test_plan_subcommand_preserves_instruction_and_options(calls, tmp_path):
    instruction = "把左边的红色方块放进右边蓝色盒子"
    result = runner.invoke(cli.app, [
        "plan", instruction, "--robot", "panda", "--seed", "7",
        "--output-dir", str(tmp_path),
    ])
    assert result.exit_code == 0, result.output
    assert calls == [(instruction, {
        "robot": "panda", "scene": None, "seed": 7, "output_dir": tmp_path, "planner": "recipe",
        "interaction_registry": None,
    })]


def test_uploaded_scene_options(calls, tmp_path):
    scene = tmp_path / "scene.xml"
    scene.write_text("<mujoco/>")
    result = runner.invoke(cli.app, ["plan", "按按钮", "--robot", "ur5e", "--scene", str(scene)])
    assert result.exit_code == 0, result.output
    assert calls[0][1] == {
        "robot": "ur5e", "scene": scene, "seed": 0, "output_dir": Path("var"), "planner": "recipe",
        "interaction_registry": None,
    }


@pytest.mark.parametrize("args", [
    ["plan", "任务", "--robot", "invalid"],
    ["plan", "任务", "--seed", "seven"],
    ["plan", "任务", "--scene", "missing-scene.xml"],
    ["plan", "   "],
])
def test_invalid_arguments_do_not_start_pipeline(calls, args):
    result = runner.invoke(cli.app, args)
    assert result.exit_code == 2
    assert not calls


def test_unsupported_scene_extension(calls, tmp_path):
    path = tmp_path / "scene.txt"
    path.write_text("<mujoco/>")
    result = runner.invoke(cli.app, ["plan", "任务", "--scene", str(path)])
    assert result.exit_code == 2
    assert not calls


@pytest.mark.parametrize("args", [["--help"], ["plan", "--help"]])
def test_help_does_not_start_pipeline(calls, args):
    result = runner.invoke(cli.app, args)
    assert result.exit_code == 0
    assert "plan" in result.output
    assert not calls


def test_pipeline_error_is_readable(monkeypatch):
    class FailingEngine:
        def plan(self, *args, **kwargs):
            raise ValueError("scene XML contains forbidden include/plugin")

    monkeypatch.setattr(cli, "_new_engine", lambda *args, **kwargs: FailingEngine())
    result = runner.invoke(cli.app, ["plan", "任务"])
    assert result.exit_code == 1
    assert "规划失败" in result.output
    assert "Traceback" not in result.output


def test_run_prints_planning_summary_before_compile_failure(monkeypatch, tmp_path):
    planned = PipelineResult(
        task_intent={"task_types": ["pick_and_place"]},
        grounded_task={"entities": [{"entity_id": "baseball_01", "object_id": "baseball_01", "grounding_method": "asset_scene_binding"}]},
        skill_plan={"steps": [{"step_id": "step-1", "skill_name": "locate", "target_object": "baseball_01"}]},
        model_call_count=2,
        model_usage={"stages": [{"stage": "task_understanding", "prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14, "finish_reason": "stop"}]},
        planner="qwen", route="B", source_scene="scene.xml",
    )
    monkeypatch.setattr(cli, "_plan", lambda *args, **kwargs: planned)
    monkeypatch.setattr(cli, "compile_directory", lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("compile failed")))
    result = runner.invoke(cli.app, ["run", "抓取棒球", "--output-dir", str(tmp_path)])
    assert result.exit_code == 1
    assert "语义子任务顺序" in result.output
    assert "Atomic Skill 调用顺序：locate" in result.output
    assert "模型调用次数：2" in result.output
    assert "规划器：qwen / Route B" in result.output
    assert "模型调用明细" in result.output
    assert "运行准备失败：compile failed" in result.output
