from pathlib import Path

from robot_agent_protocol import validate_bundle_consistency
from robot_agent_sim.pipeline.engine import PipelineEngine
from robot_agent_sim.execution.compiler import compile_directory


def test_skill_plan_compiler_control_contract(tmp_path):
    result = PipelineEngine().plan(
        "把红色方块放进蓝色盒子", robot="ur5e", output_dir=tmp_path
    )
    assert result.status == "accepted"
    bundle = compile_directory(tmp_path)
    validate_bundle_consistency(bundle)
    assert Path(bundle.commands).is_file()
    assert bundle.scene_fingerprint
