from pathlib import Path


ROOT = Path(__file__).parents[2]


def test_readme_references_existing_entrypoints_without_old_install_recipe():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for relative in (
        "packages/robot_agent_protocol", "packages/robot_agent_sim",
        "packages/robot_agent_control", "docs/architecture.md",
        "docs/deployment.md", "scripts/test_all.sh",
    ):
        assert (ROOT / relative).exists()
    assert "pip install -e packages/robot_agent_protocol" in readme
    assert "pip install -e /home/cscvlab/lht/robot-agent-control" not in readme
    assert "双仓库系统" not in readme
