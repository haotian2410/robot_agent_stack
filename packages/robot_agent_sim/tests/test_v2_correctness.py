import random
import pytest
import mujoco
from robot_agent_sim.assets.registry import AssetRecord
from robot_agent_sim.contracts.task_intent import Operation, TaskEntity, TaskIntent, TaskStatus, TaskType
from robot_agent_sim.scene.directions import normalize_direction
from robot_agent_sim.scene.composer import SceneComposer
from robot_agent_sim.backends.mujoco.backend import MujocoSceneBackend

@pytest.mark.parametrize('word,expected', [('东','right'),('西','left'),('南','back'),('北','front')])
def test_direction_alias(word, expected):
    assert normalize_direction(word) == expected

@pytest.mark.parametrize('relation', ['above', 'below'])
def test_vertical_pair_preserved(relation):
    composer = SceneComposer()
    a, b = composer._relation_pair_positions(relation, (.06,)*3, (.06,)*3)
    intent = TaskIntent(
        status=TaskStatus.ACCEPTED,
        instruction="定位对象",
        task_types=[TaskType.LOCATE],
        entities=[TaskEntity(entity_id="a", semantic_name="object a", category="cube")],
        operations=[Operation(operation_id="op-1", task_type=TaskType.LOCATE, target="a")],
    )
    a = composer._position('a', (.06,)*3, [], random.Random(7), intent, a)
    b = composer._position('b', (.06,)*3, [], random.Random(7), intent, b)
    assert (a[2] > b[2]) if relation == 'above' else (a[2] < b[2])


def test_chinese_entity_names_keep_unique_generated_ids():
    intent = TaskIntent(
        status=TaskStatus.ACCEPTED,
        instruction="把红色方块放进蓝色盒子",
        task_types=[TaskType.PICK_AND_PLACE],
        entities=[
            TaskEntity(entity_id="red_block", semantic_name="红色方块", category="object"),
            TaskEntity(entity_id="blue_box", semantic_name="蓝色盒子", category="container"),
        ],
        operations=[
            Operation(
                operation_id="op-1",
                task_type=TaskType.PICK_AND_PLACE,
                source="red_block",
                destination="blue_box",
            )
        ],
    )
    assets = {
        "red_block": AssetRecord(
            model_id="cube", model_name="cube_basic", category="cube",
            source="primitive", dimensions_m=(0.06, 0.06, 0.06),
        ),
        "blue_box": AssetRecord(
            model_id="box", model_name="open_box", category="container",
            source="primitive", dimensions_m=(0.18, 0.18, 0.10),
        ),
    }

    registry = SceneComposer().compose(intent, assets, robot="ur5e", seed=0)

    assert registry.bindings == {
        "red_block": "red_block_01",
        "blue_box": "blue_box_01",
    }
    assert [item.body_name for item in registry.objects] == [
        "red_block_01",
        "blue_box_01",
    ]


def test_generated_panda_tabletop_contact_plane_is_zero(tmp_path):
    intent = TaskIntent(
        status=TaskStatus.ACCEPTED,
        instruction="抓取方块",
        task_types=[TaskType.GRASP],
        entities=[TaskEntity(entity_id="cube", semantic_name="cube", category="cube")],
        operations=[Operation(operation_id="op-1", task_type=TaskType.GRASP, target="cube")],
    )
    assets = {
        "cube": AssetRecord(model_id="cube", model_name="cube_basic", category="cube", source="primitive", dimensions_m=(0.06, 0.06, 0.06)),
    }
    registry = SceneComposer().compose(intent, assets, robot="panda", seed=0)
    scene = MujocoSceneBackend().compose(registry, assets, tmp_path)
    model = mujoco.MjModel.from_xml_path(str(scene))
    geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "work_table_top")
    assert geom_id >= 0
    assert float(model.geom_pos[geom_id][2] + model.geom_size[geom_id][2]) == pytest.approx(0.0)
