from __future__ import annotations

import re

from ..contracts.task_intent import Operation, QuantityMode, SpatialRelationType, TaskType
from ..contracts.turn import (
    SceneEditIntent, SceneEditRelation, SceneEditType, SceneQueryIntent, SceneQueryType,
    SessionControlIntent, SessionControlType, TurnKind,
)
from .skill_planning import LLMOperationPlan, LLMPlanStep, SkillPlanLLMOutput
from .task_understanding import ParseEntity, ParseOperation, ParseRelation, TaskParseLLMOutput
from .vision_grounding import VisionCandidate, VisionLLMOutput
from ..planning.recipes import RECIPE_DEFINITIONS


class FakeTaskUnderstandingProvider:
    """Offline parser used by tests and the CLI default; no model loop/retry."""

    def understand(self, request):
        text = request.instruction.strip()
        low = text.casefold()
        control = (
            SessionControlType.PAUSE if text in {"暂停", "暂停执行", "pause"}
            else SessionControlType.RESUME if text in {"继续", "恢复", "恢复执行", "resume"}
            else SessionControlType.CLOSE if text in {"关闭会话", "结束会话", "close"}
            else None
        )
        if control is not None:
            return TaskParseLLMOutput(
                status="accepted",
                turn_kind=TurnKind.SESSION_CONTROL,
                session_control=SessionControlIntent(action=control),
            )
        # The offline provider mirrors the single Task Understanding contract
        # used by Qwen.  SceneSession no longer carries its own turn-kind
        # parser, so switching providers cannot change session routing.
        edit_operation = (
            SceneEditType.ADD if any(token in text for token in ("增加", "添加", "加一个", "加一只"))
            else SceneEditType.REMOVE if any(token in text for token in ("删除", "移除"))
            else None
        )
        if edit_operation is not None:
            assets = (("香蕉", "banana", "fruit"), ("苹果", "apple", "fruit"), ("棒球", "baseball", "ball"))
            matched = next(((name, category) for zh, name, category in assets if zh in text or name in low), None)
            if matched is None:
                return TaskParseLLMOutput(status="unsupported_task", turn_kind=TurnKind.SCENE_EDIT, raw_task=text)
            semantic_name, category = matched
            return TaskParseLLMOutput(
                status="accepted",
                turn_kind=TurnKind.SCENE_EDIT,
                scene_edit=SceneEditIntent(
                    operation=edit_operation,
                    semantic_name=semantic_name,
                    category=category,
                    relation=SceneEditRelation.LEFT_OF if "左" in text else SceneEditRelation.RIGHT_OF,
                    reference="basket" if "篮" in text else None,
                ),
            )
        existence_query = any(token in text for token in ("有没有", "是否有", "还在吗")) or ("有" in text and "吗" in text)
        if existence_query or any(token in text for token in ("几个", "多少", "数量", "在哪里", "状态")) or ("位置" in text and any(token in text for token in ("当前", "查询", "报告"))):
            query_type = SceneQueryType.EXISTENCE if existence_query else SceneQueryType.COUNT if any(token in text for token in ("几个", "多少", "数量")) else SceneQueryType.POSITION if any(token in text for token in ("在哪里", "位置")) else SceneQueryType.STATE
            labels = (("苹果", "apple", "fruit"), ("香蕉", "banana", "fruit"), ("棒球", "baseball", "ball"), ("篮子", "basket", "container"))
            match = next(((name, category) for zh, name, category in labels if zh in text or name in low), None)
            return TaskParseLLMOutput(status="accepted", turn_kind=TurnKind.SCENE_QUERY, scene_query=SceneQueryIntent(query_type=query_type, semantic_name=match[0] if match else None, category=match[1] if match else None, referent=any(token in text for token in ("它", "刚才那个", "这个"))))
        diagonals = ("东北", "东南", "西北", "西南", "左前方", "右前方", "斜上方", "左上方", "右下方", "northeast", "northwest", "southeast", "southwest", "diagonal")
        motion_words = ("移动", "移到", "往左", "往右", "往前", "往后", "往上", "往下", "move")
        # Composite directions are ambiguous only when they modify the
        # motion itself.  “左上角的棒球” is an entity selector and must be
        # parsed as spatial relations instead.
        if "东北角" in text:
            return TaskParseLLMOutput(status="direction_clarification_required", raw_direction="东北角")
        if any(token in low for token in diagonals) and any(token in low for token in motion_words):
            raw = next((token for token in diagonals if token in text or token in low), "diagonal")
            return TaskParseLLMOutput(status="direction_clarification_required", raw_direction=raw)
        if any(token in low for token in ("拧紧螺丝", "旋紧螺丝", "tighten screw")):
            return TaskParseLLMOutput(status="unsupported_task", raw_task=text)

        entities: list[ParseEntity] = []
        dialogue_match = re.search(r"\[dialogue_ref=([^\]]+)\]", text)
        dialogue_label = dialogue_match.group(1).strip() if dialogue_match else None
        dialogue_names = {
            "苹果": ("apple", "fruit"), "apple": ("apple", "fruit"),
            "香蕉": ("banana", "fruit"), "banana": ("banana", "fruit"),
            "棒球": ("baseball", "ball"), "baseball": ("baseball", "ball"),
            "篮子": ("basket", "container"), "basket": ("basket", "container"),
        }
        dialogue_spec = dialogue_names.get((dialogue_label or "").casefold())
        dialogue_subject_tokens = {
            "apple": ("苹果", "apple"), "banana": ("香蕉", "banana"),
            "baseball": ("棒球", "baseball"), "basket": ("篮子", "basket"),
        }
        dialogue_relative = None
        if dialogue_match and dialogue_spec:
            suffix = text[dialogue_match.end():]
            relation = next((value for token, value in (
                ("左边", SpatialRelationType.LEFT_OF), ("左侧", SpatialRelationType.LEFT_OF),
                ("右边", SpatialRelationType.RIGHT_OF), ("右侧", SpatialRelationType.RIGHT_OF),
                ("前面", SpatialRelationType.FRONT_OF), ("后面", SpatialRelationType.BEHIND),
            ) if suffix.startswith(token)), None)
            if relation is not None and any(
                token in suffix.casefold()
                for token in dialogue_subject_tokens.get(dialogue_spec[0], (dialogue_spec[0],))
            ):
                dialogue_relative = relation

        def quantity_for(token):
            match = re.search(r"(一|两|二|三|四|五|六|七|八|九|十|[1-9][0-9]*)个?" + re.escape(token), text)
            values = {"一": 1, "两": 2, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}
            if not match:
                return 1
            value = match.group(1)
            return values[value] if value in values else int(value)

        def quantity_mode_for(token):
            if any(marker in text for marker in ("都", "全部", "每个", "每只", "each", "all")):
                return QuantityMode.ALL
            count = quantity_for(token)
            if count > 1 and any(marker in text for marker in ("中", "其中")):
                return QuantityMode.CANDIDATE_POOL
            return QuantityMode.ALL if count > 1 else QuantityMode.SINGLE

        def add(eid, name, category, color=None, count=1, quantity_mode=QuantityMode.SINGLE, dialogue_ref=False):
            existing = next((entity for entity in entities if entity.id == eid), None)
            if existing is None:
                entities.append(ParseEntity(id=eid, name=name, category=category, dialogue_ref=dialogue_ref, color=color, count=count, quantity_mode=quantity_mode))
            elif dialogue_ref:
                existing.dialogue_ref = True

        if "红" in text or "red" in low:
            if "球" in text or "ball" in low:
                add("red_ball_01", "red ball", "ball", "red")
            else:
                add("red_cube_01", "red cube", "cube", "red")
        if ("蓝" in text or "blue" in low) and not any(
            token in text or token in low
            for token in ("柜门", "上层", "cabinet door", "upper compartment")
        ):
            add("blue_box_01", "blue box", "container", "blue")
        if "黄" in text or "yellow" in low: add("yellow_cube_01", "yellow cube", "cube", "yellow")
        if ("球" in text and "棒球" not in text) or re.search(r"\bball\b", low):
            if not any(entity.category == "ball" for entity in entities):
                add("ball_01", "ball", "ball")
        if ("盒" in text or "box" in low) and not any(entity.category == "container" for entity in entities):
            add("open_box_01", "open box", "container", None)
        if "按钮" in text or "button" in low: add("button_01", "button", "button")
        if "柜门" in text or "cabinet door" in low:
            add("cabinet_door_01", "blue cabinet door", "door", "blue")
            add("cabinet_handle_01", "blue cabinet handle", "handle", "blue")
        if "上层" in text or "upper compartment" in low:
            add(
                "upper_compartment_01",
                "blue cabinet upper compartment",
                "container",
                "blue",
            )
        if dialogue_relative is not None and dialogue_spec is not None:
            name, category = dialogue_spec
            add(f"{name}_01", name, category)
            add(f"{name}_dialogue_ref", name, category, dialogue_ref=True)
        for token, name, category in (("苹果", "apple", "fruit"), ("香蕉", "banana", "fruit"), ("棒球", "baseball", "ball"), ("篮子", "basket", "container"), ("杯子", "cup", "container"), ("魔方", "rubiks cube", "cube"), ("海绵", "sponge", "sponge"), ("勺子", "spoon", "utensil"), ("糖盒", "sugar box", "package")):
            if token in text or token in low:
                add(
                    f"{name.replace(' ', '_')}_01", name, category,
                    count=quantity_for(token), quantity_mode=quantity_mode_for(token),
                    dialogue_ref=bool(dialogue_spec and dialogue_spec[0] == name and dialogue_relative is None),
                )
        if "螺丝" in text or "screw" in low:
            add("screw_01", "screw", "screw")
        for label in ("a", "b", "c"):
            # Chinese characters are adjacent to the Latin labels, so ``\b``
            # does not create a boundary here.  Restrict the lookaround to
            # ASCII letters instead and still avoid matching words.
            if re.search(rf"(?<![a-z]){label}(?![a-z])", low):
                add(f"{label}_01", label, "object")
        if not entities: add("target_01", "target object", "cube")

        relations: list[ParseRelation] = []
        put = any(token in text for token in ("放进", "放入", "放到", "放在")) or "put" in low
        has_left = any(token in low for token in ("左", "西", "left", "west"))
        has_right = any(token in low for token in ("右", "东", "right", "east"))
        pure_motion = any(token in low for token in ("向左", "向右", "向前", "向后", "向上", "向下", "往左", "往右", "往前", "往后", "往上", "往下", "move left", "move right", "move front", "move back", "move up", "move down"))
        spatial_selector_subjects: set[str] = set()
        if dialogue_relative is not None and dialogue_spec is not None:
            name, _ = dialogue_spec
            subject_id = f"{name}_01"
            reference_id = f"{name}_dialogue_ref"
            relations.append(ParseRelation(
                scope="selection", subject=subject_id,
                relation=dialogue_relative, reference=reference_id,
            ))
            spatial_selector_subjects.add(subject_id)
        # Entity corner selectors are planar: 上/下 in 左上/左下 means
        # front/back, never the world Z axis.  Resolve each selector against
        # the nearest entity name so source and destination can use different
        # corners in one instruction.
        corner_relations = {
            "左上角": (SpatialRelationType.LEFT, SpatialRelationType.FRONT),
            "右上角": (SpatialRelationType.RIGHT, SpatialRelationType.FRONT),
            "左下角": (SpatialRelationType.LEFT, SpatialRelationType.BACK),
            "右下角": (SpatialRelationType.RIGHT, SpatialRelationType.BACK),
        }
        baseball = next((entity for entity in entities if entity.id.startswith("baseball_")), None)
        aliases = {
            "苹果": "apple", "香蕉": "banana", "棒球": "baseball", "篮子": "basket",
            "banana": "banana", "baseball": "baseball", "basket": "basket",
        }
        for entity in entities:
            entity_aliases = [alias for alias, stem in aliases.items() if entity.id.startswith(stem + "_")]
            name_positions = [text.find(alias) for alias in entity_aliases if text.find(alias) >= 0]
            if not name_positions:
                continue
            name_pos = min(name_positions)
            preceding = [(text.rfind(corner, 0, name_pos), rels) for corner, rels in corner_relations.items()]
            preceding = [(pos, rels) for pos, rels in preceding if pos >= 0]
            if preceding:
                _, selected_relations = max(preceding, key=lambda item: item[0])
                relations.extend(ParseRelation(scope="selection", subject=entity.id, relation=rel) for rel in selected_relations)
                spatial_selector_subjects.add(entity.id)
        if baseball is not None and baseball.id not in spatial_selector_subjects:
            if "最左边" in text or "最左侧" in text:
                relations.append(ParseRelation(scope="selection", subject=baseball.id, relation=SpatialRelationType.LEFTMOST))
                spatial_selector_subjects.add(baseball.id)
            elif "最右边" in text or "最右侧" in text:
                relations.append(ParseRelation(scope="selection", subject=baseball.id, relation=SpatialRelationType.RIGHTMOST))
                spatial_selector_subjects.add(baseball.id)
            elif "左边" in text or "左侧" in text:
                relations.append(ParseRelation(scope="selection", subject=baseball.id, relation=SpatialRelationType.LEFT))
                spatial_selector_subjects.add(baseball.id)
            elif "右边" in text or "右侧" in text:
                relations.append(ParseRelation(scope="selection", subject=baseball.id, relation=SpatialRelationType.RIGHT))
                spatial_selector_subjects.add(baseball.id)
        if not spatial_selector_subjects and "左上方" in text and not pure_motion:
            subject = entities[0].id
            relations.extend([ParseRelation(scope="selection", subject=subject, relation=SpatialRelationType.LEFT), ParseRelation(scope="selection", subject=subject, relation=SpatialRelationType.FRONT)])
            spatial_selector_subjects.add(subject)
        if len(entities) > 1 and has_left and not spatial_selector_subjects and not pure_motion:
            relations.append(ParseRelation(scope="selection", subject=entities[0].id, relation=SpatialRelationType.LEFT_OF, reference=entities[1].id))
        elif len(entities) > 1 and has_right and not spatial_selector_subjects and not pure_motion:
            relations.append(ParseRelation(scope="selection", subject=entities[0].id, relation=SpatialRelationType.RIGHT_OF, reference=entities[1].id))
        elif has_left and not spatial_selector_subjects and not pure_motion:
            relations.append(ParseRelation(scope="selection", subject=entities[0].id, relation=SpatialRelationType.LEFT))
        elif has_right and not spatial_selector_subjects and not pure_motion:
            relations.append(ParseRelation(scope="selection", subject=entities[0].id, relation=SpatialRelationType.RIGHT))
        if "最近" in text or "靠近" in text or "nearest" in low:
            subject = next((entity for entity in entities if entity.color == "red"), entities[0])
            reference = next((entity for entity in entities if entity.color == "yellow"), None)
            reference = reference or next((entity for entity in entities if entity.category == "container" and entity.id != subject.id), entities[-1])
            relations.append(ParseRelation(scope="selection", subject=subject.id, relation=SpatialRelationType.NEAREST, reference=reference.id))
        if "最远" in text or "farthest" in low:
            relations.append(ParseRelation(scope="selection", subject=entities[0].id, relation=SpatialRelationType.FARTHEST, reference=entities[-1].id))
        if not spatial_selector_subjects:
            extreme_markers = (
                ("最左边", SpatialRelationType.LEFTMOST), ("最右边", SpatialRelationType.RIGHTMOST),
                ("最前面", SpatialRelationType.FRONTMOST), ("最后面", SpatialRelationType.BACKMOST),
                ("最高", SpatialRelationType.HIGHEST), ("最低", SpatialRelationType.LOWEST),
            )
            for marker, relation in extreme_markers:
                if marker in text:
                    relations.append(ParseRelation(scope="selection", subject=entities[0].id, relation=relation))
                    break
        ranking_relations = {
            SpatialRelationType.NEAREST, SpatialRelationType.FARTHEST,
            SpatialRelationType.LEFTMOST, SpatialRelationType.RIGHTMOST,
            SpatialRelationType.FRONTMOST, SpatialRelationType.BACKMOST,
            SpatialRelationType.HIGHEST, SpatialRelationType.LOWEST,
        }
        ranking_subjects = {
            relation.subject for relation in relations
            if relation.scope == "selection" and relation.relation in ranking_relations
        }
        for entity in entities:
            if entity.id in ranking_subjects:
                entity.quantity_mode = QuantityMode.CANDIDATE_POOL
        if not spatial_selector_subjects and not pure_motion:
            for tokens, relation in ((('前', 'north', 'front'), SpatialRelationType.FRONT), (('后', 'south', 'back'), SpatialRelationType.BACK), (('上', 'above', 'up'), SpatialRelationType.UP), (('下', 'below', 'down'), SpatialRelationType.DOWN)):
                if (
                    any(token in text or token in low for token in tokens)
                    and not (relation == SpatialRelationType.BACK and "然后" in text)
                    and not (relation == SpatialRelationType.UP and "上层" in text)
                ):
                    relations.append(ParseRelation(scope="selection", subject=entities[0].id, relation=relation))

        operations: list[ParseOperation] = []
        door = next((entity for entity in entities if entity.category == "door"), None)
        handle = next((entity for entity in entities if entity.category == "handle"), None)
        open_requested = any(token in text or token in low for token in ("打开", "开启", "open"))
        close_requested = any(token in text or token in low for token in ("关闭", "关上", "close"))
        if open_requested and door is not None and handle is not None:
            operations.append(
                ParseOperation(type="open", target=door.id, reference=handle.id)
            )
        # Preserve entity reuse in simple chained pick-and-place language such
        # as “把 A 放进 B，再把 B 放进 C”.  Roles are local to each operation.
        letters = [entity for entity in entities if entity.category == "object" and entity.id.endswith("_01")]
        chained = len(letters) >= 3 and text.count("放") >= 2
        if chained:
            operations.extend([
                ParseOperation(type="pick_and_place", source=letters[0].id, destination=letters[1].id),
                ParseOperation(type="pick_and_place", source=letters[1].id, destination=letters[2].id),
            ])
            relations.extend([
                ParseRelation(scope="goal", subject=letters[0].id, relation=SpatialRelationType.INSIDE, reference=letters[1].id),
                ParseRelation(scope="goal", subject=letters[1].id, relation=SpatialRelationType.INSIDE, reference=letters[2].id),
            ])
        if chained and any(token in text or token in low for token in ("按", "press")):
            operations.append(ParseOperation(type="press", target=letters[2].id))
        if chained:
            return TaskParseLLMOutput(status="accepted", entities=entities, operations=operations, relations=relations)
        if put:
            source = next((entity for entity in entities if entity.color == "red"), entities[0])
            destination = next((entity for entity in entities if entity.category == "container" and entity.id != source.id), None)
            if destination is None and len(entities) > 1: destination = entities[1]
            if destination is not None:
                if destination.category != "container" and has_right:
                    operations.append(ParseOperation(type="move", target=source.id, reference=destination.id))
                else:
                    operations.append(ParseOperation(type="pick_and_place", source=source.id, destination=destination.id))
                    relations.append(ParseRelation(scope="goal", subject=source.id, relation=SpatialRelationType.INSIDE, reference=destination.id))
        if close_requested and door is not None and handle is not None:
            operations.append(
                ParseOperation(type="close", target=door.id, reference=handle.id)
            )
        if any(token in text or token in low for token in ("按", "press")):
            button = next((entity for entity in entities if entity.category == "button"), entities[-1])
            operations.append(ParseOperation(type="press", target=button.id))
        elif not operations and any(token in text or token in low for token in ("移动", "移到", "move")):
            motion_target = entities[0].id
            if baseball is not None and spatial_selector_subjects and any(token in low for token in ("向右", "向左", "向前", "向后", "向上", "向下")):
                operations.append(ParseOperation(type="move", source=motion_target))
            else:
                operations.append(ParseOperation(type="move", target=motion_target, reference=entities[1].id if len(entities) > 1 else None))
        elif not operations and any(token in text or token in low for token in ("搜索", "寻找", "查找", "search")):
            operations.append(ParseOperation(type="search", target=entities[0].id))
        elif not operations and any(token in text or token in low for token in ("定位", "找到", "找个点", "找一个点", "找个位置", "找一个位置", "locate", "find a point", "find a position")):
            operations.append(ParseOperation(type="locate", target=entities[0].id))
        elif not operations and any(token in text or token in low for token in ("抓", "拿", "拾", "捡", "grasp", "pick")):
            operations.append(ParseOperation(type="grasp", target=entities[0].id))
        elif not operations and any(token in text or token in low for token in ("释放", "放开", "release")):
            operations.append(ParseOperation(type="release", target=entities[0].id))
        if not operations:
            return TaskParseLLMOutput(status="unsupported_task", raw_task=text)
        motion_direction = None
        if any(token in low for token in motion_words):
            direction_tokens = (("向左", "left"), ("向右", "right"), ("向前", "front"), ("向后", "back"), ("向上", "up"), ("向下", "down"), ("往左", "left"), ("往右", "right"), ("往前", "front"), ("往后", "back"), ("往上", "up"), ("往下", "down"), ("left", "left"), ("right", "right"), ("front", "front"), ("back", "back"), ("up", "up"), ("down", "down"))
            motion_direction = next((value for token, value in direction_tokens if token in low), None)
        return TaskParseLLMOutput(status="accepted", entities=entities, operations=operations, relations=relations, raw_direction=motion_direction)


class FakeVisionGroundingProvider:
    def __init__(self, detections=None): self.detections = detections

    def detect(self, request):
        if self.detections is not None:
            return VisionLLMOutput(detections=self.detections)
        seen = {}
        values = []
        for index, entity in enumerate(request.entities):
            seen[entity.id] = seen.get(entity.id, 0) + 1
            detection_id = entity.id if seen[entity.id] == 1 else f"{entity.id}-{seen[entity.id]}"
            values.append(VisionCandidate(detection_id=detection_id, entity_id=entity.id, bbox=[100 + index * 100, 100 + index * 80, 280 + index * 100, 300 + index * 80]))
        return VisionLLMOutput(detections=values)


class FakeSkillPlanningProvider:
    def plan(self, request):
        plans = []
        for operation in request.context.operations:
            proxy = Operation(
                operation_id=operation.id,
                task_type=TaskType(operation.type),
                target="target" if operation.target else None,
                source="source" if operation.source else None,
                destination="destination" if operation.destination else None,
                reference="reference" if operation.reference else None,
            )
            steps = [LLMPlanStep(skill=skill, target=target, reference=reference, region=region) for skill, target, reference, region in RECIPE_DEFINITIONS[operation.type].build(proxy)]
            plans.append(LLMOperationPlan(id=operation.id, steps=steps))
        return SkillPlanLLMOutput(operations=plans)
