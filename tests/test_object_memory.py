import unittest
import numpy as np
from vln_memory.geometry_memory import Intrinsics
from vln_memory.object_models import normalize
from vln_memory.object_geometry import localize


def entity(lid="A", cat="chair", box=(350, 350, 650, 650), color="red"):
    return dict(
        local_id=lid,
        category=cat,
        bbox=list(box),
        attributes={"color": color},
        confidence=0.9,
        evidence="visible",
    )


def scene(frame, *entities):
    return dict(
        task_id="task",
        observation_id=str(frame),
        view_id="front",
        target_status="NOT_VISIBLE",
        target_entities=[],
        context_entities=list(entities),
        visual_cues=[],
    )


camera = Intrinsics.from_hfov(100, 80, 90)
depth = np.full((80, 100), 2.0)


class GeometryTests(unittest.TestCase):
    def test_rotated_pose_outlier_and_missing_depth(self):
        pose = np.array([[0, 0, 1, 4], [0, 1, 0, 1], [-1, 0, 0, 5], [0, 0, 0, 1.0]])
        obs = normalize(scene(1, entity()), 1)
        image = depth.copy()
        image[38:42, 49:51] = 7.0
        localize(obs, image, camera, pose)
        np.testing.assert_allclose(obs[0].map_position_xz, [2, 5], atol=0.04)
        obs = normalize(scene(2, entity()), 2)
        localize(obs, image * np.nan, camera, pose)
        self.assertIsNone(obs[0].map_position_xz)
        np.testing.assert_allclose(obs[0].bearing_world, [-1, 0, 0])


from vln_memory.object_memory import ObjectMemory
from vln_memory.object_association import associate
from vln_memory.object_models import ObjectRecord


class MemoryTests(unittest.TestCase):
    def test_single_view_retries_conflict_and_goal_scope(self):
        memory = ObjectMemory()
        pose = np.eye(4)

        def feed(n, cat, role, color="red"):
            result = scene(n, entity(cat=cat, color=color))
            if role == "target":
                result["target_entities"] = result.pop("context_entities")
                result["context_entities"] = []
                result["target_status"] = "VISIBLE"
            return memory.consume(result, depth, camera, pose, n)[0]

        first = feed(0, "chair", "target")
        key = first.object_id
        obj = memory.get_object(key)
        score = dict(obj.category_scores)
        self.assertEqual(feed(1, "armchair", "context").object_id, key)
        self.assertEqual(obj.category_scores, score)
        self.assertEqual(obj.track_status, "PROVISIONAL")
        pose[1, 3] += 0.3
        self.assertEqual(feed(2, "sofa", "context", "unknown").object_id, key)
        self.assertEqual(obj.semantic_status, "CONFLICTED")
        self.assertEqual(len(obj.independent_ids), 2)
        self.assertEqual(set(obj.attribute_scores["color"]), {"red"})
        goal = dict(target_category="chair", target_attributes={}, spatial_relations=[])
        memory.apply_verification(
            dict(
                task_id="task",
                observation_id="0",
                view_id="front",
                candidate_id="A",
                status="REJECTED",
                opposing_evidence=["wrong shape"],
                supporting_evidence=[],
                uncertainty_reason="",
            ),
            3,
        )
        self.assertEqual(memory.query_goal_objects(goal, "task")[key], "NON_MATCH")
        self.assertEqual(
            memory.query_goal_objects(goal, "other")[key], "POSSIBLE_MATCH"
        )
        memory.consume(scene(3), depth, camera, pose, 3)
        self.assertEqual(obj.last_seen, 2)
        self.assertEqual(len(obj.observation_ids), 3)
        with self.assertRaises(ValueError):
            feed(2, "sofa", "context")
        self.assertEqual(len(memory.observations), 3)

    def test_ambiguous_competing_and_unmapped(self):
        records = {}
        for key, x in [("a", 0), ("b", 0.4)]:
            records[key] = ObjectRecord(
                key, 0, 0, (x, 0), 0.1, category_scores={"chair": 1}
            )
        obs = normalize(scene(0, entity()), 0)[0]
        from copy import deepcopy

        items = [deepcopy(obs) for _ in range(5)]
        for o, x in zip(items, [0, 0.01, 0.2, 4]):
            o.map_position_xz = (x, 0)
            o.position_sigma_m = 0.1
        associate(items, records)
        self.assertEqual(
            [o.association_status for o in items],
            ["MATCHED", "AMBIGUOUS", "AMBIGUOUS", "NEW", "UNLOCALIZED"],
        )
