import math
import unittest

import numpy as np

from vln_memory.geometry_memory import Intrinsics, OccupancyMemory


class DepthTest(unittest.TestCase):
    def setUp(self):
        self.m = OccupancyMemory((11, 11), (-5.5, -5.5), 1)
        self.eye = np.eye(4)
        self.eye[1, 3] = 1
        self.K = Intrinsics.from_hfov(1, 1, 90)

    def frame(self, d=3., t=1, **kwargs):
        self.m.update_depth(np.array([[d]]), self.K, self.eye, (0, 0, 0), t, **kwargs)

    def test_wall_and_unseen_behind(self):
        self.frame()
        self.assertEqual(self.m.grid()[:, 5].tolist(), [-1, -1, 100, 0, 0, 0, -1, -1, -1, -1, -1])

    def test_translated_camera_yaw_and_distinct_base(self):
        self.eye[:3, :3] = [[0, 0, 1], [0, 1, 0], [-1, 0, 0]]
        self.eye[:3, 3] = (1, 1, 2)
        self.frame()
        self.assertEqual(self.m.grid()[7, 3], 100)
        self.assertEqual(self.m.grid()[7, 6], 0)
        self.assertEqual(self.m.visit_count[5, 5], 1)
        self.assertEqual(self.m.visit_count[7, 6], 0)

    def test_ground_ceiling_filtered_and_depth_is_axial(self):
        K = Intrinsics(3, 3, 1, 1, 1, 1)
        self.m.update_depth(np.full((3, 3), 2.), K, self.eye, (0, 0, 0), 1, stride=1)
        self.assertEqual(set(zip(*np.where(self.m.grid() == 100))), {(3, 3), (3, 5), (3, 7)})
        self.assertAlmostEqual(self.m.log_odds[5, 5], -0.45)

    def test_slab_clips_free_evidence(self):
        self.eye[:3, :3] = [[1, 0, 0], [0, 0.8, 0.6], [0, -0.6, 0.8]]
        self.frame(4)
        self.assertFalse(np.any(self.m.grid() == 100))
        self.assertEqual(self.m.grid()[2, 5], -1)

    def test_invalid_and_clipped_returns(self):
        for t, d in enumerate((np.nan, np.inf, -1, 0, 8, 9)):
            self.frame(d, t)
        self.assertTrue(np.all(self.m.grid() == -1))
        self.frame(7, 8)
        self.assertEqual(self.m.grid()[0, 5], 0)
        self.assertFalse(np.any(self.m.grid() == 100))

    def test_rejected_frame_is_atomic(self):
        self.frame()
        odds = self.m.log_odds.copy()
        for t in (0, 1, math.nan):
            with self.assertRaises(ValueError):
                self.frame(t=t)
        self.eye[0, 0] = -1
        with self.assertRaises(ValueError):
            self.frame(t=2)
        np.testing.assert_array_equal(odds, self.m.log_odds)
        self.assertEqual((self.m.map_version, self.m.timestamp, self.m.visit_count.sum()), (1, 1, 1))

    def test_verified_polyline_and_pose_only_jump(self):
        self.frame(np.nan)
        path = [(0, 0), (2, 0), (2, 2), (2, 0)]
        self.m.update_depth(np.array([[np.nan]]), self.K, self.eye, (2, 0, 0), 2, base_path=path)
        np.testing.assert_array_equal(self.m.visit_count[5, 5:8], [1, 1, 2])
        self.assertEqual((self.m.visit_count[6, 7], self.m.visit_count[7, 7]), (2, 1))
        self.assertEqual(self.m.visit_count.sum(), 7)
        with self.assertRaises(ValueError):
            self.m.update_depth(np.array([[3.]]), self.K, self.eye, (3, 0, 0), 3, base_path=[(0, 0), (3, 0)])
        self.assertEqual(self.m.map_version, 2)
        self.assertTrue(np.all(self.m.grid() == -1))
        self.m.update_depth(np.array([[np.nan]]), self.K, self.eye, (-3, 0, 0), 3)
        self.assertEqual(self.m.visit_count.sum(), 8)
        self.assertEqual(self.m.visit_count[5, 3], 0)

    def test_saturation_and_reversal(self):
        for t in range(12):
            self.frame(t=t)
        self.assertEqual(self.m.log_odds[2, 5], 4)
        for t in range(12, 24):
            self.frame(4, t)
        self.assertEqual(self.m.grid()[2, 5], 0)
        self.assertEqual(self.m.visit_count.sum(), 1)
