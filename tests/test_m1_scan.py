import math
import unittest

import numpy as np
from vln_memory.geometry_memory import OccupancyMemory


class ScanTest(unittest.TestCase):
    def setUp(self):
        self.memory = OccupancyMemory((21, 21), (-2.05, -2.05), 0.2)

    def scan(self, ranges, tick=1, heading=0, eye=(0, 0), inc=math.pi / 2):
        self.memory.update_scan(
            ranges, 0, inc, eye, heading, (0, 0), tick, range_min=0.08, range_max=1.8
        )

    def value(self, xy):
        return self.memory.grid()[self.memory.cell(xy)]

    def test_hit_free_occluded_and_full_turn(self):
        self.scan([1.0, 1.0, 1.0, 1.0])
        for x, y in ((1, 0), (0, 1), (-1, 0), (0, -1)):
            self.assertEqual(self.value((x, y)), 100)
            self.assertEqual(self.value((0.6 * x, 0.6 * y)), 0)
            self.assertEqual(self.value((1.4 * x, 1.4 * y)), -1)
        self.assertEqual(self.memory.map_version, 1)

    def test_clear_beam_and_invalid_beams(self):
        self.scan([np.inf, np.nan, 0.01, -np.inf])
        self.assertEqual(self.value((1.6, 0)), 0)
        self.assertEqual(self.value((2.0, 0)), -1)
        self.assertFalse((self.memory.grid() == 100).any())
        self.assertTrue(all(self.value(p) == -1 for p in [(0, 1), (-1, 0), (0, -1)]))

    def test_sensor_extrinsic_negative_increment_and_idempotent_rejection(self):
        self.scan([0.8, 1.0], heading=math.pi / 2, eye=(0.4, 0), inc=-math.pi / 2)
        self.assertEqual(self.value((0.4, 0.8)), 100)
        self.assertEqual(self.value((1.4, 0)), 100)
        before = self.memory.log_odds.copy()
        with self.assertRaises(ValueError):
            self.scan([1.0])
        np.testing.assert_array_equal(before, self.memory.log_odds)
        self.assertEqual(self.memory.visit_count.sum(), 1)

    def test_one_vote_per_sweep_and_hit_priority(self):
        self.scan([0.8, 0.8, 1.6], inc=0.0001)
        self.assertAlmostEqual(self.memory.log_odds[self.memory.cell((0.8, 0))], 0.9)
        self.assertAlmostEqual(self.memory.log_odds[self.memory.cell((0.4, 0))], -0.45)
