import unittest

import numpy as np

from vln_memory.geometry_memory import OccupancyMemory


class GridTest(unittest.TestCase):
    def test_axes_negative_origin_and_open_boundary(self):
        m = OccupancyMemory((4, 6), (-2, -1), 0.5)
        self.assertEqual(m.cell((-0.75, 0.6)), (3, 2))
        self.assertFalse(m.contains(m.cell((-2.001, 0))))
        self.assertFalse(m.contains(m.cell((1, 0))))
        self.assertTrue(m.contains(m.cell((-2, -1))))

    def test_unknown_and_visits_are_independent(self):
        m = OccupancyMemory((3, 3), (0, 0))
        for cell in [(1, 1), (1, 1), (1, 2), (1, 1)]:
            m._visit(cell)
        self.assertTrue(np.all(m.grid() == -1))
        self.assertEqual(m.visit_count[1].tolist(), [0, 2, 1])

    def test_invalid_configuration(self):
        for shape, origin, resolution in [
            ((0, 3), (0, 0), 1), ((3, 3), (0, np.nan), 1), ((3, 3), (0, 0), 0)
        ]:
            with self.assertRaises(ValueError):
                OccupancyMemory(shape, origin, resolution)
