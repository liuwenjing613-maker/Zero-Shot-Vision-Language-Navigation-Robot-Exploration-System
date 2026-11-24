"""Negative control: a 90-degree mounting error must fail the full-scan evaluator."""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evals/m1_lab"))
from scan_lab import ANGLES, N, ScanExperiment, rectangle, segment_ranges
from vln_memory.geometry_memory import OccupancyMemory


class ScanReferenceTest(unittest.TestCase):
    def test_wrong_bearing_is_rejected(self):
        xy = np.array([-2.,0.])
        ref = np.concatenate([rectangle(-4,-3,4,3),rectangle(-0.65,-0.8,0.65,0.8)])
        ranges = segment_ranges(xy,ANGLES,ref)
        with tempfile.TemporaryDirectory() as folder:
            grid = OccupancyMemory((81,101),(-5.05,-4.05),0.1)
            exp = ScanExperiment(folder,"room360",grid)
            with self.assertRaisesRegex(AssertionError, "P95"):
                exp.sweep(xy,0.,np.roll(ranges,N//4),ref,"wrong-bearing","negative control")
