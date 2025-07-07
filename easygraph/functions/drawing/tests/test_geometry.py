import math
import unittest

import numpy as np

from easygraph.functions.drawing.geometry import common_tangent_radian
from easygraph.functions.drawing.geometry import polar_position
from easygraph.functions.drawing.geometry import rad_2_deg
from easygraph.functions.drawing.geometry import radian_from_atan
from easygraph.functions.drawing.geometry import vlen


class TestGeometryUtils(unittest.TestCase):
    def test_radian_from_atan_axes(self):
        self.assertAlmostEqual(radian_from_atan(0, 1), math.pi / 2)
        self.assertAlmostEqual(radian_from_atan(0, -1), 3 * math.pi / 2)
        self.assertAlmostEqual(radian_from_atan(1, 0), 0)
        self.assertAlmostEqual(radian_from_atan(-1, 0), math.pi)

    def test_radian_from_atan_quadrants(self):
        # Q1
        self.assertAlmostEqual(radian_from_atan(1, 1), math.atan(1))
        # Q4
        self.assertAlmostEqual(radian_from_atan(1, -1), math.atan(-1) + 2 * math.pi)
        # Q2
        self.assertAlmostEqual(radian_from_atan(-1, 1), math.atan(-1) + math.pi)
        # Q3
        self.assertAlmostEqual(radian_from_atan(-1, -1), math.atan(1) + math.pi)

    def test_radian_from_atan_zero_vector(self):
        result = radian_from_atan(0, 0)
        self.assertAlmostEqual(result, 3 * math.pi / 2)

    def test_vlen(self):
        self.assertEqual(vlen((3, 4)), 5.0)
        self.assertEqual(vlen((0, 0)), 0.0)
        self.assertAlmostEqual(vlen((-3, -4)), 5.0)

    def test_common_tangent_radian_basic(self):
        r1, r2, d = 3, 2, 5
        angle = common_tangent_radian(r1, r2, d)
        expected = math.acos(abs(r2 - r1) / d)
        self.assertAlmostEqual(angle, expected)

    def test_common_tangent_radian_reversed(self):
        r1, r2, d = 2, 3, 5
        angle = common_tangent_radian(r1, r2, d)
        expected = math.pi - math.acos(abs(r2 - r1) / d)
        self.assertAlmostEqual(angle, expected)

    def test_common_tangent_radian_touching(self):
        self.assertAlmostEqual(common_tangent_radian(3, 3, 5), math.pi / 2)

    def test_common_tangent_radian_invalid(self):
        with self.assertRaises(ValueError):
            common_tangent_radian(5, 1, 2)

    def test_polar_position_origin(self):
        pos = polar_position(0, 0, np.array([5, 5]))
        np.testing.assert_array_almost_equal(pos, np.array([5, 5]))

    def test_polar_position_90deg(self):
        pos = polar_position(1, math.pi / 2, np.array([0, 0]))
        np.testing.assert_array_almost_equal(pos, np.array([0, 1]))

    def test_polar_position_negative_angle(self):
        pos = polar_position(1, -math.pi / 2, np.array([1, 1]))
        np.testing.assert_array_almost_equal(pos, np.array([1, 0]))

    def test_rad_2_deg(self):
        self.assertEqual(rad_2_deg(0), 0)
        self.assertEqual(rad_2_deg(math.pi), 180)
        self.assertEqual(rad_2_deg(2 * math.pi), 360)
        self.assertEqual(rad_2_deg(-math.pi / 2), -90)


if __name__ == "__main__":
    unittest.main()
