# test_util.py

import unittest
from autopsy.util import calc_r

class TestCalcR(unittest.TestCase):
    def test_sort_atomic_indices(self):
        positions = [...]  # Define your input data
        cell = [...]  # Define your input data
        result = calc_r.sort_atomic_indices(positions, cell)
        self.assertEqual(result, expected_result)

