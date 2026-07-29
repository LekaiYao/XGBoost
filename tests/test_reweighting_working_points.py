import unittest

import numpy as np

from workflows.reweighting.evaluate_working_points import (
    efficiencies_at_cut,
    score_cut_for_background_efficiency,
)


class ReweightingWorkingPointsTest(unittest.TestCase):
    def test_score_cut_targets_background_efficiency(self):
        scores = np.arange(1000, dtype=float)
        cut, achieved = score_cut_for_background_efficiency(scores, 0.10)
        self.assertEqual(cut, 900.0)
        self.assertAlmostEqual(achieved, 0.10)

    def test_reports_unweighted_and_reweighted_efficiency(self):
        scores = np.array([0.1, 0.4, 0.7, 0.9])
        weights = np.array([1.0, 1.0, 2.0, 6.0])
        result = efficiencies_at_cut(scores, 0.5, weights)
        self.assertAlmostEqual(result["unweighted"], 0.5)
        self.assertAlmostEqual(result["reweighted"], 0.8)
        self.assertEqual(result["passed_entries"], 2)

    def test_rejects_invalid_efficiency_weights(self):
        with self.assertRaisesRegex(ValueError, "non-negative"):
            efficiencies_at_cut([0.1, 0.9], 0.5, [1.0, -0.1])


if __name__ == "__main__":
    unittest.main()
