import unittest

import numpy as np

from workflows.compare_score_efficiency import weighted_efficiency


class ScoreEfficiencyComparisonTest(unittest.TestCase):
    def test_weighted_efficiency_and_analytic_error(self):
        scores = np.array([0.1, 0.4, 0.7, 0.9])
        weights = np.array([1.0, 2.0, 3.0, 4.0])
        efficiency, error = weighted_efficiency(scores, weights, 0.5)
        self.assertAlmostEqual(efficiency, 0.7)
        expected_variance = np.sum(weights ** 2 * (np.array([0, 0, 1, 1]) - 0.7) ** 2) / weights.sum() ** 2
        self.assertAlmostEqual(error, expected_variance ** 0.5)

    def test_invalid_weights_are_excluded(self):
        efficiency, _ = weighted_efficiency([0.2, 0.8, 0.9], [1.0, -2.0, np.nan], 0.5)
        self.assertEqual(efficiency, 0.0)


if __name__ == "__main__":
    unittest.main()
