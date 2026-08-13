import unittest

import numpy as np

from utils.score_thresholds import weighted_efficiency_thresholds


class ScoreThresholdsTest(unittest.TestCase):
    def test_reports_strict_cut_efficiency_with_repeated_scores(self):
        rows = weighted_efficiency_thresholds(
            [0.1, 0.2, 0.2, 0.9], [1.0, 1.0, 2.0, 2.0], [0.5]
        )
        self.assertEqual(rows[0]["score_threshold"], 0.2)
        self.assertAlmostEqual(rows[0]["achieved_efficiency"], 2.0 / 6.0)

    def test_rejects_empty_or_invalid_reference_weights(self):
        with self.assertRaisesRegex(ValueError, "empty"):
            weighted_efficiency_thresholds([], [], [0.5])
        with self.assertRaisesRegex(ValueError, "strictly positive"):
            weighted_efficiency_thresholds([0.1, 0.2], [1.0, 0.0], [0.5])
        with self.assertRaisesRegex(ValueError, "between zero and one"):
            weighted_efficiency_thresholds([0.1], [1.0], [1.0])


if __name__ == "__main__":
    unittest.main()
