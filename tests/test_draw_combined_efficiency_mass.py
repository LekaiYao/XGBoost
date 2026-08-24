import unittest

from workflows.draw_combined_efficiency_mass import threshold_map
from configs.year_pairings import resolve_year_pairing


class CombinedEfficiencyMassDrawTest(unittest.TestCase):
    def test_pb23_anchor_resolves_frozen_pb24_tag(self):
        anchor = "X_pb23_v3_fid3_6v5_rwr6range5v1_xgb_v1"
        pairing = resolve_year_pairing(anchor)
        self.assertEqual(pairing["tags"]["pb23"], anchor)
        self.assertEqual(
            pairing["tags"]["pb24"],
            "X_pb24_v19_fid19_6v5_rwr6range5v1_xgb_v1",
        )
        self.assertEqual(pairing["fit_scan_efficiencies"], [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40])

    def test_threshold_map(self):
        payload = {"thresholds": [
            {"target_efficiency": 0.5, "score_threshold": 0.8},
            {"target_efficiency": 0.2, "score_threshold": 0.96},
        ]}
        result = threshold_map(payload)
        self.assertEqual(result[0.5]["score_threshold"], 0.8)
        self.assertEqual(result[0.2]["score_threshold"], 0.96)

    def test_duplicate_targets_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            threshold_map({"thresholds": [
                {"target_efficiency": 0.5},
                {"target_efficiency": 0.5},
            ]})


if __name__ == "__main__":
    unittest.main()
