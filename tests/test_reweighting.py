import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import uproot

from utils.varsets import get_reweight_varset_columns
from workflows.reweighting.core import (
    build_diagnostics,
    effective_sample_size,
    load_tree_frame,
    positive_weight_tail_summary,
    resolve_weights,
    signed_weighted_auc,
    three_way_split_indices,
    validate_columns,
)
from workflows.reweighting.validate_x_splot_transfer_closure import (
    STATUS,
    normalized_histogram,
    quantile_bins,
    refresh_metadata,
    ratio_to_signed_target,
)


class ReweightingCoreTest(unittest.TestCase):
    def test_transfer_metadata_refresh_adds_status_without_changing_values(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            csv_path = output / "transfer_closure_summary.csv"
            csv_path.write_text("variable,cdf_distance_before\nBtrk1dR,0.1\n")
            binning_path = output / "binning_and_histograms.json"
            binning_path.write_text(json.dumps({"Btrk1dR": {"edges": [0, 1]}}))
            refresh_metadata(output)
            with csv_path.open(newline="") as stream:
                row = next(csv.DictReader(stream))
            self.assertEqual(row["status"], STATUS)
            self.assertEqual(float(row["cdf_distance_before"]), 0.1)
            payload = json.loads(binning_path.read_text())
            self.assertEqual(payload["status"], STATUS)
            self.assertEqual(payload["variables"]["Btrk1dR"]["edges"], [0, 1])

    def test_transfer_quantile_bins_cover_full_range(self):
        values = np.arange(101, dtype=float)
        bins = quantile_bins(values, count=10)
        self.assertEqual(bins[0], 0.0)
        self.assertEqual(bins[-1], 100.0)
        self.assertTrue(np.all(np.diff(bins) > 0.0))

    def test_signed_histogram_and_ratio_undefined_bin(self):
        values = np.array([0.2, 0.3, 1.2, 1.3])
        weights = np.array([1.0, -1.0, 2.0, 1.0])
        hist, error = normalized_histogram(values, weights, np.array([0.0, 1.0, 2.0]))
        np.testing.assert_allclose(hist, [0.0, 1.0])
        self.assertTrue(np.all(error > 0.0))
        ratio, valid, _ = ratio_to_signed_target(np.array([0.5, 1.0]), hist)
        self.assertFalse(valid[0])
        self.assertTrue(np.isnan(ratio[0]))
        self.assertEqual(ratio[1], 1.0)

    def test_named_reweight_varsets(self):
        self.assertEqual(
            get_reweight_varset_columns("pp", "R3", "X"),
            ["Bcos_dtheta", "Btktkpt", "Bchi2Prob"],
        )
        self.assertEqual(
            get_reweight_varset_columns("pp", "R5", "X"),
            ["Bcos_dtheta", "Btktkpt", "Bchi2Prob", "Btrk2Pt", "Btrk1Pt"],
        )
        self.assertEqual(
            get_reweight_varset_columns("pp", "R4_noCos", "X"),
            ["Btktkpt", "Bchi2Prob", "Btrk2Pt", "Btrk1Pt"],
        )
        self.assertEqual(len(get_reweight_varset_columns("pp", "R8", "X")), 8)

    def test_effective_sample_size_supports_signed_weights(self):
        weights = np.array([1.0, 1.0, -0.25, 0.5])
        expected = weights.sum() ** 2 / np.square(weights).sum()
        self.assertAlmostEqual(effective_sample_size(weights), expected)

    def test_positive_weight_tail_summary_uses_complete_range(self):
        weights = np.array([1.0, 2.0, 3.0, 100.0])
        summary = positive_weight_tail_summary(weights)
        self.assertEqual(summary["maximum"], 100.0)
        self.assertEqual(summary["top_1pct_events"]["event_count"], 1)
        self.assertAlmostEqual(
            summary["top_1pct_events"]["weight_fraction"], 100.0 / 106.0
        )

    def test_root_input_validation_and_signed_diagnostics(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "target.root"
            with uproot.recreate(path) as root_file:
                root_file.mktree("events", {
                    "x": np.array([0.0, 1.0, 2.0, 3.0]),
                    "sWeight": np.array([1.0, -0.1, 0.8, 0.3]),
                })
            frame = load_tree_frame(path, "events")
            validate_columns(frame, ["x"], "target")
            weights = resolve_weights(frame, "sWeight", "target")
            diagnostics = build_diagnostics(
                pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0]}),
                frame,
                ["x"],
                np.ones(4),
                np.array([0.5, 1.0, 1.5, 1.0]),
                weights,
            )
            self.assertEqual(diagnostics["weights"]["target"]["negative_entries"], 1)
            self.assertIn("cdf_distance_after", diagnostics["variables"]["x"])

    def test_rejects_missing_columns(self):
        with self.assertRaisesRegex(ValueError, "missing columns"):
            validate_columns(pd.DataFrame({"x": [1.0]}), ["y"], "sample")

    def test_three_way_split_is_disjoint_and_complete(self):
        parts = three_way_split_indices(101, random_state=7, reweighter_fraction=0.5)
        combined = np.concatenate(list(parts.values()))
        self.assertEqual(len(combined), 101)
        self.assertEqual(len(np.unique(combined)), 101)
        self.assertEqual(set(combined), set(range(101)))

    def test_signed_weighted_auc(self):
        mc_score = np.array([0.1, 0.2, 0.3])
        target_score = np.array([0.7, 0.8, 0.9])
        self.assertAlmostEqual(
            signed_weighted_auc(
                mc_score,
                target_score,
                np.ones(3),
                np.array([1.0, -0.2, 0.7]),
            ),
            1.0,
        )
        identical = np.array([0.1, 0.2, 0.3])
        self.assertAlmostEqual(
            signed_weighted_auc(identical, identical, np.ones(3), np.ones(3)),
            0.5,
        )


if __name__ == "__main__":
    unittest.main()
