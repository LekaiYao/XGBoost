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
    resolve_weights,
    signed_weighted_auc,
    three_way_split_indices,
    validate_columns,
)


class ReweightingCoreTest(unittest.TestCase):
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
