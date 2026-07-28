import unittest
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from utils.training_weights import (
    balanced_scale_pos_weight,
    resolve_training_weights,
    weighted_ks_curve,
)
from dag.make_single_workflow import make_dag
from configs.samples import infer_reweight_profile, resolve_training_reweight_config


class TrainingWeightsTest(unittest.TestCase):
    def test_resolve_training_weights(self):
        frame = pd.DataFrame({"Reweight": [0.5, 1.5, 2.0]})
        np.testing.assert_allclose(
            resolve_training_weights(frame, "Reweight", "signal"),
            [0.5, 1.5, 2.0],
        )
        np.testing.assert_allclose(
            resolve_training_weights(frame, None, "signal"),
            np.ones(3),
        )

    def test_reject_negative_training_weight(self):
        frame = pd.DataFrame({"Reweight": [1.0, -0.1]})
        with self.assertRaisesRegex(ValueError, "non-negative"):
            resolve_training_weights(frame, "Reweight", "signal")

    def test_balanced_scale_pos_weight_uses_weight_sums(self):
        labels = np.array([1, 1, 0, 0, 0])
        weights = np.array([0.5, 1.5, 1.0, 1.0, 2.0])
        self.assertAlmostEqual(balanced_scale_pos_weight(labels, weights), 2.0)

    def test_weighted_ks_curve(self):
        result = weighted_ks_curve(
            np.array([1, 1, 0, 0]),
            np.array([0.8, 0.9, 0.1, 0.2]),
            np.array([1.0, 2.0, 1.0, 1.0]),
            thresholds=np.array([0.0, 0.5, 1.0]),
        )
        self.assertAlmostEqual(result["ks_stat"], 1.0)
        self.assertEqual(result["sig_cdf"], [0.0, 0.0, 1.0])
        self.assertEqual(result["bkg_cdf"], [0.0, 1.0, 1.0])

    def test_single_dag_accepts_reweight_profile_tag(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            dag = make_dag(
                Path(temporary_directory),
                "X_pp24_v4_fid3_8v2_rwpsi2sr5v1_xgb_v1",
                False,
                False,
            )
            self.assertIn(
                'train_tag="X_pp24_v4_fid3_8v2_rwpsi2sr5v1_xgb_v1"',
                dag.read_text(),
            )

    def test_old_and_explicit_unweighted_tags_resolve_to_rw0(self):
        self.assertEqual(infer_reweight_profile("X_pp24_v3_fid2_4v1_xgb_v1"), "rw0")
        self.assertEqual(
            infer_reweight_profile("X_pp24_v4_fid3_8v2_rw0_xgb_v1"), "rw0"
        )

    def test_weighted_profile_resolves_config_and_guards_profiles(self):
        profile = resolve_training_reweight_config(
            "pp", "X", "2024", "rwpsi2sr5v1", "pp24_v4", "pp24_fid3"
        )
        self.assertEqual(profile["weight_branch"], "Reweight")
        self.assertEqual(profile["signal"]["tree"], "ntmix_X3872")
        with self.assertRaisesRegex(ValueError, "requires"):
            resolve_training_reweight_config(
                "pp", "X", "2024", "rwpsi2sr5v1", "pp24_v3", "pp24_fid2"
            )


if __name__ == "__main__":
    unittest.main()
