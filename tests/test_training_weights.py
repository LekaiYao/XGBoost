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
from configs.samples import (
    infer_reweight_profile,
    resolve_apply_config,
    resolve_draw_config,
    resolve_extra_mc_apply_config,
    resolve_fiducial_config,
    resolve_training_config,
    resolve_training_reweight_config,
)
from configs.direct_xgb_settings import DIRECT_XGB_PARAMS
from utils.apply_inputs import resolve_apply_mc_input
from utils.selection import apply_selection
from utils.varsets import get_varset_columns


class TrainingWeightsTest(unittest.TestCase):
    def test_psi2s_top_level_channel_configuration(self):
        tag = "Psi2S_pb24_v1_fid1_6v1_rwr6range4v1_xgb_v1"
        self.assertEqual(infer_reweight_profile(tag), "rwr6range4v1")

        training = resolve_training_config("pbpb", "Psi2S", "2024", "pb24_v1")
        fiducial = resolve_fiducial_config("pbpb", "Psi2S", "pb24_fid1")
        self.assertEqual(training["signal"]["tree"], "ntmix_PSI2S")
        self.assertEqual(training["background"]["tree"], "ntmix")
        self.assertIn("Bmass > 3.60 and Bmass < 3.65", training["background_selection"])
        self.assertIn("Bmass > 3.75 and Bmass < 3.80", training["background_selection"])
        self.assertNotIn("Bmass > 3.95", training["background_selection"])
        for expression in (
            training["signal_selection"], training["background_selection"],
            fiducial["expression"],
        ):
            self.assertIn("Btrk1Pt > 0.9", expression)
            self.assertIn("Btrk2Pt > 0.9", expression)
            self.assertIn("Btrk2dR <= 0.25", expression)

        profile = resolve_training_reweight_config(
            "pbpb", "Psi2S", "2024", "rwr6range4v1", "pb24_v1", "pb24_fid1"
        )
        self.assertEqual(profile["signal"]["tree"], "ntmix_PSI2S")
        self.assertEqual(profile["weight_branch"], "Reweight")
        self.assertIn("Psi2S_pp24_R6range4_rw_v1", profile["signal"]["path"])
        with self.assertRaisesRegex(ValueError, "requires"):
            resolve_training_reweight_config(
                "pbpb", "Psi2S", "2024", "rwr6range4v1", "pb24_v2", "pb24_fid1"
            )

        apply_cfg = resolve_apply_config("pbpb", "Psi2S", "2024")
        extra_cfg = resolve_extra_mc_apply_config("pbpb", "Psi2S", "2024", "x3872")
        draw_cfg = resolve_draw_config("pbpb", "Psi2S", "2024")
        self.assertEqual(apply_cfg["mc"][0]["tree"], "ntmix_PSI2S")
        self.assertEqual(extra_cfg["samples"]["x3872"]["tree"], "ntmix_X3872")
        self.assertEqual(draw_cfg["plot"]["reference_masses"], [3.686])
        self.assertEqual(draw_cfg["plot"]["mass_range"], [3.60, 3.80])
        self.assertEqual(DIRECT_XGB_PARAMS["pb24"]["Psi2S"], DIRECT_XGB_PARAMS["pb24"]["X"])

        x_training = resolve_training_config("pbpb", "X", "2024", "pb24_v17")
        self.assertEqual(training["background"]["path"], x_training["background"]["path"])
        self.assertNotEqual(training["signal"]["tree"], x_training["signal"]["tree"])

    def test_ppref_pbpb24_common_fiducial_profiles_match(self):
        pp = resolve_fiducial_config("pp", "X", "pp24_fid4")
        pbpb = resolve_fiducial_config("pbpb", "X", "pb24_fid7")
        expected = "BQvalue < 0.15 and abs(By) < 1.6 and Bpt > 10.0 and Bpt < 50.0"
        self.assertEqual(pp["expression"], expected)
        self.assertEqual(pbpb["expression"], expected)

    def test_pbpb24_v6_efficiency_reference_is_configured(self):
        training = resolve_training_config("pbpb", "X", "2024", "pb24_v6")
        reference = training["efficiency_reference_signal"]
        self.assertEqual(reference["tree"], "ntmix_X3872")
        self.assertIn("X_pp24_xsplot_R6range2_rw_v1", reference["path"])
        self.assertEqual(training["efficiency_reference_weight_branch"], "Reweight")

    def test_ppref_pbpb24_r5range3_profiles_match(self):
        pp = resolve_fiducial_config("pp", "X", "pp24_fid5")
        pbpb = resolve_fiducial_config("pbpb", "X", "pb24_fid8")
        self.assertEqual(pp["expression"], pbpb["expression"])
        self.assertIn("Btrk1Pt >= 1.0", pp["expression"])
        self.assertIn("Btrk2Pt >= 1.0", pp["expression"])

    def test_pbpb24_psi2s_r5range3_training_profile(self):
        tag = "X_pb24_v7_fid8_8v2_rwpsi2sr5range3v1_xgb_v1"
        self.assertEqual(infer_reweight_profile(tag), "rwpsi2sr5range3v1")
        training = resolve_training_config("pbpb", "X", "2024", "pb24_v7")
        fiducial = resolve_fiducial_config("pbpb", "X", "pb24_fid8")
        rng = np.random.RandomState(7)
        probe = pd.DataFrame({
            "Bpt": rng.uniform(5, 55, 2000),
            "By": rng.uniform(-2.5, 2.5, 2000),
            "BQvalue": rng.uniform(0, 0.25, 2000),
            "Bcos_dtheta": rng.uniform(-1.2, 1.2, 2000),
            "Btktkpt": rng.uniform(1, 11, 2000),
            "Bchi2Prob": rng.uniform(-0.1, 1.1, 2000),
            "Btrk1Pt": rng.uniform(0.5, 5, 2000),
            "Btrk2Pt": rng.uniform(0.5, 5, 2000),
            "Btrk1dR": rng.uniform(-0.1, 0.6, 2000),
        })
        expected = apply_selection(probe, fiducial["expression"], "fiducial")
        actual = apply_selection(probe, training["signal_selection"], "signal")
        self.assertEqual(actual.index.tolist(), expected.index.tolist())
        self.assertIn("Bmass > 3.75 and Bmass < 3.80", training["background_selection"])
        profile = resolve_training_reweight_config(
            "pbpb", "X", "2024", "rwpsi2sr5range3v1", "pb24_v7", "pb24_fid8"
        )
        self.assertEqual(profile["signal"]["tree"], "ntmix_PSI2S")
        self.assertEqual(profile["weight_branch"], "Reweight")
        self.assertTrue(profile["signal"]["path"].endswith("flat_ntmix_PbPb24_MC_PSI2S_with_reweight.root"))

    def test_pbpb24_raw_psi2s_training_profile(self):
        tag = "X_pb24_v7_fid8_8v2_rwpsi2srawv1_xgb_v1"
        self.assertEqual(infer_reweight_profile(tag), "rwpsi2srawv1")
        profile = resolve_training_reweight_config(
            "pbpb", "X", "2024", "rwpsi2srawv1", "pb24_v7", "pb24_fid8"
        )
        self.assertEqual(profile["signal"]["tree"], "ntmix_PSI2S")
        self.assertIsNone(profile["weight_branch"])
        self.assertTrue(profile["signal"]["path"].endswith("flat_ntmix_PbPb24_MC_PSI2S.root"))

    def test_pbpb24_psi2s_btrk2dr045_training_profile(self):
        tag = "X_pb24_v8_fid9_8v2_rwpsi2sr5range3dr045v1_xgb_v1"
        self.assertEqual(infer_reweight_profile(tag), "rwpsi2sr5range3dr045v1")
        training = resolve_training_config("pbpb", "X", "2024", "pb24_v8")
        fiducial = resolve_fiducial_config("pbpb", "X", "pb24_fid9")
        self.assertIn("Btrk2dR < 0.45", training["signal_selection"])
        self.assertIn("Btrk2dR < 0.45", training["background_selection"])
        self.assertIn("Btrk2dR < 0.45", fiducial["expression"])
        rng = np.random.RandomState(9)
        probe = pd.DataFrame({
            "Bpt": rng.uniform(5, 55, 2000),
            "By": rng.uniform(-2.5, 2.5, 2000),
            "BQvalue": rng.uniform(0, 0.25, 2000),
            "Bcos_dtheta": rng.uniform(-1.2, 1.2, 2000),
            "Btktkpt": rng.uniform(1, 11, 2000),
            "Bchi2Prob": rng.uniform(-0.1, 1.1, 2000),
            "Btrk1Pt": rng.uniform(0.5, 5, 2000),
            "Btrk2Pt": rng.uniform(0.5, 5, 2000),
            "Btrk1dR": rng.uniform(-0.1, 0.6, 2000),
            "Btrk2dR": rng.uniform(0, 0.6, 2000),
        })
        expected = apply_selection(probe, fiducial["expression"], "fiducial")
        actual = apply_selection(probe, training["signal_selection"], "signal")
        self.assertEqual(actual.index.tolist(), expected.index.tolist())
        profile = resolve_training_reweight_config(
            "pbpb", "X", "2024", "rwpsi2sr5range3dr045v1", "pb24_v8", "pb24_fid9"
        )
        self.assertEqual(profile["weight_branch"], "Reweight")
        self.assertEqual(profile["signal"]["tree"], "ntmix_PSI2S")
        self.assertTrue(profile["signal"]["path"].endswith("flat_ntmix_PbPb24_MC_PSI2S_with_reweight.root"))
        with self.assertRaisesRegex(ValueError, "requires"):
            resolve_training_reweight_config(
                "pbpb", "X", "2024", "rwpsi2sr5range3dr045v1", "pb24_v7", "pb24_fid8"
            )

    def test_pbpb_x_reduced_psi2s_varsets(self):
        self.assertEqual(
            get_varset_columns("pbpb", "6v5", channel="X"),
            ["Btrk1dR", "Btktkpt", "Btrk1Pt", "Bchi2Prob", "Bcos_dtheta", "Btrk2Pt"],
        )
        self.assertEqual(
            get_varset_columns("pbpb", "7v2", channel="X"),
            ["Btrk1dR", "Btrk2dR", "Btktkpt", "Btrk1Pt", "Bchi2Prob", "Bcos_dtheta", "BtrkPtimb"],
        )

    def test_pbpb24_x_common09_weighted_profiles(self):
        cases = [
            (
                "X_pb24_v9_fid10_8v2_rwr6range3v1_xgb_v1",
                "pb24_v9", "pb24_fid10", "rwr6range3v1",
                "X_pp24_xsplot_R6range3_rw_v1",
            ),
            (
                "X_pb24_v10_fid11_9v2_rwr6v2range3v1_xgb_v1",
                "pb24_v10", "pb24_fid11", "rwr6v2range3v1",
                "X_pp24_xsplot_R6v2range3_rw_v1",
            ),
        ]
        for tag, selection_profile, fid_profile, reweight_profile, source_tag in cases:
            with self.subTest(tag=tag):
                self.assertEqual(infer_reweight_profile(tag), reweight_profile)
                training = resolve_training_config("pbpb", "X", "2024", selection_profile)
                fiducial = resolve_fiducial_config("pbpb", "X", fid_profile)
                rng = np.random.RandomState(10)
                probe = pd.DataFrame({
                    "Bpt": rng.uniform(5, 55, 2000),
                    "By": rng.uniform(-2.5, 2.5, 2000),
                    "BQvalue": rng.uniform(0, 0.25, 2000),
                    "Bcos_dtheta": rng.uniform(-1.2, 1.2, 2000),
                    "Btktkpt": rng.uniform(1, 9, 2000),
                    "Bchi2Prob": rng.uniform(-0.1, 1.1, 2000),
                    "Btrk1Pt": rng.uniform(0.5, 5, 2000),
                    "Btrk2Pt": rng.uniform(0.5, 5, 2000),
                    "Btrk1dR": rng.uniform(-0.1, 0.6, 2000),
                    "Btrk2dR": rng.uniform(-0.1, 0.6, 2000),
                })
                expected = apply_selection(probe, fiducial["expression"], "fiducial")
                actual = apply_selection(probe, training["signal_selection"], "signal")
                self.assertEqual(actual.index.tolist(), expected.index.tolist())
                self.assertIn("Btrk1Pt > 0.9", fiducial["expression"])
                self.assertIn("Btrk2Pt > 0.9", fiducial["expression"])
                profile = resolve_training_reweight_config(
                    "pbpb", "X", "2024", reweight_profile,
                    selection_profile, fid_profile,
                )
                self.assertEqual(profile["weight_branch"], "Reweight")
                self.assertEqual(profile["signal"]["tree"], "ntmix_X3872")
                self.assertIn(source_tag, profile["signal"]["path"])
                with self.assertRaisesRegex(ValueError, "requires"):
                    resolve_training_reweight_config(
                        "pbpb", "X", "2024", reweight_profile,
                        "pb24_v7", "pb24_fid8",
                    )

    def test_pbpb_x_common09_nine_variable_set(self):
        self.assertEqual(
            get_varset_columns("pbpb", "9v2", channel="X"),
            [
                "Btrk1dR", "Btrk2dR", "Btktkpt", "Btrk1Pt", "Btrk2Pt",
                "Bchi2Prob", "Bcos_dtheta", "BtrkPtimb", "BtktkvProb",
            ],
        )

    def test_pbpb24_x_r6range3_btrk2dr_profiles(self):
        cases = [
            ("pb24_v11", "pb24_fid12", "rwr6range3dr045v1", 0.45),
            ("pb24_v12", "pb24_fid13", "rwr6range3dr025v1", 0.25),
        ]
        rng = np.random.RandomState(11)
        probe = pd.DataFrame({
            "Bpt": rng.uniform(5, 55, 2000),
            "By": rng.uniform(-2.5, 2.5, 2000),
            "BQvalue": rng.uniform(0, 0.25, 2000),
            "Bcos_dtheta": rng.uniform(-1.2, 1.2, 2000),
            "Btktkpt": rng.uniform(1, 9, 2000),
            "Bchi2Prob": rng.uniform(-0.1, 1.1, 2000),
            "Btrk1Pt": rng.uniform(0.5, 5, 2000),
            "Btrk2Pt": rng.uniform(0.5, 5, 2000),
            "Btrk1dR": rng.uniform(-0.1, 0.6, 2000),
            "Btrk2dR": rng.uniform(0, 0.6, 2000),
        })
        for selection_profile, fid_profile, reweight_profile, upper in cases:
            with self.subTest(reweight_profile=reweight_profile):
                training = resolve_training_config("pbpb", "X", "2024", selection_profile)
                fiducial = resolve_fiducial_config("pbpb", "X", fid_profile)
                expected = apply_selection(probe, fiducial["expression"], "fiducial")
                actual = apply_selection(probe, training["signal_selection"], "signal")
                self.assertEqual(actual.index.tolist(), expected.index.tolist())
                self.assertIn(f"Btrk2dR < {upper}", fiducial["expression"])
                profile = resolve_training_reweight_config(
                    "pbpb", "X", "2024", reweight_profile,
                    selection_profile, fid_profile,
                )
                self.assertEqual(profile["weight_branch"], "Reweight")
                self.assertIn("X_pp24_xsplot_R6range3_rw_v1", profile["signal"]["path"])

    def test_pbpb_x_r6range3_six_variable_set(self):
        self.assertEqual(
            get_varset_columns("pbpb", "6v5", channel="X"),
            ["Btrk1dR", "Btktkpt", "Btrk1Pt", "Bchi2Prob", "Bcos_dtheta", "Btrk2Pt"],
        )

    def test_pbpb24_x_r6range4_to_8_weighted_profiles(self):
        cases = [
            (13, 14, "rwr6range4v1", "R6range4", 0.45),
            (14, 15, "rwr6range5v1", "R6range5", 0.40),
            (15, 16, "rwr6range6v1", "R6range6", 0.35),
            (16, 17, "rwr6range7v1", "R6range7", 0.30),
            (17, 18, "rwr6range8v1", "R6range8", 0.25),
        ]
        rng = np.random.RandomState(12)
        probe = pd.DataFrame({
            "Bpt": rng.uniform(5, 55, 2000),
            "By": rng.uniform(-2.5, 2.5, 2000),
            "BQvalue": rng.uniform(0, 0.25, 2000),
            "Bcos_dtheta": rng.uniform(-1.2, 1.2, 2000),
            "Btktkpt": rng.uniform(1, 9, 2000),
            "Bchi2Prob": rng.uniform(-0.1, 1.1, 2000),
            "Btrk1Pt": rng.uniform(0.5, 5, 2000),
            "Btrk2Pt": rng.uniform(0.5, 5, 2000),
            "Btrk1dR": rng.uniform(-0.1, 0.6, 2000),
            "Btrk2dR": rng.uniform(-0.1, 0.6, 2000),
        })
        for version, fid, reweight_profile, source_tag, upper in cases:
            selection_profile = f"pb24_v{version}"
            fid_profile = f"pb24_fid{fid}"
            tag = (
                f"X_pb24_v{version}_fid{fid}_6v5_"
                f"{reweight_profile}_xgb_v1"
            )
            with self.subTest(tag=tag):
                self.assertEqual(infer_reweight_profile(tag), reweight_profile)
                training = resolve_training_config("pbpb", "X", "2024", selection_profile)
                fiducial = resolve_fiducial_config("pbpb", "X", fid_profile)
                expected = apply_selection(probe, fiducial["expression"], "fiducial")
                actual = apply_selection(probe, training["signal_selection"], "signal")
                self.assertEqual(actual.index.tolist(), expected.index.tolist())
                self.assertIn("Btrk1dR <= 0.45", fiducial["expression"])
                self.assertIn(f"Btrk2dR <= {upper:.2f}", fiducial["expression"])
                profile = resolve_training_reweight_config(
                    "pbpb", "X", "2024", reweight_profile,
                    selection_profile, fid_profile,
                )
                self.assertEqual(profile["weight_branch"], "Reweight")
                self.assertIn(source_tag, profile["signal"]["path"])
                with self.assertRaisesRegex(ValueError, "requires"):
                    resolve_training_reweight_config(
                        "pbpb", "X", "2024", reweight_profile,
                        "pb24_v9", "pb24_fid10",
                    )

    def test_pbpb24_x_r6range4_to_7_common_dr025_profiles(self):
        training = resolve_training_config("pbpb", "X", "2024", "pb24_v18")
        fiducial = resolve_fiducial_config("pbpb", "X", "pb24_fid19")
        self.assertIn("Btrk1dR <= 0.45", fiducial["expression"])
        self.assertIn("Btrk2dR < 0.25", fiducial["expression"])
        rng = np.random.RandomState(13)
        probe = pd.DataFrame({
            "Bpt": rng.uniform(5, 55, 2000),
            "By": rng.uniform(-2.5, 2.5, 2000),
            "BQvalue": rng.uniform(0, 0.25, 2000),
            "Bcos_dtheta": rng.uniform(-1.2, 1.2, 2000),
            "Btktkpt": rng.uniform(1, 9, 2000),
            "Bchi2Prob": rng.uniform(-0.1, 1.1, 2000),
            "Btrk1Pt": rng.uniform(0.5, 5, 2000),
            "Btrk2Pt": rng.uniform(0.5, 5, 2000),
            "Btrk1dR": rng.uniform(-0.1, 0.6, 2000),
            "Btrk2dR": rng.uniform(-0.1, 0.6, 2000),
        })
        expected = apply_selection(probe, fiducial["expression"], "fiducial")
        actual = apply_selection(probe, training["signal_selection"], "signal")
        self.assertEqual(actual.index.tolist(), expected.index.tolist())
        for number in range(4, 8):
            reweight_profile = f"rwr6range{number}dr025v1"
            tag = f"X_pb24_v18_fid19_6v5_{reweight_profile}_xgb_v1"
            with self.subTest(tag=tag):
                self.assertEqual(infer_reweight_profile(tag), reweight_profile)
                profile = resolve_training_reweight_config(
                    "pbpb", "X", "2024", reweight_profile,
                    "pb24_v18", "pb24_fid19",
                )
                self.assertEqual(profile["weight_branch"], "Reweight")
                self.assertIn(f"R6range{number}_rw_v1", profile["signal"]["path"])
                with self.assertRaisesRegex(ValueError, "requires"):
                    resolve_training_reweight_config(
                        "pbpb", "X", "2024", reweight_profile,
                        "pb24_v17", "pb24_fid18",
                    )

    def test_pbpb24_x_r6range5_train40_apply25_profile(self):
        tag = (
            "X_pb24_v14_fid19_6v5_"
            "rwr6range5train40apply25v1_xgb_v1"
        )
        reweight_profile = "rwr6range5train40apply25v1"
        self.assertEqual(infer_reweight_profile(tag), reweight_profile)

        training = resolve_training_config("pbpb", "X", "2024", "pb24_v14")
        fiducial = resolve_fiducial_config("pbpb", "X", "pb24_fid19")
        self.assertIn("Btrk2dR <= 0.40", training["signal_selection"])
        self.assertIn("Btrk2dR <= 0.40", training["background_selection"])
        self.assertIn("Btrk2dR < 0.25", fiducial["expression"])

        profile = resolve_training_reweight_config(
            "pbpb", "X", "2024", reweight_profile,
            "pb24_v14", "pb24_fid19",
        )
        self.assertEqual(profile["weight_branch"], "Reweight")
        self.assertIn("R6range5_rw_v1", profile["signal"]["path"])
        with self.assertRaisesRegex(ValueError, "requires"):
            resolve_training_reweight_config(
                "pbpb", "X", "2024", reweight_profile,
                "pb24_v18", "pb24_fid19",
            )

    def test_pbpb24_x_r6range5_narrow_sideband_profile(self):
        tag = "X_pb24_v19_fid19_6v5_rwr6range5v1_xgb_v1"
        self.assertEqual(infer_reweight_profile(tag), "rwr6range5v1")
        training = resolve_training_config("pbpb", "X", "2024", "pb24_v19")
        fiducial = resolve_fiducial_config("pbpb", "X", "pb24_fid19")
        self.assertEqual(
            training["signal_selection"],
            resolve_training_config("pbpb", "X", "2024", "pb24_v18")[
                "signal_selection"
            ],
        )
        self.assertIn("Bmass > 3.82 and Bmass < 3.85", training["background_selection"])
        self.assertIn("Bmass > 3.89 and Bmass < 3.92", training["background_selection"])
        self.assertNotIn("Bmass > 3.95", training["background_selection"])
        self.assertIn("Btrk2dR < 0.25", training["background_selection"])
        self.assertIn("Btrk2dR < 0.25", fiducial["expression"])
        profile = resolve_training_reweight_config(
            "pbpb", "X", "2024", "rwr6range5v1",
            "pb24_v19", "pb24_fid19",
        )
        self.assertIn("R6range5_rw_v1", profile["signal"]["path"])
        self.assertEqual(profile["weight_branch"], "Reweight")

    def test_pbpb24_x_extra_mc_is_explicit(self):
        extra = resolve_extra_mc_apply_config("pbpb", "X", "2024", "x3872")
        self.assertEqual(extra["samples"]["x3872"]["tree"], "ntmix_X3872")

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
            psi2s_tag = "X_pb24_v7_fid8_8v2_rwpsi2sr5range3v1_xgb_v1"
            psi2s_dag = make_dag(
                Path(temporary_directory), psi2s_tag, False, False, "x3872"
            )
            psi2s_dag_text = psi2s_dag.read_text()
            self.assertIn(f'train_tag="{psi2s_tag}"', psi2s_dag_text)
            self.assertIn("JOB APPLY_EXTRA submit_apply_job.sub", psi2s_dag_text)
            self.assertIn('apply_extra_mc="x3872"', psi2s_dag_text)
            self.assertIn("PARENT APPLY APPLY_EXTRA CHILD DRAW", psi2s_dag_text)

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

    def test_weighted_single_apply_uses_configured_signal_input(self):
        weighted = {
            "signal_path": "weighted.root:events",
            "signal_weight_branch": "Reweight",
        }
        self.assertEqual(
            resolve_apply_mc_input("nominal.root:events", [weighted]),
            "weighted.root:events",
        )
        self.assertEqual(
            resolve_apply_mc_input(
                "nominal.root:events",
                [{"signal_path": "nominal.root:events", "signal_weight_branch": None}],
            ),
            "nominal.root:events",
        )
        self.assertEqual(
            resolve_apply_mc_input("nominal.root:events", [weighted, weighted]),
            "nominal.root:events",
        )

    def test_raw_signal_override_does_not_require_weight_branch(self):
        raw_signal = {
            "signal_path": "raw_psi2s.root:events",
            "signal_input_override": True,
            "signal_weight_branch": None,
        }
        self.assertEqual(
            resolve_apply_mc_input("nominal_x.root:events", [raw_signal]),
            "raw_psi2s.root:events",
        )
        with self.assertRaisesRegex(ValueError, "missing signal_path"):
            resolve_apply_mc_input(
                "nominal_x.root:events",
                [{"signal_input_override": True, "signal_weight_branch": None}],
            )


if __name__ == "__main__":
    unittest.main()
