import unittest
import tempfile
import json
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from utils.training_weights import (
    balanced_scale_pos_weight,
    resolve_training_weights,
    weighted_ks_curve,
)
from dag.make_single_workflow import make_dag
from dag.make_dagman_workflow import make_dag as make_dagman_dag
from dag.make_reweight_workflow import make_dag as make_reweight_dag
from workflows.reweighting import validate_configured_reweight
from workflows.reweighting.run_configured_job import JOBS
from configs.samples import (
    infer_direct_xgb_version,
    infer_reweight_profile,
    resolve_apply_config,
    resolve_draw_config,
    resolve_extra_mc_apply_config,
    resolve_fiducial_config,
    resolve_training_config,
    resolve_training_reweight_config,
    resolve_training_signal_artifact_variant,
)
from configs.direct_xgb_settings import (
    DIRECT_XGB_PARAMS,
    REQUIRED_DIRECT_XGB_FIELDS,
    resolve_direct_xgb_params,
)
from utils.apply_inputs import resolve_apply_mc_input
from utils.selection import apply_selection
from utils.varsets import get_reweight_varset_columns, get_varset_columns


class TrainingWeightsTest(unittest.TestCase):
    def test_configured_reweight_validation_uses_only_ten_bins_with_sumw2_errors(self):
        self.assertEqual(validate_configured_reweight.BIN_COUNTS, (10,))
        histogram, error = validate_configured_reweight.normalized_histogram(
            np.array([0.2, 1.2, 1.4]),
            np.ones(3),
            np.array([0.0, 1.0, 2.0]),
        )
        np.testing.assert_allclose(histogram, [1.0 / 3.0, 2.0 / 3.0])
        np.testing.assert_allclose(error, [1.0 / 3.0, np.sqrt(2.0) / 3.0])

    def test_reweight_dag_enables_external_validations_by_default(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dag, splot_exists, splot_path = make_reweight_dag(
                Path(tmpdir), "Psi2S_pp24_R5_rw_v1"
            )
            dag_text = dag.read_text()
            disabled, _, _ = make_reweight_dag(
                Path(tmpdir) / "disabled", "Psi2S_pp24_R5_rw_v1",
                with_splot_validation=False,
                with_mc_domain_validation=False,
            )
            disabled_text = disabled.read_text()
        self.assertTrue(splot_exists)
        self.assertTrue(str(splot_path).endswith("PSI2S_btrk2dr_v2.root"))
        self.assertIn("JOB VALIDATE_SPLOT submit_reweight_validation.sub", dag_text)
        self.assertIn("JOB VALIDATE_MC_2023 submit_reweight_validation.sub", dag_text)
        self.assertIn("JOB VALIDATE_MC_2024 submit_reweight_validation.sub", dag_text)
        self.assertNotIn("VALIDATE_SPLOT", disabled_text)
        self.assertNotIn("VALIDATE_MC", disabled_text)

    def test_splot_validation_missing_input_is_a_successful_skip(self):
        spec = deepcopy(JOBS["Psi2S_pp24_R5_rw_v1"])
        spec["validation_splot"] = {
            "path": "/definitely/missing/splot.root",
            "tree": "events",
            "weight_branch": "signal_sWeight",
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(validate_configured_reweight.JOBS, {"test_rw": spec}, clear=False):
                with patch.object(validate_configured_reweight, "REPO_ROOT", Path(tmpdir)):
                    validate_configured_reweight.validate_splot("test_rw")
            manifest = Path(tmpdir) / "output/reweighting/test_rw/validation/splot_vs_rw/manifest.json"
            payload = json.loads(manifest.read_text())
        self.assertEqual(payload["status"], "skipped_missing_splot")

    def test_shap_and_fit_interface_are_default_for_single_and_optuna_dags(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            single = make_dag(
                Path(tmpdir), "X_pb24_v19_fid19_6v5_rwr6range5v1_xgb_v1"
            ).read_text()
            optuna = make_dagman_dag(
                Path(tmpdir), "X_pb24_v2_fid1_8v1_1o5_v1", "explicit",
                1, 1, 5, "auto",
            ).read_text()
        for dag_text in (single, optuna):
            self.assertIn("submit_shap_job.sub", dag_text)
            self.assertIn("CHILD SHAP", dag_text.replace("SH_v1", "SHAP"))
            self.assertIn("submit_fit_interface_job.sub", dag_text)
            self.assertIn("CHILD FIT_INTERFACE", dag_text.replace("FI_v1", "FIT_INTERFACE"))

    def test_direct_xgb_versions_are_sample_scoped_and_complete(self):
        pbpb_tag = "Psi2S_pb23_v1_fid1_6v1_rwr6range4v1_xgb_v1"
        pp_tag = "X_pp24_v4_fid3_8v2_rwpsi2sr5v1_xgb_v1"
        self.assertEqual(infer_direct_xgb_version(pbpb_tag), 1)
        self.assertEqual(infer_direct_xgb_version(pp_tag), 1)

        pbpb = resolve_direct_xgb_params("pbpb", 1)
        pp = resolve_direct_xgb_params("pp", 1)
        self.assertEqual(set(pbpb), REQUIRED_DIRECT_XGB_FIELDS)
        self.assertEqual(set(pp), REQUIRED_DIRECT_XGB_FIELDS)
        self.assertNotEqual(pbpb, pp)

        pbpb["max_depth"] = 999
        self.assertEqual(DIRECT_XGB_PARAMS["pbpb"][1]["max_depth"], 4)
        with self.assertRaisesRegex(ValueError, "xgb_v2"):
            resolve_direct_xgb_params("pbpb", 2)

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
        self.assertEqual(set(DIRECT_XGB_PARAMS["pbpb"][1]), REQUIRED_DIRECT_XGB_FIELDS)

        x_training = resolve_training_config("pbpb", "X", "2024", "pb24_v17")
        self.assertEqual(training["background"]["path"], x_training["background"]["path"])
        self.assertNotEqual(training["signal"]["tree"], x_training["signal"]["tree"])

    def test_psi2s_pbpb23_nominal_configuration(self):
        tag = "Psi2S_pb23_v1_fid1_6v1_rwr6range4v1_xgb_v1"
        self.assertEqual(infer_reweight_profile(tag), "rwr6range4v1")

        training23 = resolve_training_config("pbpb", "Psi2S", "2023", "pb23_v1")
        training24 = resolve_training_config("pbpb", "Psi2S", "2024", "pb24_v1")
        fiducial23 = resolve_fiducial_config("pbpb", "Psi2S", "pb23_fid1")
        fiducial24 = resolve_fiducial_config("pbpb", "Psi2S", "pb24_fid1")
        self.assertEqual(training23["signal"]["tree"], "ntmix_PSI2S")
        self.assertTrue(training23["signal"]["path"].endswith("flat_ntmix_PbPb23_MC_PSI2S.root"))
        self.assertTrue(training23["background"]["path"].endswith("flat_ntmix_PbPb23_DATA.root"))
        self.assertEqual(training23["signal_selection"], training24["signal_selection"])
        self.assertEqual(training23["background_selection"], training24["background_selection"])
        self.assertEqual(fiducial23["expression"], fiducial24["expression"])

        profile = resolve_training_reweight_config(
            "pbpb", "Psi2S", "2023", "rwr6range4v1", "pb23_v1", "pb23_fid1"
        )
        self.assertEqual(profile["signal"]["tree"], "ntmix_PSI2S")
        self.assertEqual(profile["weight_branch"], "Reweight")
        self.assertTrue(
            profile["signal"]["path"].endswith(
                "flat_ntmix_PbPb23_MC_PSI2S_with_reweight.root"
            )
        )

        apply_cfg = resolve_apply_config("pbpb", "Psi2S", "2023")
        extra_cfg = resolve_extra_mc_apply_config("pbpb", "Psi2S", "2023", "x3872")
        self.assertEqual(apply_cfg["mc"][0]["tree"], "ntmix_PSI2S")
        self.assertEqual(extra_cfg["samples"]["x3872"]["tree"], "ntmixX3872")
        self.assertEqual(infer_direct_xgb_version(tag), 1)
        self.assertEqual(
            resolve_direct_xgb_params("pbpb", 1), DIRECT_XGB_PARAMS["pbpb"][1]
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            dag_path = make_dag(
                Path(tmpdir), tag, with_shap=True, use_precut=False,
                apply_extra_mc="x3872",
                pre_reweight_job="Psi2S_pb23_R6range4_migrate_v1",
            )
            dag_text = dag_path.read_text()
        self.assertIn('reweight_job="Psi2S_pb23_R6range4_migrate_v1"', dag_text)
        self.assertIn("PARENT REWEIGHT CHILD TRAIN", dag_text)
        self.assertIn("PARENT TRAIN CHILD APPLY APPLY_EXTRA", dag_text)
        self.assertIn("PARENT DRAW CHILD SHAP", dag_text)

    def test_psi2s_bmu_aux_training_profiles_preserve_baseline_profiles(self):
        for year, selection, fiducial in (
            ("2023", "pb23_v1", "pb23_fid1"),
            ("2024", "pb24_v1", "pb24_fid1"),
        ):
            baseline = resolve_training_reweight_config(
                "pbpb", "Psi2S", year, "rwr6range4v1", selection, fiducial
            )
            expanded = resolve_training_reweight_config(
                "pbpb", "Psi2S", year, "rwr6range4bmuauxv1", selection, fiducial
            )
            self.assertEqual(expanded["weight_branch"], baseline["weight_branch"])
            self.assertEqual(
                expanded["required_selection_profile"],
                baseline["required_selection_profile"],
            )
            self.assertEqual(
                expanded["required_fid_profile"],
                baseline["required_fid_profile"],
            )
            self.assertIn("expanded_training_v1", expanded["signal"]["path"])

    def test_x_bmu_aux_training_profiles_preserve_baseline_profiles(self):
        for year, selection, fiducial in (
            ("2023", "pb23_v3", "pb23_fid3"),
            ("2024", "pb24_v19", "pb24_fid19"),
        ):
            baseline = resolve_training_reweight_config(
                "pbpb", "X", year, "rwr6range5v1", selection, fiducial
            )
            expanded = resolve_training_reweight_config(
                "pbpb", "X", year, "rwr6range5bmuauxv1", selection, fiducial
            )
            self.assertEqual(expanded["weight_branch"], baseline["weight_branch"])
            self.assertEqual(expanded["required_selection_profile"], selection)
            self.assertEqual(expanded["required_fid_profile"], fiducial)
            self.assertIn("expanded_training_v1", expanded["signal"]["path"])

    def test_x_muon_artifact_variant_keeps_reweight_profile_semantics(self):
        baseline_columns = get_varset_columns("pbpb", "6v5", "X")
        self.assertIsNone(
            resolve_training_signal_artifact_variant(
                "pbpb", "X", "2023", "rwr6range5v1", baseline_columns
            )
        )
        for year, varset in (
            ("2023", "7v5"),
            ("2024", "8v4"),
            ("2024", "8v5"),
            ("2023", "8v6"),
            ("2024", "9v3"),
            ("2023", "10v1"),
        ):
            variant = resolve_training_signal_artifact_variant(
                "pbpb", "X", year, "rwr6range5v1",
                get_varset_columns("pbpb", varset, "X"),
            )
            self.assertEqual(variant["artifact_variant"], "bmuaux_v2")
            self.assertIn("expanded_training_v2", variant["signal"]["path"])
            dataset = "pb23" if year == "2023" else "pb24"
            selection = "v3_fid3" if year == "2023" else "v19_fid19"
            tag = f"X_{dataset}_{selection}_{varset}_rwr6range5v1_xgb_v1"
            self.assertEqual(infer_reweight_profile(tag), "rwr6range5v1")

    def test_psi2s_r5_without_btktkpt_configuration(self):
        self.assertEqual(
            get_varset_columns("pbpb", "5v1", "Psi2S"),
            ["Btrk1dR", "Btrk1Pt", "Bchi2Prob", "Bcos_dtheta", "Btrk2Pt"],
        )
        self.assertEqual(
            get_reweight_varset_columns("pp", "R5", "Psi2S"),
            ["Bcos_dtheta", "Bchi2Prob", "Btrk2Pt", "Btrk1Pt", "Btrk1dR"],
        )

        pp = resolve_training_config("pp", "Psi2S", "2024", "pp24_v2")
        self.assertNotIn("Btktkpt", pp["signal_selection"])
        for year, selection, fiducial in (
            ("2023", "pb23_v2", "pb23_fid2"),
            ("2024", "pb24_v2", "pb24_fid2"),
        ):
            training = resolve_training_config("pbpb", "Psi2S", year, selection)
            fid = resolve_fiducial_config("pbpb", "Psi2S", fiducial)
            self.assertNotIn("Btktkpt", training["signal_selection"])
            self.assertNotIn("Btktkpt", fid["expression"])
            profile = resolve_training_reweight_config(
                "pbpb", "Psi2S", year, "rwr5v1", selection, fiducial
            )
            self.assertEqual(profile["weight_branch"], "Reweight")
            self.assertIn("Psi2S_pp24_R5_rw_v1", profile["signal"]["path"])

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

    def test_pbpb23_x_narrow_background_weighted_profile(self):
        tag = "X_pb23_v3_fid3_6v5_rwr6range5v1_xgb_v1"
        self.assertEqual(infer_reweight_profile(tag), "rwr6range5v1")
        training = resolve_training_config("pbpb", "X", "2023", "pb23_v3")
        fiducial = resolve_fiducial_config("pbpb", "X", "pb23_fid3")
        self.assertEqual(training["signal"]["tree"], "ntmixX3872")
        self.assertIn("Bmass > 3.82 and Bmass < 3.85", training["background_selection"])
        self.assertIn("Bmass > 3.89 and Bmass < 3.92", training["background_selection"])
        for expression in (training["signal_selection"], fiducial["expression"]):
            self.assertIn("Btrk1Pt > 0.9", expression)
            self.assertIn("Btrk2Pt > 0.9", expression)
            self.assertIn("Btrk2dR < 0.25", expression)

        profile = resolve_training_reweight_config(
            "pbpb", "X", "2023", "rwr6range5v1", "pb23_v3", "pb23_fid3"
        )
        self.assertEqual(profile["signal"]["tree"], "ntmixX3872")
        self.assertEqual(profile["weight_branch"], "Reweight")
        self.assertTrue(
            profile["signal"]["path"].endswith(
                "flat_ntmix_PbPb23_MCX3872_with_reweight.root"
            )
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            dag_path = make_dag(
                Path(tmpdir), tag, with_shap=False, use_precut=False,
                pre_reweight_job="X_pb23_R6range5_migrate_v1",
            )
            dag_text = dag_path.read_text()
        self.assertIn("JOB REWEIGHT submit_reweight_job.sub", dag_text)
        self.assertIn("PARENT REWEIGHT CHILD TRAIN", dag_text)
        self.assertIn('reweight_job="X_pb23_R6range5_migrate_v1"', dag_text)

    def test_x_r5v3_without_btktkpt_configuration(self):
        expected = ["Btrk1dR", "Btrk1Pt", "Bchi2Prob", "Bcos_dtheta", "Btrk2Pt"]
        self.assertEqual(get_reweight_varset_columns("pp", "R5v3", "X"), expected)
        self.assertEqual(get_varset_columns("pbpb", "5v3", "X"), expected)

        pp = resolve_training_config("pp", "X", "2024", "pp24_v5")
        self.assertNotIn("Btktkpt", pp["signal_selection"])
        self.assertIn("Btrk2dR <= 0.4", pp["signal_selection"])
        for year, selection, fiducial, profile_name in (
            ("2023", "pb23_v4", "pb23_fid4", "rwr5v3v1"),
            ("2024", "pb24_v20", "pb24_fid20", "rwr5v3v1"),
        ):
            training = resolve_training_config("pbpb", "X", year, selection)
            fid = resolve_fiducial_config("pbpb", "X", fiducial)
            self.assertNotIn("Btktkpt", training["signal_selection"])
            self.assertNotIn("Btktkpt", training["background_selection"])
            self.assertNotIn("Btktkpt", fid["expression"])
            self.assertIn("Btrk2dR < 0.25", training["signal_selection"])
            profile = resolve_training_reweight_config(
                "pbpb", "X", year, profile_name, selection, fiducial
            )
            self.assertEqual(profile["weight_branch"], "Reweight")
            self.assertIn("X_pp24_xsplot_R5v3_rw_v1", profile["signal"]["path"])

        with tempfile.TemporaryDirectory() as tmpdir:
            dag, splot_exists, _ = make_reweight_dag(
                Path(tmpdir), "X_pp24_xsplot_R5v3_rw_v1"
            )
            dag_text = dag.read_text()
        self.assertTrue(splot_exists)
        self.assertIn("X_pb23_R5v3_migrate_v1", dag_text)
        self.assertIn("VALIDATE_MC_2023", dag_text)
        self.assertIn("VALIDATE_MC_2024", dag_text)
        self.assertIn("VALIDATE_SPLOT", dag_text)

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

    def test_x_baseline_expanded_sideband_profiles(self):
        cases = (
            ("2023", "pb23_v3", "pb23_v5", "pb23_fid3", "ntmixX3872"),
            ("2024", "pb24_v19", "pb24_v21", "pb24_fid19", "ntmix_X3872"),
        )
        for year, baseline_profile, comparison_profile, fid_profile, tree in cases:
            baseline = resolve_training_config(
                "pbpb", "X", year, baseline_profile
            )
            comparison = resolve_training_config(
                "pbpb", "X", year, comparison_profile
            )
            self.assertEqual(
                comparison["signal_selection"], baseline["signal_selection"]
            )
            self.assertIn(
                "Bmass > 3.80 and Bmass < 3.85",
                comparison["background_selection"],
            )
            self.assertIn(
                "Bmass > 3.89 and Bmass < 3.94",
                comparison["background_selection"],
            )
            self.assertNotIn(
                "Bmass > 3.82 and Bmass < 3.85",
                comparison["background_selection"],
            )
            profile = resolve_training_reweight_config(
                "pbpb", "X", year, "rwr6range5v1",
                comparison_profile, fid_profile,
            )
            self.assertEqual(profile["signal"]["tree"], tree)
            self.assertIn("R6range5_rw_v1", profile["signal"]["path"])
            self.assertEqual(profile["weight_branch"], "Reweight")

    def test_x_baseline_broad_sideband_profiles(self):
        cases = (
            ("2023", "pb23_v3", "pb23_v6", "pb23_fid3", "ntmixX3872"),
            ("2024", "pb24_v19", "pb24_v22", "pb24_fid19", "ntmix_X3872"),
        )
        for year, baseline_profile, comparison_profile, fid_profile, tree in cases:
            baseline = resolve_training_config(
                "pbpb", "X", year, baseline_profile
            )
            comparison = resolve_training_config(
                "pbpb", "X", year, comparison_profile
            )
            self.assertEqual(
                comparison["signal_selection"], baseline["signal_selection"]
            )
            self.assertIn(
                "Bmass > 3.75 and Bmass < 3.85",
                comparison["background_selection"],
            )
            self.assertIn(
                "Bmass > 3.89 and Bmass < 3.99",
                comparison["background_selection"],
            )
            profile = resolve_training_reweight_config(
                "pbpb", "X", year, "rwr6range5v1",
                comparison_profile, fid_profile,
            )
            self.assertEqual(profile["signal"]["tree"], tree)
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
