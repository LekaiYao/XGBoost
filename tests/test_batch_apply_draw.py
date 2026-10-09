"""Exercise APPLY -> DRAW on tiny ROOT inputs, without training a model."""
import contextlib
import io
import json
import runpy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import uproot

from configs import samples
from utils import paths
from utils.score_thresholds import weighted_efficiency_thresholds
from workflows.integration.export_x_fit_scan_manifest import build_fit_scan_manifest


TAG = "X_pb24_v18_fid19_6v5_rwr6range7dr025v1_xgb_v1"
GROUP_TAG = "X_pb24_v2_fid1_8v1_1o5_v1"


class IdentityScaler:
    def transform(self, frame):
        return frame.to_numpy(dtype=float)


class ScoreModel:
    def predict_proba(self, frame):
        score = frame.to_numpy(dtype=float)[:, 0] / 10
        return np.column_stack([1 - score, score])


class BatchApplyDrawTest(unittest.TestCase):
    def setUp(self):
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.configs = {}
        # Synthetic APPLY fixtures are independent of archived production recipes.
        from utils.varsets import VARSETS
        self.stack.enter_context(patch.dict(VARSETS["pbpb"]["X"], {"6v5": ["x"], "8v1": ["x"]}))
        channel = samples.SAMPLES["pbpb"]["channels"]["X"]
        self.stack.enter_context(patch.dict(channel["selection_profiles"], {"pb24_v18": {}, "pb24_v2": {}}))
        self.stack.enter_context(patch.dict(channel["fiducial_profiles"], {"pb24_fid19": "x > 0", "pb24_fid1": "x > 0"}))
        self.arrays = {
            "x": np.linspace(0, 9, 40), "Bmass": np.full(40, 3.87),
            "Reweight": np.linspace(1, 3, 40),
            "pThatreweight": np.linspace(3, 1, 40),
        }
        raw = self.root / "raw.root"
        self.write_root(raw, "ntmix", self.arrays)
        spec = {"path": str(raw), "tree": "ntmix"}
        self.stack.enter_context(patch.object(paths, "SELECTED_DIR", str(self.root / "selected")))
        self.stack.enter_context(patch.object(paths, "resolve_model_config_path", side_effect=lambda tag: str(self.root / (tag + ".json"))))
        self.stack.enter_context(patch.object(paths, "resolve_model_path", side_effect=lambda tag: tag + ".model"))
        self.stack.enter_context(patch.object(paths, "resolve_scaler_path", side_effect=lambda tag: tag + ".scaler"))
        self.stack.enter_context(patch("joblib.load", side_effect=self.load_model))
        self.stack.enter_context(patch.object(samples, "resolve_apply_config", return_value={
            "mc": [spec], "data": [spec], "dataset_source": "fixture",
        }))
        self.stack.enter_context(patch.object(samples, "resolve_draw_config", return_value={
            "data": spec, "plot": {"mass_range": [3.8, 4.0], "bin_width": .01},
        }))
        self.stack.enter_context(patch.object(samples, "resolve_training_config", return_value={
            "signal_selection": "x > 0", "background_selection": "x > 0",
            "apply_fiducial_selection": True,
        }))
        self.stack.enter_context(patch.object(samples, "resolve_fiducial_config", return_value={"expression": "x > 0"}))
        self.stack.enter_context(patch.object(plt, "savefig"))
        self.stack.enter_context(patch.object(plt, "tight_layout"))

    @staticmethod
    def write_root(path, tree, arrays):
        with uproot.recreate(path) as f:
            f.mktree(tree, arrays)

    def load_model(self, path):
        tag, kind = path.rsplit(".", 1)
        if tag not in self.configs:
            raise FileNotFoundError(tag)
        return ScoreModel() if kind == "model" else IdentityScaler()

    def add_model(self, tag, factors, legacy=False, reference=True, reverse=False):
        cfg = {"input_columns": ["x"], "trans_columns": ["x_trans"], "signal_input_override": False}
        arrays = {key: value.copy() for key, value in self.arrays.items()}
        if reverse:
            arrays["Reweight"] = arrays["Reweight"][::-1]
        if reference:
            source = self.root / (tag + ".root")
            self.write_root(source, "reference", arrays)
            cfg["efficiency_reference_signal"] = str(source) + ":reference"
            prefix = "efficiency_reference"
        else:
            prefix = "signal"
        if legacy:
            cfg[prefix + "_weight_branch"] = factors[0] if factors else None
        else:
            cfg[prefix + "_weight_branches"] = factors
        self.configs[tag] = cfg
        (self.root / (tag + ".json")).write_text(json.dumps(cfg))
        return arrays

    def run_workflow(self, tags, prefix=""):
        for module in ("workflows.batch_apply_scores", "workflows.batch_draw_scores"):
            with patch("sys.argv", [module, "--output-prefix", prefix, *tags]), contextlib.redirect_stdout(io.StringIO()):
                runpy.run_module(module, run_name="__main__")
        return Path(paths.selected_dir(paths.train_group_tag(tags)))

    def check_thresholds(self, path, arrays, factors):
        payload = json.loads(path.read_text())
        self.assertEqual(payload["weight_branches"], factors)
        self.assertEqual(payload["weight_branch"], factors[0] if len(factors) == 1 else None)
        mask = arrays["x"] > 0
        weights = np.ones(mask.sum())
        for factor in factors:
            weights *= arrays[factor][mask]
        expected = weighted_efficiency_thresholds(
            arrays["x"][mask] / 10, weights,
            [row["target_efficiency"] for row in payload["thresholds"]],
        )
        np.testing.assert_allclose(
            [row["score_threshold"] for row in payload["thresholds"]],
            [row["score_threshold"] for row in expected],
        )
        return payload

    def test_single_legacy_reference_draw_exports_manifest(self):
        arrays = self.add_model(TAG, ["Reweight"], legacy=True)
        directory = self.run_workflow([TAG])
        self.check_thresholds(directory / "cut_scan/thresholds.json", arrays, ["Reweight"])
        manifest = build_fit_scan_manifest(TAG, directory)
        self.assertEqual(manifest["threshold_provenance"]["event_weight_branch"], "Reweight")
        self.assertEqual(len(manifest["working_points"]), 7)

    def test_product_weights_do_not_claim_legacy_reweight_contract(self):
        factors = ["pThatreweight", "Reweight"]
        arrays = self.add_model(TAG, factors)
        directory = self.run_workflow([TAG])
        self.check_thresholds(directory / "cut_scan/thresholds.json", arrays, factors)
        with self.assertRaisesRegex(ValueError, "threshold weight branch"):
            build_fit_scan_manifest(TAG, directory)

    def test_grouped_references_keep_each_model_weights_and_prefix(self):
        second = GROUP_TAG[:-1] + "2"
        first_arrays = self.add_model(GROUP_TAG, ["Reweight"], legacy=True)
        second_arrays = self.add_model(second, ["pThatreweight", "Reweight"], reverse=True)
        directory = self.run_workflow([GROUP_TAG, second], prefix="check_")
        for tag, arrays, factors in ((GROUP_TAG, first_arrays, ["Reweight"]), (second, second_arrays, ["pThatreweight", "Reweight"])):
            payload = self.check_thresholds(directory / "cut_scan" / ("check_" + tag + "_thresholds.json"), arrays, factors)
            with uproot.open(payload["reference_file"]) as f:
                tree = f["reference"]
                self.assertIn("Prediction_" + tag, tree.keys())
                self.assertEqual(tree.num_entries, 39)
        self.assertFalse((directory / "check_REFERENCE_MC_with_score.root").exists())

    def test_missing_model_is_skipped_without_losing_grouped_reference(self):
        arrays = self.add_model(GROUP_TAG, ["pThatreweight"])
        directory = self.run_workflow([GROUP_TAG, GROUP_TAG[:-1] + "2"])
        self.check_thresholds(directory / "cut_scan" / (GROUP_TAG + "_thresholds.json"), arrays, ["pThatreweight"])

    def test_unweighted_model_uses_compact_mc(self):
        arrays = self.add_model(TAG, [], reference=False)
        directory = self.run_workflow([TAG])
        payload = self.check_thresholds(directory / "cut_scan/thresholds.json", arrays, [])
        self.assertTrue(payload["reference_file"].endswith("MC_with_score.root"))


if __name__ == "__main__":
    unittest.main()
