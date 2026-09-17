import contextlib
import io
import json
import runpy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import uproot
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

from configs import samples, optuna_spaces
from utils import paths
from utils.training_weights import (
    resolve_training_weights, training_weight_branches, weight_branches_from_metadata,
)


class RecordingClassifier(XGBClassifier):
    calls = []

    def fit(self, X, y, **kwargs):
        self.calls.append({"X": X.copy(), "y": np.asarray(y).copy(),
                           "kwargs": kwargs.copy(), "scale_pos_weight": self.get_params()["scale_pos_weight"]})
        return super().fit(X, y, **kwargs)


class PthatWeightsTest(unittest.TestCase):
    def test_factors_and_invalid_values(self):
        frame = pd.DataFrame({"pThatreweight": [.1, .5, 2.], "Reweight": [3., 2., .5]})
        factors = training_weight_branches("pThatreweight", "Reweight")
        np.testing.assert_allclose(resolve_training_weights(frame, factors, "MC"), [.3, 1., 1.])
        with self.assertRaisesRegex(ValueError, "missing"):
            resolve_training_weights(frame.drop(columns="pThatreweight"), factors, "MC")
        for values in ([1., -1., 1.], [1., np.nan, 1.], [1., np.inf, 1.], [0., 0., 0.]):
            bad = frame.copy(); bad["pThatreweight"] = values
            with self.assertRaises(ValueError): resolve_training_weights(bad, factors, "MC")
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            resolve_training_weights(frame, ["pThatreweight"] * 2, "MC")
        self.assertEqual(weight_branches_from_metadata({"signal_weight_branch": "Reweight"}), ["Reweight"])
        self.assertEqual(weight_branches_from_metadata({"signal_weight_branch": "Reweight", "signal_weight_branches": factors}), factors)
        self.assertEqual(weight_branches_from_metadata({"signal_weight_branch": "Reweight", "signal_weight_branches": []}), [])

    def test_resolver(self):
        for sample, channel, year, profile in [("pp", "X", "2024", "pp24_v6"), ("pbpb", "X", "2023", "pb23_v3"), ("pp", "Psi2S", "2024", "pp24_v1")]:
            cfg = samples.resolve_training_config(sample, channel, year, profile)
            self.assertEqual(cfg["signal_base_weight_branch"], "pThatreweight")
        cfg = samples.resolve_training_config("pbpb", "X", "2023", "pb23_v3")
        self.assertTrue(cfg["signal"]["path"].endswith("flat_ntmix_PbPb23_MC_X3872.root"))
        self.assertEqual(cfg["signal"]["tree"], "ntmix_X3872")

    def run_training(self, optuna=False, combined=False):
        import workflows.xgboost_train_direct as direct
        from utils.varsets import get_varset_columns
        features = get_varset_columns("pp", "8v2", "X")
        tag = "X_pp24_v6_fid4_8v2_1o1_v1" if optuna else "X_pp24_v6_fid4_8v2_xgb_v1"
        rng = np.random.RandomState(31)
        signal = pd.DataFrame({k: rng.normal(.5, 1, 80) for k in features})
        background = pd.DataFrame({k: rng.normal(0, 1, 100) for k in features})
        signal["pThatreweight"] = np.linspace(.1, 2., len(signal))
        signal["Reweight"] = np.linspace(2., .5, len(signal))
        expected = signal["pThatreweight"].to_numpy() * (signal["Reweight"].to_numpy() if combined else 1.)
        RecordingClassifier.calls = []
        with tempfile.TemporaryDirectory() as directory, contextlib.ExitStack() as stack:
            root = Path(directory)
            for name, frame in (("signal", signal), ("background", background)):
                with uproot.recreate(root / f"{name}.root") as f:
                    f["events"] = {k: frame[k].to_numpy() for k in frame}
            original_resolver = samples.resolve_training_config
            def resolver(*args):
                cfg = original_resolver(*args)
                cfg.update(signal={"path": str(root / "signal.root"), "tree": "events"},
                           background={"path": str(root / "background.root"), "tree": "events"},
                           signal_selection=None, background_selection=None)
                return cfg
            stack.enter_context(patch.object(samples, "resolve_training_config", resolver))
            stack.enter_context(patch.object(direct, "resolve_training_config", resolver))
            for key in ("MODELS_DIR", "TRAINING_DIR", "SHAP_DIR", "SELECTED_DIR"):
                stack.enter_context(patch.object(paths, key, str(root / key.lower())))
            stack.enter_context(patch("sys.argv", ["training", tag]))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            if optuna:
                stack.enter_context(patch("xgboost.XGBClassifier", RecordingClassifier))
                stack.enter_context(patch.dict(optuna_spaces.OPTUNA_SPACES["pp24"]["X"], {"v1": {
                    "n_estimators": {"type": "int", "low": 4, "high": 4},
                    "max_depth": {"type": "int", "low": 2, "high": 2}}}))
                stack.enter_context(patch.dict(optuna_spaces.OPTUNA_TRAINING_OPTIONS["pp24"]["X"], {"v1": {"early_stopping_rounds": 2}}))
                runpy.run_module("workflows.condor_optuna_XGBoost", run_name="__main__")
            else:
                stack.enter_context(patch.object(direct, "XGBClassifier", RecordingClassifier))
                stack.enter_context(patch.object(direct, "resolve_direct_xgb_params", return_value={"n_estimators": 4, "max_depth": 2, "n_jobs": 1}))
                if combined:
                    cfg = {"signal": {"path": str(root / "signal.root"), "tree": "events"}, "weight_branch": "Reweight"}
                    stack.enter_context(patch.object(direct, "resolve_training_reweight_config", return_value=cfg))
                direct.main()
            config = json.loads(Path(paths.condor_model_config_path(tag)).read_text())
            factors = ["pThatreweight", "Reweight"] if combined else ["pThatreweight"]
            self.assertEqual(config["signal_weight_branches"], factors)
            self.assertEqual(config["efficiency_reference_weight_branches"], factors)
            metadata = json.loads((Path(paths.condor_model_dir(tag)) / "run_metadata.json").read_text())
            self.assertEqual(metadata["notes"]["signal_weight_branches"], factors)
            all_weights = np.r_[expected, np.ones(len(background))]
            for call in RecordingClassifier.calls:
                kwargs = call["kwargs"]
                want = all_weights[call["X"].index.to_numpy()]
                np.testing.assert_allclose(kwargs["sample_weight"], want)
                self.assertAlmostEqual(call["scale_pos_weight"], want[~call["y"].astype(bool)].sum() / want[call["y"].astype(bool)].sum())
                for (X, y), weights in zip(kwargs["eval_set"], kwargs["sample_weight_eval_set"]):
                    np.testing.assert_allclose(weights, all_weights[X.index.to_numpy()])
            self.assertEqual(len(RecordingClassifier.calls), 2 if optuna else 1)
            import joblib
            model = joblib.load(paths.condor_model_path(tag))
            final = RecordingClassifier.calls[-1]
            auc_expected = roc_auc_score(final["y"], model.predict_proba(final["X"])[:, 1], sample_weight=final["kwargs"]["sample_weight"])
            roc = json.loads((Path(paths.condor_training_dir(tag)) / "roc.json").read_text())
            self.assertAlmostEqual(roc["train"]["auc"], auc_expected)

    def test_direct_pthat(self): self.run_training()
    def test_direct_product(self): self.run_training(combined=True)
    def test_optuna_pthat(self): self.run_training(optuna=True)


if __name__ == "__main__":
    unittest.main()
