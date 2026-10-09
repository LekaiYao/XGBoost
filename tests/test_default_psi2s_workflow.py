import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from configs.samples import default_apply_extra_mc
from dag.make_single_workflow import make_dag as single
from dag.make_dagman_workflow import make_dag as batch
from dag import make_feature_selection_workflow as feature


class DefaultPsi2STest(unittest.TestCase):
    def test_single_default_and_explicit_opt_out(self):
        with tempfile.TemporaryDirectory() as tmp:
            for tag in ("X_pb18_v1_fid1_3v3_rw0_xgb_v1", "X_pb23_v9_fid1_3v3_rw0_xgb_v1"):
                self.assertEqual(default_apply_extra_mc(tag), "psi2s")
                text = single(Path(tmp), tag).read_text()
                self.assertIn('apply_extra_mc="psi2s"', text)
                self.assertIn("PARENT TRAIN CHILD APPLY APPLY_EXTRA", text)
                self.assertIn("PARENT APPLY APPLY_EXTRA CHILD DRAW", text)
                self.assertNotIn("JOB APPLY_EXTRA", single(Path(tmp), tag, apply_extra_mc="0").read_text())
            tag = "Bd_pb23_v1_fid1_9v1_xgb_v1"
            self.assertEqual(default_apply_extra_mc(tag), "0")
            self.assertNotIn("JOB APPLY_EXTRA", single(Path(tmp), tag, use_precut=True).read_text())

    def test_batch_dependencies(self):
        with tempfile.TemporaryDirectory() as tmp:
            text = batch(Path(tmp), "X_pb23_v9_fid1_3v3_1o10", "legacy", 1, 2, 10, "auto").read_text()
            for n in (1, 2):
                self.assertIn(f"PARENT TR_v{n} CHILD PSI_v{n}", text)
                self.assertIn(f"PARENT AP_v{n} PSI_v{n} CHILD DR_v{n}", text)
            self.assertEqual(text.count('apply_extra_mc="psi2s"'), 2)

    def test_feature_new_and_completed_missing_companion(self):
        tag = "X_pb23_v9_fid1_3v3_rw0_xgb_v1"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model, selected, shap = root / "model", root / "selected", root / "shap"
            with patch.object(feature, "resolve_feature_selection_experiment", return_value={"baseline": tag, "candidates": []}), patch.object(feature, "model_path", return_value=model), patch.object(feature, "selected_dir", return_value=selected), patch.object(feature, "punzi_dir", return_value=selected / "punzi"), patch.object(feature, "shap_importance_path", return_value=shap), patch.object(feature, "shap_importance_fraction_path", return_value=shap), patch.object(feature, "shap_summary_path", return_value=shap), patch.object(feature, "shap_bar_path", return_value=shap), patch.object(feature, "shap_cumulative_path", return_value=shap):
                text = feature.make_dag("mock", root / "dags").read_text()
                self.assertIn("PARENT TRAIN_0 CHILD PSI_0", text)
                self.assertIn("PARENT PSI_0 CHILD DRAW_0", text)
                self.assertIn("PARENT SHAP_0 PSI_0 CHILD RANK", text)
                for p in (model, shap, selected / "batch_apply_summary.json", selected / "cut_scan/thresholds.json", selected / "punzi/scan.json"):
                    p.parent.mkdir(parents=True, exist_ok=True)
                    p.touch()
                text = feature.make_dag("mock", root / "dags").read_text()
                self.assertNotIn("JOB TRAIN", text)
                self.assertIn("JOB PSI_0", text)
                self.assertIn("PARENT PSI_0 CHILD RANK", text)
                (selected / "MC_psi2s_with_score.root").touch()
                text = feature.make_dag("mock", root / "dags").read_text()
                self.assertNotIn("JOB PSI_0", text)


if __name__ == "__main__":
    unittest.main()
