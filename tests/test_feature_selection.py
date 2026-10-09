import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from configs.feature_selection import FEATURE_SELECTION_EXPERIMENTS, experiment_for_tag, resolve_feature_selection_experiment
from utils import paths
from utils.varsets import get_varset_columns, infer_varset_from_tag
from dag.make_feature_selection_workflow import make_dag
from workflows.rank_feature_selection import ordered_ranking

EXP = "fs_9v9_drop1_X_pb23_v27_fid18_9v9_rw0_xgb_v1"

class FeatureSelectionTest(unittest.TestCase):
    def setUp(self):
        from utils.varsets import VARSETS
        from configs.samples import SAMPLES
        base = "X_pb23_v27_fid18_9v9_rw0_xgb_v1"
        variables = ["a", "b"]
        candidates = [{"train_tag": "X_pb23_v27_fid18_1v1_rw0_xgb_v1", "removed_variable": "a"},
                      {"train_tag": "X_pb23_v27_fid18_1v2_rw0_xgb_v1", "removed_variable": "b"}]
        config = {"baseline": base, "operation": "drop1", "candidates": candidates}
        for context in [patch.dict(FEATURE_SELECTION_EXPERIMENTS, {EXP: config}, clear=True),
                        patch.dict(VARSETS["pbpb"]["X"], {"9v9": variables, "1v1": ["b"], "1v2": ["a"]}, clear=True),
                        patch.dict(SAMPLES["pbpb"]["channels"]["X"]["selection_profiles"],
                                   {"pb23_v27": {"signal_selection": "True", "background_selection": "True"}}, clear=True)]:
            context.start()
            self.addCleanup(context.stop)

    def test_candidates_and_paths(self):
        cfg = resolve_feature_selection_experiment(EXP)
        base = get_varset_columns("pbpb", "9v9", "X")
        self.assertEqual({r["removed_variable"] for r in cfg["candidates"]}, set(base))
        for row in cfg["candidates"]:
            tag = row["train_tag"]
            self.assertEqual(get_varset_columns("pbpb", infer_varset_from_tag(tag), "X"),
                             [v for v in base if v != row["removed_variable"]])
            for root, fn in [(paths.MODELS_DIR, paths.model_dir), (paths.TRAINING_DIR, paths.training_dir),
                             (paths.SELECTED_DIR, paths.selected_dir), (paths.SHAP_DIR, paths.shap_dir)]:
                self.assertEqual(Path(fn(tag)), Path(root)/EXP/tag)
        self.assertEqual(Path(paths.model_dir(cfg["baseline"])), Path(paths.MODELS_DIR)/cfg["baseline"])
        self.assertEqual(paths.condor_model_dir(cfg["candidates"][0]["train_tag"]),
                         paths.model_dir(cfg["candidates"][0]["train_tag"]))

    def test_ambiguous_owner_rejected(self):
        tag = FEATURE_SELECTION_EXPERIMENTS[EXP]["candidates"][0]["train_tag"]
        with patch.dict(FEATURE_SELECTION_EXPERIMENTS, {"duplicate": {"candidates": [{"train_tag": tag}]}}):
            with self.assertRaisesRegex(ValueError, "Ambiguous"):
                experiment_for_tag(tag)

    def test_dag_and_resume_protection(self):
        with tempfile.TemporaryDirectory() as d:
            with patch("dag.make_feature_selection_workflow.Path.is_file", return_value=True):
                dag = make_dag(EXP, d).read_text()
                self.assertEqual(dag.count("JOB "), 1)  # completed candidates reused
            with patch("dag.make_feature_selection_workflow.model_path", return_value=d), \
                 patch("dag.make_feature_selection_workflow.Path.is_file", return_value=False):
                with self.assertRaisesRegex(FileExistsError, "Partial"):
                    make_dag(EXP, d)

    def test_total_auc_only_and_baseline_tie(self):
        rows = [{"train_tag": "candidate", "combined_auc": .8, "test_auc": .95},
                {"train_tag": "baseline", "combined_auc": .81, "test_auc": .7}]
        self.assertEqual(ordered_ranking(rows, "baseline")[0]["train_tag"], "baseline")
        rows[0]["combined_auc"] = .81
        self.assertEqual(ordered_ranking(rows, "baseline")[0]["train_tag"], "baseline")
