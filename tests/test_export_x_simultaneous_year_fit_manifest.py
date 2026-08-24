import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import uproot

from workflows.integration import export_x_simultaneous_year_fit_manifest as exporter


ANCHOR = "X_pb23_v3_fid3_6v5_rwr6range5v1_xgb_v1"
PB24 = "X_pb24_v19_fid19_6v5_rwr6range5v1_xgb_v1"


def _write_root(path):
    with uproot.recreate(path) as root_file:
        root_file.mktree("ntmix", {
            "Bmass": np.array([3.84, 3.872, 3.91], dtype=np.float32),
            "Prediction": np.array([0.90, 0.95, 0.99], dtype=np.float32),
        })


class ExportXSimultaneousYearFitManifestTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.directories = {ANCHOR: self.root / "pb23", PB24: self.root / "pb24"}
        self.models = {ANCHOR: self.root / "model23.json", PB24: self.root / "model24.json"}
        for tag, directory in self.directories.items():
            directory.mkdir()
            _write_root(directory / "DATA_with_score.root")
            threshold_dir = directory / "cut_scan/weighted_signal_efficiency"
            threshold_dir.mkdir(parents=True)
            rows = [{
                "target_efficiency": target,
                "score_threshold": 0.99 - index * 0.01,
                "achieved_efficiency": target - 1e-5,
                "data_entries": 100 + index,
            } for index, target in enumerate([0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40])]
            (threshold_dir / "thresholds.json").write_text(json.dumps({
                "train_tag": tag,
                "efficiency_label": "weighted signal efficiency",
                "weight_branch": "Reweight",
                "thresholds": rows,
            }))
        base_model = {
            "input_columns": ["Btrk1dR", "Btktkpt", "Btrk1Pt", "Bchi2Prob", "Bcos_dtheta", "Btrk2Pt"],
            "reweight_profile": "rwr6range5v1",
            "model_params": {"n_estimators": 1000, "max_depth": 4, "scale_pos_weight": 2.0},
        }
        self.models[ANCHOR].write_text(json.dumps(base_model))
        base_model["scale_pos_weight"] = 5.0
        base_model["model_params"]["scale_pos_weight"] = 5.0
        self.models[PB24].write_text(json.dumps(base_model))

    def tearDown(self):
        self.temporary.cleanup()

    def _patches(self):
        return (
            patch.object(exporter, "selected_dir", side_effect=lambda tag: str(self.directories[tag])),
            patch.object(exporter, "resolve_model_config_path", side_effect=lambda tag: str(self.models[tag])),
        )

    def test_manifest_uses_pb23_anchor_and_two_categories(self):
        selected_patch, model_patch = self._patches()
        with selected_patch, model_patch:
            manifest = exporter.build_manifest(ANCHOR, self.directories[ANCHOR])
        schema = json.loads(exporter.SCHEMA_PATH.read_text())
        self.assertEqual(manifest["schema_version"], schema["properties"]["schema_version"]["const"])
        self.assertEqual(manifest["contract"], exporter.CONTRACT)
        self.assertEqual(manifest["anchor_train_tag"], ANCHOR)
        self.assertEqual(manifest["pairing"]["categories"]["pb24"]["train_tag"], PB24)
        self.assertEqual(len(manifest["working_points"]), 7)
        self.assertEqual(manifest["nominal_fit_contract"]["shared_parameters"], ["signal_mean"])
        self.assertEqual(
            manifest["nominal_fit_contract"]["signal"]["yields"],
            "independent_nonnegative_by_category",
        )
        self.assertIn("background-only toys", manifest["nominal_fit_contract"]["significance"]["calibration"])

    def test_export_is_atomic_and_versioned(self):
        selected_patch, model_patch = self._patches()
        with selected_patch, model_patch:
            output = exporter.export_manifest(ANCHOR, self.directories[ANCHOR])
        self.assertEqual(output.name, exporter.MANIFEST_FILENAME)
        self.assertEqual(json.loads(output.read_text())["contract"], exporter.CONTRACT)
        self.assertFalse(output.with_name(f".{output.name}.tmp").exists())

    def test_rejects_model_configuration_mismatch(self):
        model = json.loads(self.models[PB24].read_text())
        model["input_columns"] = ["different"]
        self.models[PB24].write_text(json.dumps(model))
        selected_patch, model_patch = self._patches()
        with selected_patch, model_patch, self.assertRaisesRegex(ValueError, "input_columns"):
            exporter.build_manifest(ANCHOR, self.directories[ANCHOR])


if __name__ == "__main__":
    unittest.main()
