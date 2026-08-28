import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import uproot

from workflows.integration import export_x_simultaneous_year_mc_shape_manifest as exporter


ANCHOR = "X_pb23_v3_fid3_6v5_rwr6range5v1_xgb_v1"
PB24 = "X_pb24_v19_fid19_6v5_rwr6range5v1_xgb_v1"
TARGETS = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]


def _write_inputs(directory: Path, tree: str, include_weight: bool = True) -> None:
    with uproot.recreate(directory / "DATA_with_score.root") as root_file:
        root_file.mktree(
            "ntmix",
            {
                "Bmass": np.array([3.84, 3.872, 3.91], dtype=np.float32),
                "Prediction": np.array([0.8, 0.95, 0.99], dtype=np.float32),
            },
        )
    branches = {
        "Bmass": np.array([3.868, 3.872, 3.876], dtype=np.float32),
        "Prediction": np.array([0.85, 0.95, 0.99], dtype=np.float32),
    }
    if include_weight:
        branches["Reweight"] = np.array([0.8, 1.0, 1.2], dtype=np.float32)
    with uproot.recreate(directory / "MC_with_score.root") as root_file:
        root_file.mktree(tree, branches)


class ExportXSimultaneousMCShapeManifestTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.directories = {ANCHOR: self.root / "pb23", PB24: self.root / "pb24"}
        self.models = {ANCHOR: self.root / "model23.json", PB24: self.root / "model24.json"}
        for tag, directory in self.directories.items():
            directory.mkdir()
            _write_inputs(directory, "ntmixX3872" if tag == ANCHOR else "ntmix_X3872")
            threshold_dir = directory / "cut_scan/weighted_signal_efficiency"
            threshold_dir.mkdir(parents=True)
            rows = [
                {
                    "target_efficiency": target,
                    "score_threshold": 0.99 - index * 0.01,
                    "achieved_efficiency": target - 1e-5,
                    "data_entries": 100 + index,
                }
                for index, target in enumerate(TARGETS)
            ]
            (threshold_dir / "thresholds.json").write_text(
                json.dumps(
                    {
                        "train_tag": tag,
                        "efficiency_label": "weighted signal efficiency",
                        "weight_branch": "Reweight",
                        "thresholds": rows,
                    }
                ),
                encoding="utf-8",
            )
        base_model = {
            "input_columns": ["a", "b"],
            "reweight_profile": "rwr6range5v1",
            "model_params": {"max_depth": 4, "scale_pos_weight": 2.0},
        }
        self.models[ANCHOR].write_text(json.dumps(base_model), encoding="utf-8")
        pb24 = copy.deepcopy(base_model)
        pb24["model_params"]["scale_pos_weight"] = 5.0
        self.models[PB24].write_text(json.dumps(pb24), encoding="utf-8")

    def tearDown(self):
        self.temporary.cleanup()

    def _patches(self):
        return (
            patch.object(exporter, "selected_dir", side_effect=lambda tag: str(self.directories[tag])),
            patch.object(
                exporter,
                "resolve_model_config_path",
                side_effect=lambda tag: str(self.models[tag]),
            ),
        )

    def test_contract_contains_year_specific_mc_shape_inputs(self):
        selected_patch, model_patch = self._patches()
        with selected_patch, model_patch:
            manifest = exporter.build_manifest(ANCHOR, self.directories[ANCHOR])
        self.assertEqual(manifest["schema_version"], 2)
        self.assertEqual(manifest["contract"], exporter.CONTRACT)
        categories = manifest["pairing"]["categories"]
        self.assertEqual(categories["pb23"]["signal_mc"]["tree"], "ntmixX3872")
        self.assertEqual(categories["pb24"]["signal_mc"]["tree"], "ntmix_X3872")
        fit = manifest["nominal_fit_contract"]
        self.assertEqual(fit["shared_parameters"], ["data_signal_mean"])
        self.assertEqual(fit["signal_mc"]["model"], "common_mean_double_gaussian")
        self.assertEqual(fit["data_fit"]["mean_gev"]["range"], [3.86669, 3.87669])
        self.assertEqual(fit["data_fit"]["category_width_scale"]["range"], [0.9, 1.5])
        self.assertEqual(len(manifest["working_points"]), 7)

    def test_export_writes_versioned_manifest_and_hash(self):
        selected_patch, model_patch = self._patches()
        with selected_patch, model_patch:
            output = exporter.export_manifest(ANCHOR, self.directories[ANCHOR])
        self.assertEqual(output.name, exporter.MANIFEST_FILENAME)
        self.assertTrue(output.with_suffix(output.suffix + ".sha256").is_file())

    def test_rejects_missing_mc_weight(self):
        _write_inputs(self.directories[PB24], "ntmix_X3872", include_weight=False)
        selected_patch, model_patch = self._patches()
        with selected_patch, model_patch, self.assertRaisesRegex(ValueError, "Reweight"):
            exporter.build_manifest(ANCHOR, self.directories[ANCHOR])


if __name__ == "__main__":
    unittest.main()
