import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import uproot

from workflows.integration.export_analysis_manifest import (
    _mass_windows_from_selection,
    build_manifest,
    export_manifest,
)


TRAIN_TAG = "Bu_pp24_v1_fid1_10v1_xgb_v1"


def _write_root(path: Path, tree: str, include_prediction: bool = True):
    branches = {"Bmass": np.array([5.1, 5.28, 5.5], dtype=np.float32)}
    if include_prediction:
        branches["Prediction"] = np.array([0.1, 0.6, 0.9], dtype=np.float32)
    with uproot.recreate(path) as root_file:
        root_file.mktree(tree, branches)


class ExportAnalysisManifestTest(unittest.TestCase):
    def test_extracts_chained_sidebands(self):
        windows = _mass_windows_from_selection(
            "((5.0 < Bmass < 5.2) or (5.36 < Bmass < 5.56)) and Bpt > 7.5"
        )
        self.assertEqual(windows, [(5.0, 5.2), (5.36, 5.56)])

    def test_builds_and_writes_manifest(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            selected = Path(temporary_directory)
            _write_root(selected / "DATA_with_score.root", "ntKp")
            _write_root(selected / "MC_with_score.root", "ntKp")

            manifest = build_manifest(TRAIN_TAG, selected)
            self.assertEqual(manifest["schema_version"], 1)
            self.assertEqual(manifest["channel"], "Bu")
            self.assertEqual(manifest["dataset"], "pp24")
            self.assertEqual(manifest["collision_system"], "pp")
            self.assertEqual(manifest["score_branch"], "Prediction")
            self.assertEqual(manifest["artifacts"]["data"]["path"], "DATA_with_score.root")
            self.assertEqual(manifest["artifacts"]["data"]["tree"], "ntKp")
            self.assertEqual(manifest["sideband_windows"][0], {"low": 5.0, "high": 5.2})
            self.assertNotIn("punzi", json.dumps(manifest).lower())
            self.assertNotIn("Analysis_CODES", json.dumps(manifest))

            manifest_path = export_manifest(TRAIN_TAG, selected)
            self.assertEqual(manifest_path, selected / "analysis_manifest.json")
            self.assertEqual(json.loads(manifest_path.read_text()), manifest)

    def test_rejects_missing_prediction_branch(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            selected = Path(temporary_directory)
            _write_root(selected / "DATA_with_score.root", "ntKp", include_prediction=False)
            _write_root(selected / "MC_with_score.root", "ntKp")
            with self.assertRaisesRegex(ValueError, "Missing score branch 'Prediction'"):
                build_manifest(TRAIN_TAG, selected)

    def test_rejects_missing_tree(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            selected = Path(temporary_directory)
            _write_root(selected / "DATA_with_score.root", "wrong_tree")
            _write_root(selected / "MC_with_score.root", "ntKp")
            with self.assertRaisesRegex(ValueError, "Missing TTree 'ntKp'"):
                build_manifest(TRAIN_TAG, selected)


if __name__ == "__main__":
    unittest.main()
