import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import uproot

from workflows.integration.export_x_fit_scan_manifest import (
    CONTRACT,
    MANIFEST_FILENAME,
    SCHEMA_PATH,
    TARGET_EFFICIENCIES,
    build_fit_scan_manifest,
    export_fit_scan_manifest,
)


TRAIN_TAG = "X_pb24_v18_fid19_6v5_rwr6range7dr025v1_xgb_v1"


def _write_data(path: Path, include_score: bool = True) -> None:
    branches = {
        "Bmass": np.array([3.84, 3.872, 3.91], dtype=np.float32),
    }
    if include_score:
        branches["Prediction"] = np.array([0.90, 0.95, 0.99], dtype=np.float32)
    with uproot.recreate(path) as root_file:
        root_file.mktree("ntmix", branches)


class ExportXFitScanManifestTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.selected = Path(self.temporary.name)
        _write_data(self.selected / "DATA_with_score.root")
        rows = [
            {
                "target_efficiency": target,
                "score_threshold": 0.99 - 0.01 * index,
                "achieved_efficiency": target - 1e-5,
            }
            for index, target in enumerate(TARGET_EFFICIENCIES)
        ]
        threshold_dir = self.selected / "cut_scan/weighted_signal_efficiency"
        threshold_dir.mkdir(parents=True)
        (threshold_dir / "thresholds.json").write_text(
            json.dumps(
                {
                    "train_tag": TRAIN_TAG,
                    "efficiency_label": "weighted signal efficiency",
                    "weight_branch": "Reweight",
                    "thresholds": rows,
                }
            ),
            encoding="utf-8",
        )

    def tearDown(self):
        self.temporary.cleanup()

    def test_manifest_matches_versioned_schema_and_nominal_contract(self):
        manifest = build_fit_scan_manifest(TRAIN_TAG, self.selected)
        schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
        self.assertEqual(schema["$schema"], "https://json-schema.org/draft/2020-12/schema")
        self.assertTrue(set(schema["required"]).issubset(manifest))
        self.assertEqual(
            manifest["schema_version"],
            schema["properties"]["schema_version"]["const"],
        )
        self.assertEqual(
            manifest["contract"], schema["properties"]["contract"]["const"]
        )

        self.assertEqual(manifest["schema_version"], 2)
        self.assertEqual(manifest["contract"], CONTRACT)
        self.assertEqual(len(manifest["working_points"]), 7)
        self.assertEqual(manifest["score"]["comparison_operator"], ">")
        self.assertFalse(manifest["score"]["equality_passes"])
        self.assertEqual(
            manifest["nominal_fit_contract"]["signal"]["model"],
            "single_gaussian",
        )
        self.assertFalse(
            manifest["nominal_fit_contract"]["signal"]["mc_shape_constraint"]
        )
        self.assertEqual(
            manifest["nominal_fit_contract"]["signal"]["sigma_gev"]["range"],
            [0.002, 0.008],
        )
        self.assertNotIn("signal_mc", manifest["inputs"])

    def test_versioned_export_does_not_overwrite_legacy_manifest(self):
        legacy = self.selected / "fit_scan_manifest.json"
        version_one = self.selected / "fit_scan_manifest.data_only_nominal_v1.json"
        legacy.write_text('{"legacy": true}\n', encoding="utf-8")
        version_one.write_text('{"schema_version": 1}\n', encoding="utf-8")
        output = export_fit_scan_manifest(TRAIN_TAG, self.selected)
        self.assertEqual(output.name, MANIFEST_FILENAME)
        self.assertEqual(json.loads(legacy.read_text()), {"legacy": True})
        self.assertEqual(json.loads(version_one.read_text()), {"schema_version": 1})
        self.assertEqual(json.loads(output.read_text())["contract"], CONTRACT)

    def test_rejects_non_reweighted_efficiency_thresholds(self):
        threshold_path = self.selected / "cut_scan/weighted_signal_efficiency/thresholds.json"
        payload = json.loads(threshold_path.read_text())
        payload["weight_branch"] = None
        threshold_path.write_text(json.dumps(payload))
        with self.assertRaisesRegex(ValueError, "threshold weight branch"):
            build_fit_scan_manifest(TRAIN_TAG, self.selected)

    def test_rejects_missing_scored_data_field(self):
        _write_data(self.selected / "DATA_with_score.root", include_score=False)
        with self.assertRaisesRegex(ValueError, "Prediction"):
            build_fit_scan_manifest(TRAIN_TAG, self.selected)


if __name__ == "__main__":
    unittest.main()
