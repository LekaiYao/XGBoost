import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import uproot

from workflows.integration.export_psi2s_fit_scan_manifest import (
    CONTRACT,
    MANIFEST_FILENAME,
    SCHEMA_PATH,
    TARGET_EFFICIENCIES,
    build_fit_scan_manifest,
    export_fit_scan_manifest,
)


TRAIN_TAG = "Psi2S_pb24_v1_fid1_6v1_rwr6range4v1_xgb_v1"


def _write_inputs(directory: Path, include_weight: bool = True) -> None:
    with uproot.recreate(directory / "DATA_with_score.root") as root_file:
        root_file.mktree(
            "ntmix",
            {
                "Bmass": np.array([3.62, 3.686, 3.77], dtype=np.float32),
                "Prediction": np.array([0.80, 0.95, 0.99], dtype=np.float32),
            },
        )
    mc_branches = {
        "Bmass": np.array([3.682, 3.686, 3.690], dtype=np.float32),
        "Prediction": np.array([0.85, 0.95, 0.99], dtype=np.float32),
    }
    if include_weight:
        mc_branches["Reweight"] = np.array([0.8, 1.0, 1.2], dtype=np.float32)
    with uproot.recreate(directory / "MC_with_score.root") as root_file:
        root_file.mktree("ntmix_PSI2S", mc_branches)


class ExportPsi2SFitScanManifestTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.selected = Path(self.temporary.name)
        _write_inputs(self.selected)
        rows = [
            {
                "target_efficiency": target,
                "score_threshold": 0.99 - 0.01 * index,
                "achieved_efficiency": target - 1e-5,
                "data_entries": 100 + 10 * index,
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

    def test_manifest_matches_versioned_contract(self):
        manifest = build_fit_scan_manifest(TRAIN_TAG, self.selected)
        schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))

        self.assertEqual(manifest["schema_version"], 1)
        self.assertEqual(manifest["contract"], CONTRACT)
        self.assertEqual(manifest["channel"], "Psi2S")
        self.assertTrue(set(schema["required"]).issubset(manifest))
        self.assertEqual(len(manifest["working_points"]), 7)
        self.assertEqual(manifest["working_points"][0]["key"], "psi2seff10")
        self.assertEqual(manifest["working_points"][-1]["key"], "psi2seff40")
        self.assertEqual(
            manifest["working_points"][0]["fiducial_score_selected_data_entries"],
            100,
        )
        self.assertEqual(manifest["inputs"]["data"]["event_weight"], "unit")
        self.assertEqual(
            manifest["inputs"]["signal_mc"]["event_weight_branch"], "Reweight"
        )
        self.assertEqual(
            manifest["nominal_fit_contract"]["mass_range_gev"], [3.6, 3.8]
        )
        self.assertEqual(
            manifest["nominal_fit_contract"]["background"]["order"], 2
        )
        self.assertFalse(
            manifest["nominal_fit_contract"]["background"][
                "additional_stability_models_required"
            ]
        )

    def test_export_uses_distinct_versioned_filename(self):
        legacy_x = self.selected / "fit_scan_manifest.data_only_nominal_v2.json"
        legacy_x.write_text('{"channel": "X"}\n', encoding="utf-8")
        output = export_fit_scan_manifest(TRAIN_TAG, self.selected)
        self.assertEqual(output.name, MANIFEST_FILENAME)
        self.assertEqual(json.loads(legacy_x.read_text()), {"channel": "X"})
        self.assertEqual(json.loads(output.read_text())["contract"], CONTRACT)

    def test_rejects_missing_mc_weight_branch(self):
        _write_inputs(self.selected, include_weight=False)
        with self.assertRaisesRegex(ValueError, "Reweight"):
            build_fit_scan_manifest(TRAIN_TAG, self.selected)

    def test_rejects_nonmonotonic_data_entries(self):
        threshold_path = self.selected / "cut_scan/weighted_signal_efficiency/thresholds.json"
        payload = json.loads(threshold_path.read_text())
        payload["thresholds"][3]["data_entries"] = 1
        threshold_path.write_text(json.dumps(payload))
        with self.assertRaisesRegex(ValueError, "DATA entries"):
            build_fit_scan_manifest(TRAIN_TAG, self.selected)

    def test_rejects_x_tag(self):
        with self.assertRaisesRegex(ValueError, "Psi2S"):
            build_fit_scan_manifest(
                "X_pb24_v19_fid19_6v5_rwr6range5v1_xgb_v1", self.selected
            )


if __name__ == "__main__":
    unittest.main()
