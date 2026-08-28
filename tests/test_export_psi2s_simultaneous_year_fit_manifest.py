import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import uproot

from workflows.integration import (
    export_psi2s_simultaneous_year_fit_manifest as exporter,
)


ANCHOR = "Psi2S_pb23_v1_fid1_6v1_rwr6range4v1_xgb_v1"
PB24 = "Psi2S_pb24_v1_fid1_6v1_rwr6range4v1_xgb_v1"
TARGETS = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]


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


class ExportPsi2SSimultaneousYearFitManifestTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.directories = {ANCHOR: self.root / "pb23", PB24: self.root / "pb24"}
        self.models = {ANCHOR: self.root / "model23.json", PB24: self.root / "model24.json"}
        for tag, directory in self.directories.items():
            directory.mkdir()
            _write_inputs(directory)
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
            "input_columns": [
                "Btrk1dR",
                "Btktkpt",
                "Btrk1Pt",
                "Bchi2Prob",
                "Bcos_dtheta",
                "Btrk2Pt",
            ],
            "reweight_profile": "rwr6range4v1",
            "model_params": {
                "n_estimators": 1000,
                "max_depth": 4,
                "scale_pos_weight": 2.0,
            },
        }
        self.models[ANCHOR].write_text(json.dumps(base_model), encoding="utf-8")
        pb24_model = copy.deepcopy(base_model)
        pb24_model["model_params"]["scale_pos_weight"] = 5.0
        self.models[PB24].write_text(json.dumps(pb24_model), encoding="utf-8")

    def tearDown(self):
        self.temporary.cleanup()

    def _patches(self):
        return (
            patch.object(
                exporter,
                "selected_dir",
                side_effect=lambda tag: str(self.directories[tag]),
            ),
            patch.object(
                exporter,
                "resolve_model_config_path",
                side_effect=lambda tag: str(self.models[tag]),
            ),
        )

    def test_manifest_carries_data_mc_and_two_year_categories(self):
        selected_patch, model_patch = self._patches()
        with selected_patch, model_patch:
            manifest = exporter.build_manifest(ANCHOR, self.directories[ANCHOR])
        schema = json.loads(exporter.SCHEMA_PATH.read_text(encoding="utf-8"))

        self.assertEqual(
            manifest["schema_version"],
            schema["properties"]["schema_version"]["const"],
        )
        self.assertEqual(manifest["contract"], exporter.CONTRACT)
        self.assertEqual(manifest["pairing"]["categories"]["pb24"]["train_tag"], PB24)
        self.assertEqual(
            manifest["pairing"]["categories"]["pb23"]["signal_mc"]["tree"],
            "ntmix_PSI2S",
        )
        self.assertEqual(len(manifest["working_points"]), 7)
        self.assertEqual(manifest["working_points"][0]["key"], "psi2seff10")
        contract = manifest["nominal_fit_contract"]
        self.assertEqual(contract["shared_parameters"], ["signal_mean"])
        self.assertEqual(contract["signal"]["model"], "double_gaussian_mc_shape")
        self.assertTrue(
            contract["signal"]["mc_fit"]["performed_independently_by_category"]
        )
        self.assertEqual(
            contract["signal"]["data_fit"]["fixed_from_category_mc"],
            ["sigma1", "sigma2", "fraction"],
        )

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
        with selected_patch, model_patch, self.assertRaisesRegex(
            ValueError, "input_columns"
        ):
            exporter.build_manifest(ANCHOR, self.directories[ANCHOR])

    def test_rejects_missing_mc_weight_branch(self):
        _write_inputs(self.directories[PB24], include_weight=False)
        selected_patch, model_patch = self._patches()
        with selected_patch, model_patch, self.assertRaisesRegex(ValueError, "Reweight"):
            exporter.build_manifest(ANCHOR, self.directories[ANCHOR])


if __name__ == "__main__":
    unittest.main()
