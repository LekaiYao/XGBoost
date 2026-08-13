import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from utils.run_metadata import save_run_metadata


class RunMetadataTest(unittest.TestCase):
    def test_explicit_split_fractions_are_written(self):
        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "run_metadata.json"
            with (
                patch("utils.run_metadata.metadata_path", return_value=str(output_path)),
                patch("utils.run_metadata.build_artifacts", return_value={}),
            ):
                save_run_metadata(
                    train_tag="X_pb24_v7_fid8_8v2_xgb_v1",
                    training_script="test",
                    signal_path="signal.root:tree",
                    background_path="background.root:tree",
                    signal_selection="",
                    background_selection="",
                    input_columns=["x"],
                    trans_columns=["x_trans"],
                    pos_weight=1.0,
                    fixed_model_params={},
                    best_model_params={},
                    train_fraction=0.75,
                    val_fraction=0.0,
                    test_fraction=0.25,
                )

            with output_path.open() as stream:
                metadata = json.load(stream)
            self.assertEqual(metadata["train_fraction"], 0.75)
            self.assertEqual(metadata["val_fraction"], 0.0)
            self.assertEqual(metadata["test_fraction"], 0.25)


if __name__ == "__main__":
    unittest.main()
