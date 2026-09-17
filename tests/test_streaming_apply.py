import tempfile
import unittest
from pathlib import Path

import numpy as np
import uproot

from utils.streaming_apply import write_scored_root


class IdentityScaler:
    def transform(self, frame):
        return frame.to_numpy(dtype=float)


class SumModel:
    def predict_proba(self, frame):
        score = frame.to_numpy(dtype=float).sum(axis=1) / 10.0
        return np.column_stack([1.0 - score, score])


class StreamingApplyTest(unittest.TestCase):
    def test_writes_all_input_branches_and_score_as_ttree(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            input_path = Path(temporary_directory) / "input.root"
            output_path = Path(temporary_directory) / "output.root"
            with uproot.recreate(input_path) as root_file:
                root_file.mktree(
                    "events",
                    {
                        "x": np.array([1.0, 2.0, 3.0, 4.0]),
                        "keep": np.array([10, 11, 12, 13], dtype=np.int32),
                    },
                )
            bundle = {
                "scaler": IdentityScaler(),
                "model": SumModel(),
                "input_columns": ["x"],
                "trans_columns": ["x_trans"],
                "score_column": "Prediction",
            }
            result = write_scored_root(
                f"{input_path}:events",
                output_path,
                "events",
                [bundle],
                step_size=2,
            )
            self.assertEqual(result, {"entries": 4, "chunks": 2})
            with uproot.open(output_path) as root_file:
                tree = root_file["events"]
                self.assertEqual(tree.classname, "TTree")
                self.assertEqual(tree.num_entries, 4)
                np.testing.assert_array_equal(
                    tree["keep"].array(library="np"),
                    [10, 11, 12, 13],
                )
                np.testing.assert_allclose(
                    tree["Prediction"].array(library="np"),
                    [0.1, 0.2, 0.3, 0.4],
                )


    def test_fiducial_filter_skips_empty_chunks_preserves_weights_and_mass(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "input.root"
            target = Path(directory) / "output.root"
            with uproot.recreate(source) as f:
                f.mktree("events", {
                    "x": np.array([0., 0.25, 0.1, 0.2, 0.3]),
                    "Bmass": np.array([3.7, 3.8, 3.872, 3.83, 3.9]),
                    "pThatreweight": np.array([1., 2., 3., 4., 5.]),
                })
            bundle = {"scaler": IdentityScaler(), "model": SumModel(),
                      "input_columns": ["x"], "trans_columns": ["x_trans"],
                      "score_column": "Prediction"}
            result = write_scored_root(f"{source}:events", target, "events", [bundle],
                                       step_size=2, selection="x > 0 and x < 0.25")
            self.assertEqual(result, {"entries": 2, "chunks": 1})
            with uproot.open(target) as f:
                arrays = f["events"].arrays(library="np")
                np.testing.assert_array_equal(arrays["pThatreweight"], [3., 4.])
                np.testing.assert_allclose(arrays["Bmass"], [3.872, 3.83])
                np.testing.assert_allclose(arrays["Prediction"], [0.01, 0.02])
            with self.assertRaisesRegex(ValueError, "no entries after selection"):
                write_scored_root(f"{source}:events", target, "events", [bundle],
                                  step_size=2, selection="x > 1")
            self.assertTrue(target.exists())
            self.assertFalse(target.with_suffix(".root.tmp").exists())


if __name__ == "__main__":
    unittest.main()
