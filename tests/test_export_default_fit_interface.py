import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from workflows.integration import export_default_fit_interface as dispatcher


class ExportDefaultFitInterfaceTest(unittest.TestCase):
    def test_unconfigured_tag_is_a_successful_skip(self):
        with patch.object(dispatcher, "resolve_year_pairing_for_tag", return_value=None):
            self.assertIsNone(dispatcher.export_default_fit_interface("X_pb24_unpaired"))

    def test_missing_current_artifact_fails(self):
        pairing = {
            "anchor_train_tag": "anchor",
            "tags": {"pb23": "anchor", "pb24": "peer"},
        }
        missing = Path("/definitely/missing/current.root")
        with patch.object(dispatcher, "resolve_year_pairing_for_tag", return_value=pairing), \
             patch.object(dispatcher, "_required_paths", return_value=(missing,)):
            with self.assertRaisesRegex(FileNotFoundError, "Current tag"):
                dispatcher.export_default_fit_interface("anchor")

    def test_missing_peer_is_a_successful_skip(self):
        pairing = {
            "anchor_train_tag": "anchor",
            "tags": {"pb23": "anchor", "pb24": "peer"},
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            ready = Path(tmpdir) / "ready"
            ready.touch()
            missing = Path(tmpdir) / "missing"
            with patch.object(dispatcher, "resolve_year_pairing_for_tag", return_value=pairing), \
                 patch.object(
                     dispatcher,
                     "_required_paths",
                     side_effect=lambda tag: (ready,) if tag == "anchor" else (missing,),
                 ):
                self.assertIsNone(dispatcher.export_default_fit_interface("anchor"))

    def test_ready_x_pair_exports_from_anchor(self):
        pairing = {
            "anchor_train_tag": "anchor",
            "tags": {"pb23": "anchor", "pb24": "peer"},
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            ready = root / "ready"
            ready.touch()
            expected = root / "manifest.json"
            with patch.object(dispatcher, "resolve_year_pairing_for_tag", return_value=pairing), \
                 patch.object(dispatcher, "_required_paths", return_value=(ready,)), \
                 patch.object(dispatcher, "selected_dir", return_value=str(root)), \
                 patch.object(dispatcher, "infer_channel_from_tag", return_value="X"), \
                 patch.object(dispatcher, "export_x_manifest", return_value=expected) as export:
                self.assertEqual(
                    dispatcher.export_default_fit_interface("peer"), expected
                )
            export.assert_called_once_with("anchor")


if __name__ == "__main__":
    unittest.main()
