import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from workflows.prepare_bd_pbpb_precut_inputs import _run_phase_with_retry


class PrepareBdPbpbPrecutInputsTest(unittest.TestCase):
    def test_run_phase_creates_output_parent_before_root(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            output_path = root / "nested" / "input" / "precut.root"
            with patch(
                "workflows.prepare_bd_pbpb_precut_inputs._run_root_copytree",
                return_value=SimpleNamespace(
                    returncode=12,
                    stdout="",
                    stderr="ROOT_PRECUT_ERROR open_output",
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "returncode=12"):
                    _run_phase_with_retry(
                        source_spec=f"{root / 'source.root'}:events",
                        selection_expr="x > 0",
                        output_path=output_path,
                        input_dir=root,
                        max_retries=1,
                        retry_delay=0.0,
                        cache_size_mb=0,
                        phase_name="test",
                        macro_path=root / "macro.C",
                        force=False,
                    )
            self.assertTrue(output_path.parent.is_dir())


if __name__ == "__main__":
    unittest.main()
