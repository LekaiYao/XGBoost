import tempfile
import unittest
from pathlib import Path

import numpy as np
import uproot

from workflows.integration.build_psi2s_expanded_closure_artifacts import (
    ADDED_BRANCHES,
    CURRENT_BRANCHES,
    EXPANDED_BRANCHES,
    RAW_IDENTITY_BRANCHES,
    TREE_NAME,
    build_expanded_mc,
)


def _values(names, entries=5):
    return {
        name: (np.arange(entries, dtype=np.float32) + index / 100.0)
        for index, name in enumerate(names)
    }


class BuildPsi2SExpandedClosureArtifactsTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.raw = self.root / "raw.root"
        self.current = self.root / "current.root"
        self.output = self.root / "closure25.root"
        raw = _values(RAW_IDENTITY_BRANCHES + ADDED_BRANCHES)
        current = {name: raw[name] for name in RAW_IDENTITY_BRANCHES}
        current["Reweight"] = np.linspace(0.5, 1.5, 5, dtype=np.float32)
        current["Prediction"] = np.linspace(0.1, 0.9, 5, dtype=np.float32)
        with uproot.recreate(self.raw) as root_file:
            root_file.mktree(TREE_NAME, raw)
        with uproot.recreate(self.current) as root_file:
            root_file.mktree(TREE_NAME, current)

    def tearDown(self):
        self.temporary.cleanup()

    def test_build_preserves_current_and_adds_raw_branches(self):
        metadata = build_expanded_mc(self.raw, self.current, self.output)
        self.assertEqual(metadata["entries"], 5)
        self.assertEqual(len(metadata["branches"]), 25)
        self.assertFalse(
            metadata["provenance"]["identity_validation"]["prediction_recomputed"]
        )
        with uproot.open(self.output) as root_file:
            tree = root_file[TREE_NAME]
            self.assertEqual(tuple(tree.keys()), EXPANDED_BRANCHES)
            self.assertEqual(tree.num_entries, 5)

    def test_rejects_entry_order_mismatch(self):
        raw = _values(RAW_IDENTITY_BRANCHES + ADDED_BRANCHES)
        raw[RAW_IDENTITY_BRANCHES[0]] = raw[RAW_IDENTITY_BRANCHES[0]][::-1]
        with uproot.recreate(self.raw) as root_file:
            root_file.mktree(TREE_NAME, raw)
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            build_expanded_mc(self.raw, self.current, self.output)

    def test_rejects_overwriting_input(self):
        with self.assertRaisesRegex(ValueError, "must not overwrite"):
            build_expanded_mc(self.raw, self.current, self.current)


if __name__ == "__main__":
    unittest.main()
