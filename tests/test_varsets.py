import unittest

from utils.varsets import get_varset_columns, infer_varset_from_tag


class VarsetsTest(unittest.TestCase):
    def test_pp_x_5v2_resolves_from_full_tag(self):
        tag = "X_pp24_v4_fid3_5v2_rw0_xgb_v1"
        self.assertEqual(infer_varset_from_tag(tag), "5v2")
        self.assertEqual(
            get_varset_columns("pp", "5v2", "X"),
            ["Btrk1dR", "BtrkPtimb", "Btrk1Pt", "Btrk2Pt", "BtktkvProb"],
        )


if __name__ == "__main__":
    unittest.main()
