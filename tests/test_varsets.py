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

    def test_psi2s_6v1_resolves_from_full_tag(self):
        tag = "Psi2S_pb24_v1_fid1_6v1_rwr6range4v1_xgb_v1"
        expected = [
            "Btrk1dR", "Btktkpt", "Btrk1Pt",
            "Bchi2Prob", "Bcos_dtheta", "Btrk2Pt",
        ]
        self.assertEqual(infer_varset_from_tag(tag), "6v1")
        self.assertEqual(get_varset_columns("pbpb", "6v1", "Psi2S"), expected)
        self.assertEqual(get_varset_columns("pp", "6v1", "Psi2S"), expected)


if __name__ == "__main__":
    unittest.main()
