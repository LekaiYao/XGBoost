VARSET_COLUMNS = {
    "4v": ["Btrk1dR", "Btrk2dR", "BtrkPtimb", "Bchi2Prob"],
    "4v2": ["Bchi2Prob", "Btrk1dR", "BtrkPtimb", "Btrk2Pt"],
    "5v": ["Bchi2Prob", "Btrk1dR", "Btrk2dR", "BtrkPtimb", "Btrk2Pt"],
    "6v": ["Bchi2Prob", "Btrk1dR", "BtrkPtimb", "Btrk2Pt", "BtktkvProb", "Bcos_dtheta"],
    "7v": ["Bchi2Prob", "Btrk1dR", "Btrk2dR", "BtrkPtimb", "Btrk2Pt", "Balpha", "Bnorm_trk1Dxy"],
    "7v2": ["Bchi2Prob", "Btrk1dR", "Btrk2dR", "BtrkPtimb", "Btrk2Pt", "BtktkvProb", "Bcos_dtheta"],
}

SUPPORTED_VARSETS = tuple(VARSET_COLUMNS.keys())


def infer_varset_from_tag(tag):
    # Match longer keys first (e.g. 7v2 before 7v).
    for key in sorted(SUPPORTED_VARSETS, key=len, reverse=True):
        if f"_{key}_" in tag:
            return key
    return None
