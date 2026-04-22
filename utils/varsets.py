VARSET_COLUMNS = {
    "4v": ["Btrk1dR", "Btrk2dR", "BtrkPtimb", "Bchi2Prob"],
    "4v2": ["Bchi2Prob", "Btrk1dR", "BtrkPtimb", "Btrk2Pt"],
    "5v": ["Bchi2Prob", "Btrk1dR", "Btrk2dR", "BtrkPtimb", "Btrk2Pt"],
    "5v2": ["Bchi2Prob", "Btrk1dR", "Btrk2dR", "Btrk1Pt", "Btrk2Pt"],
    "5v3": ["Btrk1dR", "Btrk2dR", "BtrkPtimb", "Btrk2Pt", "Bcos_dtheta"],
    "6v": ["Bchi2Prob", "Btrk1dR", "BtrkPtimb", "Btrk2Pt", "BtktkvProb", "Bcos_dtheta"],
    "6v2": ["Bchi2Prob", "Btrk1dR", "Btrk2dR", "BtrkPtimb", "BtktkvProb", "Bcos_dtheta"],
    "6v3": ["Bchi2Prob", "Btrk1dR", "Btrk2dR", "BtrkPtimb", "Btrk2Pt", "Bcos_dtheta"],
    "7v2": ["Bchi2Prob", "Btrk1dR", "Btrk2dR", "BtrkPtimb", "Btrk2Pt", "BtktkvProb", "Bcos_dtheta"],
    "7v3": ["Bchi2Prob", "Btrk1dR", "BtrkPtimb", "Btrk1Pt", "Btrk2Pt", "BtktkvProb", "Bcos_dtheta"],
    "7v4": ["Bpt", "Bchi2Prob", "Btrk1dR", "Btrk2dR", "BtrkPtimb", "Btrk2Pt", "Bcos_dtheta"],
    "8v": ["Bpt", "Bchi2Prob", "Btrk1dR", "BtrkPtimb", "Btrk1Pt", "Btrk2Pt", "BtktkvProb", "Bcos_dtheta"],
    "8v2": [
        "Bchi2Prob",
        "Btrk1dR",
        "Btrk2dR",
        "BtrkPtimb",
        "Btrk2Pt",
        "BtktkvProb",
        "Bcos_dtheta",
        "Btrk1Pt",
    ],
    "8v3": [
        "Bchi2Prob",
        "Btrk1dR",
        "Btrk2dR",
        "BtrkPtimb",
        "Btrk2Pt",
        "BtktkvProb",
        "Bcos_dtheta",
        "Btktkpt",
    ],
    "8v4": ["Bpt", "Bchi2Prob", "Btrk1dR", "Btrk2dR", "BtrkPtimb", "Btrk2Pt", "BtktkvProb", "Bcos_dtheta"],
    "9v": [
        "Bchi2Prob",
        "Btrk1dR",
        "Btrk2dR",
        "BtrkPtimb",
        "Btktkpt",
        "Bcos_dtheta",
        "Btrk1Pt",
        "Btrk2Pt",
        "BtktkvProb",
    ],
}

SUPPORTED_VARSETS = tuple(VARSET_COLUMNS.keys())


def infer_varset_from_tag(tag):
    # Match longer keys first (e.g. 7v2 before 7v).
    for key in sorted(SUPPORTED_VARSETS, key=len, reverse=True):
        if f"_{key}_" in tag:
            return key
    return None
