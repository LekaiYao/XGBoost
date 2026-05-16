VARSETS = {
    "pbpb": {
        "4v": ["Btrk1dR", "Btrk2dR", "BtrkPtimb", "Bchi2Prob"],
        "4v2": ["Bchi2Prob", "Btrk1dR", "BtrkPtimb", "Btrk2Pt"],
        "5v": ["Bchi2Prob", "Btrk1dR", "Btrk2dR", "BtrkPtimb", "Btrk2Pt"],
        "5v2": ["Bchi2Prob", "Btrk1dR", "Btrk2dR", "Btrk1Pt", "Btrk2Pt"],
        "5v3": ["Btrk1dR", "Btrk2dR", "BtrkPtimb", "Btrk2Pt", "Bcos_dtheta"],
        "6v": ["Bchi2Prob", "Btrk1dR", "BtrkPtimb", "Btrk2Pt", "BtktkvProb", "Bcos_dtheta"],
        "6v2": ["Bchi2Prob", "Btrk1dR", "Btrk2dR", "BtrkPtimb", "BtktkvProb", "Bcos_dtheta"],
        "6v3": ["Bchi2Prob", "Btrk1dR", "Btrk2dR", "BtrkPtimb", "Btrk2Pt", "Bcos_dtheta"],
        "6v4": ["Btrk2dR", "BtrkPtimb", "Btrk1Pt", "Btrk2Pt", "BtktkvProb", "Bcos_dtheta"],
        "7v2": ["Bchi2Prob", "Btrk1dR", "Btrk2dR", "BtrkPtimb", "Btrk2Pt", "BtktkvProb", "Bcos_dtheta"],
        "7v3": ["Bchi2Prob", "Btrk1dR", "BtrkPtimb", "Btrk1Pt", "Btrk2Pt", "BtktkvProb", "Bcos_dtheta"],
        "7v4": ["Bpt", "Bchi2Prob", "Btrk1dR", "Btrk2dR", "BtrkPtimb", "Btrk2Pt", "Bcos_dtheta"],
        "8v": ["Bpt", "Bchi2Prob", "Btrk1dR", "BtrkPtimb", "Btrk1Pt", "Btrk2Pt", "BtktkvProb", "Bcos_dtheta"],
        "8v2": ["Bchi2Prob", "Btrk1dR", "Btrk2dR", "BtrkPtimb", "Btrk2Pt", "BtktkvProb", "Bcos_dtheta", "Btrk1Pt"],
        "8v3": ["Bchi2Prob", "Btrk1dR", "Btrk2dR", "BtrkPtimb", "Btrk2Pt", "BtktkvProb", "Bcos_dtheta", "Btktkpt"],
        "8v4": ["Bpt", "Bchi2Prob", "Btrk1dR", "Btrk2dR", "BtrkPtimb", "Btrk2Pt", "BtktkvProb", "Bcos_dtheta"],
        "9v": ["Bchi2Prob", "Btrk1dR", "Btrk2dR", "BtrkPtimb", "Btktkpt", "Bcos_dtheta", "Btrk1Pt", "Btrk2Pt", "BtktkvProb"],
    },
    "pp": {
        "5v2": ["Bchi2Prob", "Btrk1dR", "Btrk2dR", "Btrk1Pt", "Btrk2Pt"],
        "6v4": ["Btrk2dR", "BtrkPtimb", "Btrk1Pt", "Btrk2Pt", "BtktkvProb", "Bcos_dtheta"],
    },
}

DEFAULT_SAMPLE = "pbpb"
SUPPORTED_SAMPLES = tuple(VARSETS.keys())
SUPPORTED_VARSETS = tuple(VARSETS[DEFAULT_SAMPLE].keys())
VARSET_COLUMNS = VARSETS[DEFAULT_SAMPLE]


def infer_sample_from_tag(tag):
    if tag.startswith("pp"):
        return "pp"
    return "pbpb"


def get_varset_columns(sample, varset):
    if sample not in VARSETS:
        raise ValueError(f"Unsupported sample '{sample}'. Expected one of {SUPPORTED_SAMPLES}.")
    if varset not in VARSETS[sample]:
        raise ValueError(f"Unsupported varset '{varset}' for sample '{sample}'.")
    return list(VARSETS[sample][varset])


def infer_varset_from_tag(tag, sample=None):
    sample_key = sample or infer_sample_from_tag(tag)
    candidates = tuple(VARSETS.get(sample_key, {}).keys())
    for key in sorted(candidates, key=len, reverse=True):
        if f"_{key}_" in tag:
            return key
    return None
