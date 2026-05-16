from utils.tagging import split_channel_tag

_BASE_VARSETS = {
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

VARSETS = {
    sample: {
        "default": dict(varsets),
        "X": dict(varsets),
        "Bu": dict(varsets),
        "Bd": dict(varsets),
        "Bs": dict(varsets),
    }
    for sample, varsets in _BASE_VARSETS.items()
}

DEFAULT_SAMPLE = "pbpb"
SUPPORTED_SAMPLES = tuple(VARSETS.keys())
SUPPORTED_VARSETS = tuple(VARSETS[DEFAULT_SAMPLE]["default"].keys())
VARSET_COLUMNS = VARSETS[DEFAULT_SAMPLE]["default"]


def infer_sample_from_tag(tag):
    _, body = split_channel_tag(tag)
    if body.startswith("pp"):
        return "pp"
    return "pbpb"


def infer_channel_from_tag(tag):
    channel, _ = split_channel_tag(tag)
    return channel


def get_varset_columns(sample, varset, channel="default"):
    if sample not in VARSETS:
        raise ValueError(f"Unsupported sample '{sample}'. Expected one of {SUPPORTED_SAMPLES}.")
    by_channel = VARSETS[sample]
    channel_key = channel if channel in by_channel else "default"
    if varset not in by_channel[channel_key]:
        raise ValueError(f"Unsupported varset '{varset}' for sample '{sample}' channel '{channel_key}'.")
    return list(by_channel[channel_key][varset])


def infer_varset_from_tag(tag, sample=None):
    sample_key = sample or infer_sample_from_tag(tag)
    candidates = tuple(VARSETS.get(sample_key, {}).get("default", {}).keys())
    for key in sorted(candidates, key=len, reverse=True):
        if f"_{key}_" in tag:
            return key
    return None
