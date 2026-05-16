from utils.tagging import split_channel_tag

PBPB_VARSETS_X = {
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
}

PBPB_VARSETS_BU = {
    "4v": ["Btrk1dR", "Btrk2dR", "BtrkPtimb", "Bchi2Prob"],
    "4v2": ["Bchi2Prob", "Btrk1dR", "BtrkPtimb", "Btrk2Pt"],
}

PBPB_VARSETS_BD = {
    "4v": ["Btrk1dR", "Btrk2dR", "BtrkPtimb", "Bchi2Prob"],
    "4v2": ["Bchi2Prob", "Btrk1dR", "BtrkPtimb", "Btrk2Pt"],
}

PBPB_VARSETS_BS = {
    "4v": ["Btrk1dR", "Btrk2dR", "BtrkPtimb", "Bchi2Prob"],
    "4v2": ["Bchi2Prob", "Btrk1dR", "BtrkPtimb", "Btrk2Pt"],
}

PP_VARSETS_X = {
    "5v2": ["Bchi2Prob", "Btrk1dR", "Btrk2dR", "Btrk1Pt", "Btrk2Pt"],
    "6v4": ["Btrk2dR", "BtrkPtimb", "Btrk1Pt", "Btrk2Pt", "BtktkvProb", "Bcos_dtheta"],
}

PP_VARSETS_BU = {
    "5v2": ["Bchi2Prob", "Btrk1dR", "Btrk2dR", "Btrk1Pt", "Btrk2Pt"],
}

PP_VARSETS_BD = {
    "5v2": ["Bchi2Prob", "Btrk1dR", "Btrk2dR", "Btrk1Pt", "Btrk2Pt"],
}

PP_VARSETS_BS = {
    "5v2": ["Bchi2Prob", "Btrk1dR", "Btrk2dR", "Btrk1Pt", "Btrk2Pt"],
}

VARSETS = {
    "pbpb": {
        "X": PBPB_VARSETS_X,
        "Bu": PBPB_VARSETS_BU,
        "Bd": PBPB_VARSETS_BD,
        "Bs": PBPB_VARSETS_BS,
    },
    "pp": {
        "X": PP_VARSETS_X,
        "Bu": PP_VARSETS_BU,
        "Bd": PP_VARSETS_BD,
        "Bs": PP_VARSETS_BS,
    },
}

DEFAULT_SAMPLE = "pbpb"
SUPPORTED_SAMPLES = tuple(VARSETS.keys())
SUPPORTED_VARSETS = tuple(sorted({k for ch in VARSETS[DEFAULT_SAMPLE].values() for k in ch.keys()}))
VARSET_COLUMNS = VARSETS[DEFAULT_SAMPLE]["X"]


def infer_sample_from_tag(tag):
    _, body = split_channel_tag(tag)
    if body.startswith("pp"):
        return "pp"
    return "pbpb"


def infer_channel_from_tag(tag):
    channel, _ = split_channel_tag(tag)
    return channel


def get_varset_columns(sample, varset, channel):
    if sample not in VARSETS:
        raise ValueError(f"Unsupported sample '{sample}'. Expected one of {SUPPORTED_SAMPLES}.")
    by_channel = VARSETS[sample]
    if channel not in by_channel:
        raise ValueError(f"Unsupported channel '{channel}' for sample '{sample}'. Expected one of {tuple(by_channel.keys())}.")
    if varset not in by_channel[channel]:
        raise ValueError(f"Unsupported varset '{varset}' for sample '{sample}' channel '{channel}'.")
    return list(by_channel[channel][varset])


def infer_varset_from_tag(tag, sample=None):
    sample_key = sample or infer_sample_from_tag(tag)
    channel = infer_channel_from_tag(tag)
    candidates = tuple(VARSETS.get(sample_key, {}).get(channel, {}).keys())
    for key in sorted(candidates, key=len, reverse=True):
        if f"_{key}_" in tag:
            return key
    return None
