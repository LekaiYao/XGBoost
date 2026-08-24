from copy import deepcopy


YEAR_PAIRINGS = {
    "X_pb23_v3_fid3_6v5_rwr6range5v1_xgb_v1": {
        "anchor_dataset": "pb23",
        "paired_pb24_tag": "X_pb24_v19_fid19_6v5_rwr6range5v1_xgb_v1",
        "selection_policy": "year-specific score thresholds matched at common weighted X efficiency",
        "fit_scan_efficiencies": [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40],
    },
}


def resolve_year_pairing(anchor_train_tag: str) -> dict:
    pairing = YEAR_PAIRINGS.get(anchor_train_tag)
    if pairing is None:
        raise ValueError(
            f"No PbPb year pairing configured for anchor tag '{anchor_train_tag}'. "
            f"Available anchors: {tuple(YEAR_PAIRINGS)}"
        )
    return {
        "anchor_train_tag": anchor_train_tag,
        "tags": {
            "pb23": anchor_train_tag,
            "pb24": pairing["paired_pb24_tag"],
        },
        **deepcopy(pairing),
    }
