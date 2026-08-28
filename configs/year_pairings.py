from copy import deepcopy
from typing import Optional


YEAR_PAIRINGS = {
    "X_pb23_v3_fid3_6v5_rwr6range5v1_xgb_v1": {
        "anchor_dataset": "pb23",
        "paired_pb24_tag": "X_pb24_v19_fid19_6v5_rwr6range5v1_xgb_v1",
        "selection_policy": "year-specific score thresholds matched at common weighted X efficiency",
        "fit_scan_efficiencies": [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40],
    },
    "X_pb23_v4_fid4_5v3_rwr5v3v1_xgb_v1": {
        "anchor_dataset": "pb23",
        "paired_pb24_tag": "X_pb24_v20_fid20_5v3_rwr5v3v1_xgb_v1",
        "selection_policy": "year-specific score thresholds matched at common weighted X efficiency",
        "fit_scan_efficiencies": [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40],
    },
    "X_pb23_v5_fid3_6v5_rwr6range5v1_xgb_v1": {
        "anchor_dataset": "pb23",
        "paired_pb24_tag": "X_pb24_v21_fid19_6v5_rwr6range5v1_xgb_v1",
        "selection_policy": "year-specific score thresholds matched at common weighted X efficiency",
        "fit_scan_efficiencies": [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40],
    },
    "X_pb23_v3_fid3_5v3_rwr6range5v1_xgb_v1": {
        "anchor_dataset": "pb23",
        "paired_pb24_tag": "X_pb24_v19_fid19_5v3_rwr6range5v1_xgb_v1",
        "selection_policy": "year-specific score thresholds matched at common weighted X efficiency",
        "fit_scan_efficiencies": [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40],
    },
    "X_pb23_v3_fid3_7v3_rwr6range5bmuauxv1_xgb_v1": {
        "anchor_dataset": "pb23",
        "paired_pb24_tag": "X_pb24_v19_fid19_7v3_rwr6range5bmuauxv1_xgb_v1",
        "selection_policy": "year-specific score thresholds matched at common weighted X efficiency",
        "fit_scan_efficiencies": [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40],
    },
    "X_pb23_v3_fid3_7v4_rwr6range5bmuauxv1_xgb_v1": {
        "anchor_dataset": "pb23",
        "paired_pb24_tag": "X_pb24_v19_fid19_7v4_rwr6range5bmuauxv1_xgb_v1",
        "selection_policy": "year-specific score thresholds matched at common weighted X efficiency",
        "fit_scan_efficiencies": [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40],
    },
    "X_pb23_v3_fid3_8v3_rwr6range5bmuauxv1_xgb_v1": {
        "anchor_dataset": "pb23",
        "paired_pb24_tag": "X_pb24_v19_fid19_8v3_rwr6range5bmuauxv1_xgb_v1",
        "selection_policy": "year-specific score thresholds matched at common weighted X efficiency",
        "fit_scan_efficiencies": [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40],
    },
    "X_pb23_v3_fid3_7v5_rwr6range5v1_xgb_v1": {
        "anchor_dataset": "pb23",
        "paired_pb24_tag": "X_pb24_v19_fid19_7v5_rwr6range5v1_xgb_v1",
        "selection_policy": "year-specific score thresholds matched at common weighted X efficiency",
        "fit_scan_efficiencies": [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40],
    },
    "X_pb23_v3_fid3_8v4_rwr6range5v1_xgb_v1": {
        "anchor_dataset": "pb23",
        "paired_pb24_tag": "X_pb24_v19_fid19_8v4_rwr6range5v1_xgb_v1",
        "selection_policy": "year-specific score thresholds matched at common weighted X efficiency",
        "fit_scan_efficiencies": [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40],
    },
    "X_pb23_v3_fid3_8v5_rwr6range5v1_xgb_v1": {
        "anchor_dataset": "pb23",
        "paired_pb24_tag": "X_pb24_v19_fid19_8v5_rwr6range5v1_xgb_v1",
        "selection_policy": "year-specific score thresholds matched at common weighted X efficiency",
        "fit_scan_efficiencies": [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40],
    },
    "X_pb23_v3_fid3_8v6_rwr6range5v1_xgb_v1": {
        "anchor_dataset": "pb23",
        "paired_pb24_tag": "X_pb24_v19_fid19_8v6_rwr6range5v1_xgb_v1",
        "selection_policy": "year-specific score thresholds matched at common weighted X efficiency",
        "fit_scan_efficiencies": [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40],
    },
    "X_pb23_v3_fid3_9v3_rwr6range5v1_xgb_v1": {
        "anchor_dataset": "pb23",
        "paired_pb24_tag": "X_pb24_v19_fid19_9v3_rwr6range5v1_xgb_v1",
        "selection_policy": "year-specific score thresholds matched at common weighted X efficiency",
        "fit_scan_efficiencies": [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40],
    },
    "X_pb23_v3_fid3_10v1_rwr6range5v1_xgb_v1": {
        "anchor_dataset": "pb23",
        "paired_pb24_tag": "X_pb24_v19_fid19_10v1_rwr6range5v1_xgb_v1",
        "selection_policy": "year-specific score thresholds matched at common weighted X efficiency",
        "fit_scan_efficiencies": [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40],
    },
    "Psi2S_pb23_v1_fid1_6v1_rwr6range4v1_xgb_v1": {
        "anchor_dataset": "pb23",
        "paired_pb24_tag": "Psi2S_pb24_v1_fid1_6v1_rwr6range4v1_xgb_v1",
        "selection_policy": (
            "year-specific score thresholds matched at common weighted Psi2S efficiency"
        ),
        "fit_scan_efficiencies": [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40],
    },
    "Psi2S_pb23_v2_fid2_5v1_rwr5v1_xgb_v1": {
        "anchor_dataset": "pb23",
        "paired_pb24_tag": "Psi2S_pb24_v2_fid2_5v1_rwr5v1_xgb_v1",
        "selection_policy": (
            "year-specific score thresholds matched at common weighted Psi2S efficiency"
        ),
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


def resolve_year_pairing_for_tag(train_tag: str) -> Optional[dict]:
    """Resolve a configured pairing from either its PbPb23 or PbPb24 tag."""
    for anchor_train_tag, pairing in YEAR_PAIRINGS.items():
        if train_tag in (anchor_train_tag, pairing["paired_pb24_tag"]):
            return resolve_year_pairing(anchor_train_tag)
    return None
