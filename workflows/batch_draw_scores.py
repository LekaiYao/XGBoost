import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import uproot

from configs.samples import (
    infer_channel_from_tag,
    infer_dataset_year,
    infer_fid_profile,
    infer_sample_from_tag,
    resolve_draw_config,
    resolve_fiducial_config,
)
from utils.paths import ensure_dir, selected_dir, train_group_tag

if len(sys.argv) < 2:
    print(
        "Usage: python3 workflows/batch_draw_scores.py "
        "[--output-tag <output_tag>] [--output-prefix <prefix>] [--fid-profile <auto|fid|fid2|fid3>] "
        "<train_tag> [<train_tag> ...]"
    )
    sys.exit(1)

args = sys.argv[1:]
output_tag = None
output_prefix = ""
fid_profile = "auto"
train_tags = []

i = 0
while i < len(args):
    token = args[i]
    if token == "--output-tag" and i + 1 < len(args):
        output_tag = args[i + 1]
        i += 2
        continue
    if token == "--output-prefix" and i + 1 < len(args):
        output_prefix = args[i + 1]
        i += 2
        continue
    if token == "--fid-profile" and i + 1 < len(args):
        fid_profile = args[i + 1]
        i += 2
        continue
    train_tags.append(token)
    i += 1

if fid_profile not in {"auto", "fid", "fid2", "fid3"}:
    raise ValueError("--fid-profile must be one of: auto, fid, fid2, fid3")

group_tag = train_group_tag(train_tags)
output_tag = output_tag or group_tag
sample_key = infer_sample_from_tag(output_tag)
channel = infer_channel_from_tag(output_tag)
dataset_year = infer_dataset_year(output_tag, sample_key)
active_fid = infer_fid_profile(output_tag, sample_key) if fid_profile == "auto" else fid_profile
fid_cfg = resolve_fiducial_config(sample_key, channel, active_fid)
draw_cfg = resolve_draw_config(sample_key, channel, dataset_year)

TREE = draw_cfg["data"]["tree"]
MASS_RANGE = tuple(draw_cfg["plot"]["mass_range"])
BIN_WIDTH = float(draw_cfg["plot"]["bin_width"])
BINS = np.arange(MASS_RANGE[0], MASS_RANGE[1] + BIN_WIDTH, BIN_WIDTH)
REFERENCE_MASSES = list(draw_cfg["plot"].get("reference_masses", []))
score_cuts = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.90, 0.95]

input_file = os.path.join(selected_dir(output_tag), f"{output_prefix}DATA_with_score.root")
if not os.path.exists(input_file):
    print(f"Grouped DATA file not found: {input_file}")
    sys.exit(1)

root_file = uproot.open(input_file)
tree = root_file[TREE]
available_branches = set(tree.keys())
score_branch_map = {}
valid_train_tags, missing_train_tags = [], []
for train_tag in train_tags:
    tagged_branch = f"xgb_score_{train_tag}"
    if tagged_branch in available_branches:
        score_branch_map[train_tag] = tagged_branch
        valid_train_tags.append(train_tag)
    elif "xgb_score" in available_branches:
        score_branch_map[train_tag] = "xgb_score"
        valid_train_tags.append(train_tag)
    else:
        missing_train_tags.append(train_tag)

if not valid_train_tags:
    print("No valid score branches found.")
    sys.exit(1)

branches = ["Bmass", "BQvalue", "By", "Bpt", "CentBin"] + sorted({score_branch_map[tag] for tag in valid_train_tags})
df = tree.arrays(branches, library="pd")

df_base = df[(df["Bmass"] > MASS_RANGE[0]) & (df["Bmass"] < MASS_RANGE[1])]
if fid_cfg.get("bqvalue_max") is not None:
    df_base = df_base[df_base["BQvalue"] < fid_cfg["bqvalue_max"]]

fid_mask = np.ones(len(df_base), dtype=bool)
if fid_cfg.get("by_max") is not None:
    fid_mask &= np.abs(df_base["By"]) < fid_cfg["by_max"]
if fid_cfg.get("bpt_min") is not None:
    fid_mask &= df_base["Bpt"] > fid_cfg["bpt_min"]
if fid_cfg.get("bpt_max") is not None:
    fid_mask &= df_base["Bpt"] < fid_cfg["bpt_max"]
if fid_cfg.get("centbin_min") is not None:
    fid_mask &= df_base["CentBin"] > fid_cfg["centbin_min"]
if fid_cfg.get("centbin_max") is not None:
    fid_mask &= df_base["CentBin"] < fid_cfg["centbin_max"]
df_fid = df_base[fid_mask]

# Significance scan is intentionally disabled for now.
for train_tag in valid_train_tags:
    score_column = score_branch_map[train_tag]
    cut_scan_root = ensure_dir(os.path.join(selected_dir(output_tag), "cut_scan"))
    if len(valid_train_tags) == 1:
        output_dir = cut_scan_root
    else:
        output_dir = ensure_dir(os.path.join(cut_scan_root, f"{output_prefix}{train_tag}"))
    for cut in score_cuts:
        cut_tag = int(round(cut * 1000)) if cut >= 0.99 else int(round(cut * 100))
        df_cut = df_fid[df_fid[score_column] > cut]

        plt.figure(figsize=(6, 6))
        plt.hist(df_cut["Bmass"], bins=BINS, histtype="step", linewidth=2)
        if channel == "X":
            for mass in REFERENCE_MASSES:
                plt.axvline(mass, linestyle="--", linewidth=1.2, color="gray", alpha=0.8)
        plt.xlabel("Bmass (GeV)")
        plt.ylabel("Entries")
        plt.title(f"{train_tag} | score > {cut}")
        plt.xlim(MASS_RANGE[0], MASS_RANGE[1])
        plt.grid(alpha=0.3)
        plt.tight_layout()

        if cut >= 0.99:
            out_name = os.path.join(output_dir, f"DATA_{active_fid}_cut{cut_tag:04d}.pdf")
        else:
            out_name = os.path.join(output_dir, f"DATA_{active_fid}_cut{cut_tag:03d}.pdf")
        plt.savefig(out_name)
        plt.close()
