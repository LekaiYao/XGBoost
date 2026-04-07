import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import uproot

from utils.paths import (
    cut_scan_dir,
    ensure_dir,
    resolve_data_output_path,
)

if len(sys.argv) != 2:
    print("Usage: python3 draw.py <train_tag>")
    sys.exit(1)

train_tag = sys.argv[1]
TREE = "ntmix"
MASS_RANGE = (3.62, 4.0)
BINS = np.arange(MASS_RANGE[0], MASS_RANGE[1] + 0.01, 0.01)
BQVALUE_MAX = 0.13
score_cuts = [0.0, 0.5, 0.6, 0.8, 0.9, 0.92, 0.94, 0.95, 0.96, 0.97, 0.99, 0.993, 0.995]


def output_name(output_dir, cut):
    if cut >= 0.99:
        cut_tag = int(round(cut * 1000))
        cut_suffix = f"cut{cut_tag:04d}"
    else:
        cut_tag = int(round(cut * 100))
        cut_suffix = f"cut{cut_tag:03d}"
    return os.path.join(output_dir, f"DATA_{cut_suffix}.pdf")


data_input_file = resolve_data_output_path(train_tag)
output_dir = ensure_dir(cut_scan_dir(train_tag))

if not os.path.exists(data_input_file):
    print(f"DATA input file not found for train_tag={train_tag}: {data_input_file}")
    sys.exit(1)

branches = ["Bmass", "BQvalue", "xgb_score"]

print(f"Loading DATA: {data_input_file}")
df_data = uproot.open(data_input_file)[TREE].arrays(branches, library="pd")
print(f"Total DATA events: {len(df_data)}")

if "xgb_score" not in df_data.columns:
    print("Missing score branch in DATA file: xgb_score")
    sys.exit(1)

df_data = df_data[
    (df_data["Bmass"] > MASS_RANGE[0])
    & (df_data["Bmass"] < MASS_RANGE[1])
    & (df_data["BQvalue"] < BQVALUE_MAX)
].copy()

print(f"After DATA mass + BQvalue<{BQVALUE_MAX}: {len(df_data)}")

for cut in score_cuts:
    df_cut = df_data[df_data["xgb_score"] > cut]

    plt.figure(figsize=(6, 6))
    plt.hist(
        df_cut["Bmass"],
        bins=BINS,
        histtype="step",
        linewidth=2,
    )

    n_entries = len(df_cut)
    mean = df_cut["Bmass"].mean()
    std = df_cut["Bmass"].std()
    textstr = "\n".join(
        (
            f"Entries = {n_entries}",
            f"Mean = {mean:.4f}",
            f"Std Dev = {std:.4f}",
            f"BQvalue < {BQVALUE_MAX}",
        )
    )

    plt.text(
        0.97,
        0.97,
        textstr,
        transform=plt.gca().transAxes,
        fontsize=10,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    plt.xlabel("Bmass (GeV)")
    plt.ylabel("Entries")
    plt.title(f"{train_tag} | score > {cut}")
    plt.xlim(MASS_RANGE[0], MASS_RANGE[1])
    plt.grid(alpha=0.3)
    plt.tight_layout()

    out_name = output_name(output_dir, cut)
    plt.savefig(out_name)
    plt.close()

    print(f"Saved: {out_name}")

print(f"\nAll plots saved in: {output_dir}")
