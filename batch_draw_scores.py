import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import uproot

from utils.paths import cut_scan_dir, data_output_path, ensure_dir, train_group_tag

if len(sys.argv) < 2:
    print("Usage: python3 batch_draw_scores.py <train_tag> [<train_tag> ...]")
    sys.exit(1)

train_tags = sys.argv[1:]
group_tag = train_group_tag(train_tags)
TREE = "ntmix"
MASS_RANGE = (3.62, 4.0)
BINS = np.arange(MASS_RANGE[0], MASS_RANGE[1] + 0.01, 0.01)
BQVALUE_MAX = 0.13
score_cuts = [0.0, 0.5, 0.6, 0.8, 0.9, 0.92, 0.94, 0.95, 0.96, 0.97, 0.99, 0.993, 0.995]

input_file = data_output_path(group_tag)
if not os.path.exists(input_file):
    print(f"Grouped DATA file not found for group_tag={group_tag}: {input_file}")
    sys.exit(1)

score_branches = [f"xgb_score_{train_tag}" for train_tag in train_tags]
branches = ["Bmass", "BQvalue"] + score_branches

print(f"Loading grouped DATA: {input_file}")
df = uproot.open(input_file)[TREE].arrays(branches, library="pd")
print(f"Total events: {len(df)}")

df = df[
    (df["Bmass"] > MASS_RANGE[0])
    & (df["Bmass"] < MASS_RANGE[1])
    & (df["BQvalue"] < BQVALUE_MAX)
]
print(f"After mass + BQvalue<{BQVALUE_MAX}: {len(df)}")

for train_tag in train_tags:
    score_column = f"xgb_score_{train_tag}"
    output_dir = ensure_dir(cut_scan_dir(train_tag))

    for cut in score_cuts:
        df_cut = df[df[score_column] > cut]

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

        if cut >= 0.99:
            cut_tag = int(round(cut * 1000))
            out_name = os.path.join(output_dir, f"DATA_cut{cut_tag:04d}.pdf")
        else:
            cut_tag = int(round(cut * 100))
            out_name = os.path.join(output_dir, f"DATA_cut{cut_tag:03d}.pdf")

        plt.savefig(out_name)
        plt.close()
        print(f"Saved: {out_name}")

print(f"\nAll plots saved for group: {group_tag}")
