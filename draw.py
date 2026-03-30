import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import uproot

from paths import cut_scan_dir, ensure_dir, resolve_data_output_path

if len(sys.argv) != 2:
    print("Usage: python3 draw.py <train_tag>")
    sys.exit(1)

train_tag = sys.argv[1]
TREE = "tree"
MASS_RANGE = (3.6, 4.0)
score_cuts = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

input_file = resolve_data_output_path(train_tag)
output_dir = ensure_dir(cut_scan_dir(train_tag))

if not os.path.exists(input_file):
    print(f"Input file not found for train_tag={train_tag}: {input_file}")
    sys.exit(1)

print(f"Loading: {input_file}")
df = uproot.open(input_file)[TREE].arrays(library="pd")
print(f"Total events: {len(df)}")

df = df[(df["Bmass"] > MASS_RANGE[0]) & (df["Bmass"] < MASS_RANGE[1])]
print(f"After mass cut: {len(df)}")

bins = np.linspace(MASS_RANGE[0], MASS_RANGE[1], 80)

for cut in score_cuts:
    df_cut = df[df["xgb_score"] > cut]

    plt.figure(figsize=(6, 6))
    plt.hist(
        df_cut["Bmass"],
        bins=bins,
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

    cut_tag = int(cut * 100)
    out_name = os.path.join(output_dir, f"X_cut{cut_tag:03d}.pdf")
    plt.savefig(out_name)
    plt.close()

    print(f"Saved: {out_name}")

print(f"\nAll plots saved in: {output_dir}")
