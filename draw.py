import sys
import os
import uproot
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =========================
# train tag
# =========================
if len(sys.argv) != 2:
    print("Usage: python3 draw.py <train_tag>")
    sys.exit(1)

train_tag = sys.argv[1]

# =========================
# input / output
# =========================
FILE = f"selected_events/DATA_with_score_{train_tag}.root"
TREE = "tree"

OUTPUT_DIR = f"selected_events/{train_tag}_pdf"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# config
# =========================
MASS_RANGE = (3.6, 4.0)
score_cuts = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# =========================
# load data
# =========================
print(f"Loading: {FILE}")
df = uproot.open(FILE)[TREE].arrays(library="pd")
print(f"Total events: {len(df)}")

# apply mass window
df = df[(df["Bmass"] > MASS_RANGE[0]) & (df["Bmass"] < MASS_RANGE[1])]
print(f"After mass cut: {len(df)}")

# bins
bins = np.linspace(MASS_RANGE[0], MASS_RANGE[1], 80)

# =========================
# loop over score cuts
# =========================
for cut in score_cuts:

    df_cut = df[df["xgb_score"] > cut]

    plt.figure(figsize=(6,6))

    plt.hist(
        df_cut["Bmass"],
        bins=bins,
        histtype="step",
        linewidth=2
    )

    # stats box (ROOT-like)
    n_entries = len(df_cut)
    mean = df_cut["Bmass"].mean()
    std = df_cut["Bmass"].std()

    textstr = '\n'.join((
        f'Entries = {n_entries}',
        f'Mean = {mean:.4f}',
        f'Std Dev = {std:.4f}'
    ))

    plt.text(
        0.97, 0.97, textstr,
        transform=plt.gca().transAxes,
        fontsize=10,
        verticalalignment='top',
        horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
    )

    plt.xlabel("Bmass (GeV)")
    plt.ylabel("Entries")
    plt.title(f"{train_tag} | score > {cut}")

    plt.xlim(MASS_RANGE[0], MASS_RANGE[1])

    plt.grid(alpha=0.3)
    plt.tight_layout()

    cut_tag = int(cut * 100)   # 0.1 → 10
    out_name = os.path.join(OUTPUT_DIR, f"X_cut{cut_tag:03d}.pdf")

    plt.savefig(out_name)
    plt.close()

    print(f"Saved: {out_name}")

print(f"\nAll plots saved in: {OUTPUT_DIR}")