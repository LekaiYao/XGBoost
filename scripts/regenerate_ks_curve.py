"""Regenerate KS curve plot from an existing ks.json file.

Usage:
    .venv/bin/python scripts/regenerate_ks_curve.py <train_tag>
    .venv/bin/python scripts/regenerate_ks_curve.py X_pp24_v3_fid2_4v1_xgb_v1

Reads output/training/<train_tag>/ks.json and overwrites
output/training/<train_tag>/ks_curve.pdf.
"""
import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Regenerate KS curve plot from ks.json")
    parser.add_argument("train_tag", help="Training tag (e.g. X_pp24_v3_fid2_4v1_xgb_v1)")
    args = parser.parse_args()

    ks_json_path = os.path.join("output", "training", args.train_tag, "ks.json")
    if not os.path.exists(ks_json_path):
        raise FileNotFoundError(f"ks.json not found: {ks_json_path}")

    with open(ks_json_path) as f:
        ks = json.load(f)

    # CMS AN style
    _cms_an_rc = {
        "font.family": "serif",
        "font.size": 13,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 11,
        "legend.frameon": False,
        "lines.linewidth": 1.5,
    }
    _saved_rc = {k: plt.rcParams.get(k) for k in _cms_an_rc}
    plt.rcParams.update(_cms_an_rc)

    ks_plot_path = os.path.join("output", "training", args.train_tag, "ks_curve.pdf")

    plt.figure(figsize=(7, 5))
    # Train set: blue family
    plt.plot(
        ks["train"]["score_thresholds"], ks["train"]["sig_cdf"],
        color="tab:blue", linestyle="-",
        label="Train signal",
    )
    plt.plot(
        ks["train"]["score_thresholds"], ks["train"]["bkg_cdf"],
        color="tab:blue", linestyle="--",
        label="Train background",
    )
    # Test set: orange family
    plt.plot(
        ks["test"]["score_thresholds"], ks["test"]["sig_cdf"],
        color="tab:orange", linestyle="-",
        label="Test signal",
    )
    plt.plot(
        ks["test"]["score_thresholds"], ks["test"]["bkg_cdf"],
        color="tab:orange", linestyle="--",
        label="Test background",
    )
    # KS values in upper-left text box (avoids overlap with legend at lower right)
    _ks_text = (
        f"Train KS = {ks['train']['ks_stat']:.3f}\n"
        f"Test KS  = {ks['test']['ks_stat']:.3f}"
    )
    plt.text(0.03, 0.97, _ks_text, transform=plt.gca().transAxes,
             fontsize=10, ha="left", va="top",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85,
                       edgecolor="gray", linewidth=0.5))
    plt.xlabel("BDT score")
    plt.ylabel("Cumulative fraction")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(alpha=0.25, linestyle="--", linewidth=0.5)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(ks_plot_path)
    plt.close()
    print(f"KS curve plot regenerated: {ks_plot_path}")

    plt.rcParams.update(_saved_rc)


if __name__ == "__main__":
    main()
