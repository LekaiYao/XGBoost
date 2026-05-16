import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import uproot

from configs.samples import infer_channel_from_tag, infer_fid_profile, infer_sample_from_tag, resolve_fiducial_config
from utils.paths import ensure_dir, selected_dir, train_group_tag

if len(sys.argv) < 2:
    print(
        "Usage: python3 workflows/batch_draw_scores.py "
        "[--output-tag <output_tag>] [--output-prefix <prefix>] [--fid-profile <auto|fid|fid3>] "
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

if fid_profile not in {"auto", "fid", "fid3"}:
    raise ValueError("--fid-profile must be one of: auto, fid, fid3")

group_tag = train_group_tag(train_tags)
output_tag = output_tag or group_tag
sample_key = infer_sample_from_tag(output_tag)
channel = infer_channel_from_tag(output_tag)
active_fid = infer_fid_profile(output_tag, sample_key) if fid_profile == "auto" else fid_profile
fid_cfg = resolve_fiducial_config(sample_key, channel, active_fid)

TREE = "ntmix"
MASS_RANGE = (3.62, 4.0)
BINS = np.arange(MASS_RANGE[0], MASS_RANGE[1] + 0.01, 0.01)
REFERENCE_MASSES = [3.686, 3.872]
score_cuts = [0.0, 0.5, 0.7, 0.8, 0.82, 0.84, 0.86, 0.88, 0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 0.993, 0.996]
X3872_MASS = 3.872
SIGMA_SUMMARY_NAME = f"{output_prefix}X3872_sigma_summary_{active_fid}.md"

input_file = os.path.join(selected_dir(output_tag), f"{output_prefix}DATA_with_score.root")
if not os.path.exists(input_file):
    print(f"Grouped DATA file not found: {input_file}")
    sys.exit(1)

root_file = uproot.open(input_file)
tree = root_file[TREE]
available_branches = set(tree.keys())
score_branches = [f"xgb_score_{train_tag}" for train_tag in train_tags]
valid_train_tags, missing_train_tags = [], []
for train_tag, score_branch in zip(train_tags, score_branches):
    (valid_train_tags if score_branch in available_branches else missing_train_tags).append(train_tag)

if not valid_train_tags:
    print("No valid score branches found.")
    sys.exit(1)

branches = ["Bmass", "BQvalue", "By", "Bpt", "CentBin"] + [f"xgb_score_{tag}" for tag in valid_train_tags]
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


def format_cut(cut):
    return f"{cut:.3f}" if cut >= 0.99 else f"{cut:.2f}"


def x3872_sigma_from_masses(masses):
    counts, edges = np.histogram(masses, bins=BINS)
    signal_bin = np.searchsorted(edges, X3872_MASS, side="right") - 1
    if signal_bin < 2 or signal_bin > len(counts) - 3:
        return None
    n_bin = float(counts[signal_bin])
    sideband_bins = [signal_bin - 2, signal_bin - 1, signal_bin + 1, signal_bin + 2]
    b_hat = float(np.mean([counts[idx] for idx in sideband_bins]))
    if b_hat <= 0.0:
        return None
    return {"n_bin": n_bin, "b_hat": b_hat, "sigma": float((n_bin - b_hat) / np.sqrt(b_hat))}


sigma_summary = {}
for train_tag in valid_train_tags:
    score_column = f"xgb_score_{train_tag}"
    cut_scan_root = ensure_dir(os.path.join(selected_dir(output_tag), "cut_scan"))
    if len(valid_train_tags) == 1:
        output_dir = cut_scan_root
    else:
        output_dir = ensure_dir(os.path.join(cut_scan_root, f"{output_prefix}{train_tag}"))
    sigma_summary[train_tag] = []
    for cut in score_cuts:
        cut_tag = int(round(cut * 1000)) if cut >= 0.99 else int(round(cut * 100))
        df_cut = df_fid[df_fid[score_column] > cut]
        sigma_result = x3872_sigma_from_masses(df_cut["Bmass"])
        if sigma_result is not None:
            sigma_summary[train_tag].append({"cut": float(cut), **sigma_result})

        plt.figure(figsize=(6, 6))
        plt.hist(df_cut["Bmass"], bins=BINS, histtype="step", linewidth=2)
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

summary_path = os.path.join(selected_dir(output_tag), SIGMA_SUMMARY_NAME)
with open(summary_path, "w") as f:
    f.write(f"# X(3872) Sigma Summary ({active_fid})\n\n")
    if missing_train_tags:
        f.write("## Skipped Train Tags (missing score branches)\n\n")
        for train_tag in missing_train_tags:
            f.write(f"- {train_tag}\n")
        f.write("\n")
    for train_tag in valid_train_tags:
        results = sigma_summary.get(train_tag, [])
        ge3 = sorted([r for r in results if r["sigma"] >= 3.0], key=lambda x: x["cut"])
        ge5 = sorted([r for r in results if r["sigma"] >= 5.0], key=lambda x: x["cut"])
        f.write(f"## {train_tag}\n\n")
        f.write("### >= 3 sigma\n")
        f.writelines([f"- cut>{format_cut(r['cut'])}: sigma={r['sigma']:.3f}, N_bin={r['n_bin']:.0f}, B_hat={r['b_hat']:.3f}\n" for r in ge3] or ["- none\n"])
        f.write("\n### >= 5 sigma\n")
        f.writelines([f"- cut>{format_cut(r['cut'])}: sigma={r['sigma']:.3f}, N_bin={r['n_bin']:.0f}, B_hat={r['b_hat']:.3f}\n" for r in ge5] or ["- none\n"])
        f.write("\n")

print(f"Saved sigma summary: {summary_path}")
