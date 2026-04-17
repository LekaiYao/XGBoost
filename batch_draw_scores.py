import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import uproot

from utils.paths import ensure_dir, selected_dir, train_group_tag

if len(sys.argv) < 2:
    print(
        "Usage: python3 batch_draw_scores.py "
        "[--output-tag <output_tag>] [--output-prefix <prefix>] "
        "<train_tag> [<train_tag> ...]"
    )
    sys.exit(1)

args = sys.argv[1:]
output_tag = None
output_prefix = ""
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
    train_tags.append(token)
    i += 1

group_tag = train_group_tag(train_tags)
output_tag = output_tag or group_tag
TREE = "ntmix"
MASS_RANGE = (3.62, 4.0)
BINS = np.arange(MASS_RANGE[0], MASS_RANGE[1] + 0.01, 0.01)
if output_tag.startswith("pb23v6_") or any(tag.startswith("pb23v6_") for tag in train_tags):
    FID_LABEL = "fid3"
    BQVALUE_MAX = 0.2
    BY_MAX = 1.2
    BPT_MIN = 10.0
    BPT_MAX = 50.0
    CENTBIN_MIN = 20.0
    CENTBIN_USE_MAX = False
    CENTBIN_MAX = None
else:
    FID_LABEL = "fid"
    BQVALUE_MAX = 0.13
    BY_MAX = 1.6
    BPT_MIN = 15.0
    BPT_MAX = 50.0
    CENTBIN_MIN = 0
    CENTBIN_USE_MAX = True
    CENTBIN_MAX = 90
REFERENCE_MASSES = [3.686, 3.872]
score_cuts = [0.0, 0.5, 0.7, 0.8, 0.82, 0.84, 0.86, 0.88, 0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 0.993, 0.996]
X3872_MASS = 3.872
SIGMA_SUMMARY_NAME = f"{output_prefix}X3872_sigma_summary_{FID_LABEL}.md"

input_file = os.path.join(selected_dir(output_tag), f"{output_prefix}DATA_with_score.root")
if not os.path.exists(input_file):
    print(f"Grouped DATA file not found for output_tag={output_tag}: {input_file}")
    sys.exit(1)

root_file = uproot.open(input_file)
tree = root_file[TREE]
available_branches = set(tree.keys())
score_branches = [f"xgb_score_{train_tag}" for train_tag in train_tags]
valid_train_tags = []
missing_train_tags = []
for train_tag, score_branch in zip(train_tags, score_branches):
    if score_branch in available_branches:
        valid_train_tags.append(train_tag)
    else:
        missing_train_tags.append(train_tag)

if not valid_train_tags:
    print("No valid score branches found in grouped DATA file; nothing to draw.")
    print(f"Missing train tags: {missing_train_tags}")
    sys.exit(1)

branches = ["Bmass", "BQvalue", "By", "Bpt", "CentBin"] + [f"xgb_score_{tag}" for tag in valid_train_tags]

print(f"Loading grouped DATA: {input_file}")
df = tree.arrays(branches, library="pd")
print(f"Total events: {len(df)}")
print(f"Drawing train tags: {valid_train_tags}")
if missing_train_tags:
    print(f"Skipping missing score branches for train tags: {missing_train_tags}")

df_base = df[
    (df["Bmass"] > MASS_RANGE[0])
    & (df["Bmass"] < MASS_RANGE[1])
    & (df["BQvalue"] < BQVALUE_MAX)
]
fid_mask = (
    (np.abs(df_base["By"]) < BY_MAX)
    & (df_base["Bpt"] > BPT_MIN)
    & (df_base["Bpt"] < BPT_MAX)
    & (df_base["CentBin"] > CENTBIN_MIN)
)
if CENTBIN_USE_MAX and CENTBIN_MAX is not None:
    fid_mask = fid_mask & (df_base["CentBin"] < CENTBIN_MAX)
df_fid = df_base[fid_mask]

print(f"After mass + BQvalue<{BQVALUE_MAX}: {len(df_base)}")
if CENTBIN_USE_MAX and CENTBIN_MAX is not None:
    centbin_desc = f"{CENTBIN_MIN} < CentBin < {CENTBIN_MAX}"
else:
    centbin_desc = f"CentBin > {CENTBIN_MIN}"
print("Using fiducial profile:", FID_LABEL)
print(
    "After mass + fiducial cuts: "
    f"BQvalue<{BQVALUE_MAX}, |By|<{BY_MAX}, "
    f"{BPT_MIN}<Bpt<{BPT_MAX}, {centbin_desc}: {len(df_fid)}"
)


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

    sigma = (n_bin - b_hat) / np.sqrt(b_hat)
    return {"n_bin": n_bin, "b_hat": b_hat, "sigma": float(sigma)}


sigma_summary = {}

for train_tag in valid_train_tags:
    score_column = f"xgb_score_{train_tag}"
    output_dir = ensure_dir(os.path.join(selected_dir(output_tag), f"{output_prefix}{train_tag}"))
    sigma_summary[train_tag] = []

    for cut in score_cuts:
        cut_tag = int(round(cut * 1000)) if cut >= 0.99 else int(round(cut * 100))
        df_cut = df_fid[df_fid[score_column] > cut]
        sigma_result = x3872_sigma_from_masses(df_cut["Bmass"])
        if sigma_result is not None:
            sigma_summary[train_tag].append({"cut": float(cut), **sigma_result})

        plt.figure(figsize=(6, 6))
        plt.hist(
            df_cut["Bmass"],
            bins=BINS,
            histtype="step",
            linewidth=2,
        )
        for mass in REFERENCE_MASSES:
            plt.axvline(mass, linestyle="--", linewidth=1.2, color="gray", alpha=0.8)

        n_entries = len(df_cut)
        mean = df_cut["Bmass"].mean()
        std = df_cut["Bmass"].std()
        textstr = "\n".join(
            (
                f"Entries = {n_entries}",
                f"Mean = {mean:.4f}",
                f"Std Dev = {std:.4f}",
                f"BQvalue < {BQVALUE_MAX}",
                f"|By| < {BY_MAX}",
                f"{BPT_MIN} < Bpt < {BPT_MAX}",
                centbin_desc,
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
            out_name = os.path.join(output_dir, f"DATA_{FID_LABEL}_cut{cut_tag:04d}.pdf")
        else:
            out_name = os.path.join(output_dir, f"DATA_{FID_LABEL}_cut{cut_tag:03d}.pdf")

        plt.savefig(out_name)
        plt.close()
        print(f"Saved [fid]: {out_name}")

summary_path = os.path.join(selected_dir(output_tag), SIGMA_SUMMARY_NAME)
with open(summary_path, "w") as f:
    f.write(f"# X(3872) Sigma Summary ({FID_LABEL})\n\n")
    f.write(
        "Definition:\n"
        "- Signal bin: the histogram bin containing Bmass = 3.872\n"
        "- sigma = (N_bin - B_hat) / sqrt(B_hat)\n"
        "- B_hat: mean of sideband bins [i-2, i-1, i+1, i+2]\n"
        f"- {FID_LABEL} cuts: BQvalue < {BQVALUE_MAX}, |By| < {BY_MAX}, {BPT_MIN} < Bpt < {BPT_MAX}, "
        f"{centbin_desc}\n"
        f"- bins: {MASS_RANGE[0]} to {MASS_RANGE[1]} with width 0.01\n\n"
    )
    if missing_train_tags:
        f.write("## Skipped Train Tags (missing score branches)\n\n")
        for train_tag in missing_train_tags:
            f.write(f"- {train_tag}\n")
        f.write("\n")

    for train_tag in valid_train_tags:
        results = sigma_summary.get(train_tag, [])
        ge3 = [r for r in results if r["sigma"] >= 3.0]
        ge5 = [r for r in results if r["sigma"] >= 5.0]
        ge3.sort(key=lambda item: item["cut"])
        ge5.sort(key=lambda item: item["cut"])

        f.write(f"## {train_tag}\n\n")
        f.write("### >= 3 sigma\n")
        if ge3:
            for item in ge3:
                f.write(
                    f"- cut>{format_cut(item['cut'])}: sigma={item['sigma']:.3f}, "
                    f"N_bin={item['n_bin']:.0f}, B_hat={item['b_hat']:.3f}\n"
                )
        else:
            f.write("- none\n")
        f.write("\n")

        f.write("### >= 5 sigma\n")
        if ge5:
            for item in ge5:
                f.write(
                    f"- cut>{format_cut(item['cut'])}: sigma={item['sigma']:.3f}, "
                    f"N_bin={item['n_bin']:.0f}, B_hat={item['b_hat']:.3f}\n"
                )
        else:
            f.write("- none\n")
        f.write("\n")

print(f"Saved sigma summary: {summary_path}")
print(f"\nAll plots saved for group: {group_tag}")
