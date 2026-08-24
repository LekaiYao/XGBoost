import argparse
import json
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import uproot

from configs.samples import (
    infer_channel_from_tag,
    infer_dataset_token_from_tag,
    infer_dataset_year,
    infer_fid_profile,
    infer_sample_from_tag,
    resolve_draw_config,
    resolve_fiducial_config,
)
from configs.year_pairings import resolve_year_pairing
from utils.paths import cut_scan_dir, selected_dir
from utils.selection import apply_selection, selection_columns


def threshold_map(payload):
    rows = payload.get("thresholds", [])
    result = {float(row["target_efficiency"]): row for row in rows}
    if len(result) != len(rows):
        raise ValueError("Duplicate target efficiencies in thresholds payload")
    return result


def format_title(tag_a, tag_b, target, entries_a, entries_b):
    label = f"{tag_a} + {tag_b}"
    wrapped = "\n".join(textwrap.wrap(label, width=58, break_long_words=True, break_on_hyphens=False))
    return f"{wrapped}\nweighted X efficiency {target:.0%}; entries {entries_a} + {entries_b}"


def load_data(tag):
    sample = infer_sample_from_tag(tag)
    channel = infer_channel_from_tag(tag)
    year = infer_dataset_year(tag, sample)
    dataset = infer_dataset_token_from_tag(tag)
    fid_profile = infer_fid_profile(tag, sample)
    fiducial = resolve_fiducial_config(sample, channel, fid_profile)["expression"]
    draw_cfg = resolve_draw_config(sample, channel, year)
    input_path = Path(selected_dir(tag)) / "DATA_with_score.root"
    if not input_path.is_file():
        raise FileNotFoundError(f"Missing scored DATA: {input_path}")
    tree_name = draw_cfg["data"]["tree"]
    with uproot.open(input_path) as root_file:
        tree = root_file[tree_name]
        score_branch = f"Prediction_{tag}" if f"Prediction_{tag}" in tree else "Prediction"
        columns = list(dict.fromkeys(["Bmass", score_branch] + selection_columns(fiducial)))
        frame = tree.arrays(columns, library="pd")
    mass_range = tuple(float(x) for x in draw_cfg["plot"]["mass_range"])
    frame = frame[(frame["Bmass"] > mass_range[0]) & (frame["Bmass"] < mass_range[1])]
    frame = apply_selection(frame, fiducial, f"fiducial profile {fid_profile}")

    threshold_path = Path(cut_scan_dir(tag)) / "weighted_signal_efficiency" / "thresholds.json"
    if not threshold_path.is_file():
        raise FileNotFoundError(f"Missing weighted-efficiency thresholds: {threshold_path}")
    thresholds = json.loads(threshold_path.read_text())
    if thresholds.get("weight_branch") != "Reweight":
        raise ValueError(f"Tag '{tag}' thresholds are not Reweight-based")
    return {
        "tag": tag,
        "dataset": dataset,
        "input_path": str(input_path),
        "tree": tree_name,
        "score_branch": score_branch,
        "fid_profile": fid_profile,
        "fiducial_selection": fiducial,
        "mass_range": mass_range,
        "bin_width": float(draw_cfg["plot"]["bin_width"]),
        "reference_masses": [float(x) for x in draw_cfg["plot"].get("reference_masses", [])],
        "frame": frame,
        "threshold_path": str(threshold_path),
        "thresholds": threshold_map(thresholds),
    }


def main():
    parser = argparse.ArgumentParser(description="Draw combined-year DATA mass spectra at matched weighted X efficiencies.")
    parser.add_argument("anchor_train_tag")
    parser.add_argument("legacy_train_tags", nargs="*")
    args = parser.parse_args()

    pairing = resolve_year_pairing(args.anchor_train_tag)
    configured_tags = [pairing["tags"]["pb23"], pairing["tags"]["pb24"]]
    if args.legacy_train_tags and args.legacy_train_tags != configured_tags:
        raise ValueError(
            f"Explicit tags do not match pairing for '{args.anchor_train_tag}': "
            f"{args.legacy_train_tags} != {configured_tags}"
        )
    samples = [load_data(tag) for tag in configured_tags]
    for key in ("fiducial_selection", "mass_range", "bin_width", "reference_masses"):
        if samples[0][key] != samples[1][key]:
            raise ValueError(f"Combined draw requires matching {key}: {samples[0][key]} != {samples[1][key]}")
    targets_a = set(samples[0]["thresholds"])
    targets_b = set(samples[1]["thresholds"])
    if targets_a != targets_b:
        raise ValueError(f"Efficiency target mismatch: {sorted(targets_a)} != {sorted(targets_b)}")

    output_dir = Path(cut_scan_dir(args.anchor_train_tag)) / "combined_pb23_pb24_weighted_signal_efficiency"
    manifest_path = output_dir / "thresholds.json"
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite combined draw outputs: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    mass_range = samples[0]["mass_range"]
    bins = np.arange(mass_range[0], mass_range[1] + samples[0]["bin_width"], samples[0]["bin_width"])
    rows = []
    for target in sorted(targets_a, reverse=True):
        selected = []
        row = {"target_efficiency": target}
        for index, sample in enumerate(samples):
            threshold_row = sample["thresholds"][target]
            threshold = float(threshold_row["score_threshold"])
            frame = sample["frame"]
            cut_frame = frame[frame[sample["score_branch"]] > threshold]
            selected.append(cut_frame["Bmass"].to_numpy(dtype=float))
            suffix = "a" if index == 0 else "b"
            row[f"dataset_{suffix}"] = sample["dataset"]
            row[f"score_threshold_{suffix}"] = threshold
            row[f"achieved_efficiency_{suffix}"] = float(threshold_row["achieved_efficiency"])
            row[f"data_entries_{suffix}"] = int(len(cut_frame))
        combined_mass = np.concatenate(selected)
        row["combined_data_entries"] = int(len(combined_mass))

        plt.figure(figsize=(6, 6))
        plt.hist(combined_mass, bins=bins, histtype="step", linewidth=2)
        for mass in samples[0]["reference_masses"]:
            plt.axvline(mass, linestyle="--", linewidth=1.2, color="gray", alpha=0.8)
        plt.xlabel("Bmass (GeV)")
        plt.ylabel("Entries")
        plt.title(
            format_title(configured_tags[0], configured_tags[1], target, len(selected[0]), len(selected[1])),
            fontsize=9,
        )
        plt.xlim(mass_range)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / f"DATA_pb23_pb24_eff{int(round(target * 100)):03d}.pdf", bbox_inches="tight")
        plt.close()
        rows.append(row)

    sample_metadata = []
    for sample in samples:
        sample_metadata.append({key: sample[key] for key in (
            "tag", "dataset", "input_path", "tree", "score_branch", "fid_profile",
            "fiducial_selection", "threshold_path",
        )})
    manifest_path.write_text(json.dumps({
        "schema_version": 1,
        "output_tag": args.anchor_train_tag,
        "pairing_anchor": args.anchor_train_tag,
        "selection_policy": "year-specific score thresholds matched at common weighted X efficiency",
        "samples": sample_metadata,
        "mass_range": list(mass_range),
        "bin_width": samples[0]["bin_width"],
        "reference_masses": samples[0]["reference_masses"],
        "thresholds": rows,
    }, indent=2))
    print(manifest_path)


if __name__ == "__main__":
    main()
