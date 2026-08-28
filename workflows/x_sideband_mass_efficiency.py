"""Compare DATA score-cut efficiency versus mass for X sideband trainings."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import uproot

from configs.samples import (
    infer_channel_from_tag,
    infer_dataset_year,
    infer_fid_profile,
    infer_sample_from_tag,
    resolve_draw_config,
    resolve_fiducial_config,
)
from utils.paths import comparison_dir, cut_scan_dir, data_output_path
from utils.selection import apply_selection, selection_columns


COMPARISON_TAG = "x_sideband_mass_efficiency_v1"
GROUPS = {
    "pb23": {
        "baseline [3.82,3.85] U [3.89,3.92]": "X_pb23_v3_fid3_6v5_rwr6range5v1_xgb_v1",
        "expanded [3.80,3.85] U [3.89,3.94]": "X_pb23_v5_fid3_6v5_rwr6range5v1_xgb_v1",
        "broad [3.75,3.85] U [3.89,3.99]": "X_pb23_v6_fid3_6v5_rwr6range5v1_xgb_v1",
    },
    "pb24": {
        "baseline [3.82,3.85] U [3.89,3.92]": "X_pb24_v19_fid19_6v5_rwr6range5v1_xgb_v1",
        "expanded [3.80,3.85] U [3.89,3.94]": "X_pb24_v21_fid19_6v5_rwr6range5v1_xgb_v1",
        "broad [3.75,3.85] U [3.89,3.99]": "X_pb24_v22_fid19_6v5_rwr6range5v1_xgb_v1",
    },
}
MASS_RANGE = (3.75, 3.99)
BIN_WIDTH = 0.005
PLOT_TARGETS = (0.20, 0.30)


def threshold_map(payload: dict) -> dict[float, dict]:
    rows = payload.get("thresholds", [])
    mapped = {float(row["target_efficiency"]): row for row in rows}
    if len(mapped) != len(rows):
        raise ValueError("Duplicate target efficiencies in thresholds payload")
    return mapped


def binomial_efficiency(after: np.ndarray, before: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    after = np.asarray(after, dtype=float)
    before = np.asarray(before, dtype=float)
    efficiency = np.divide(after, before, out=np.full_like(after, np.nan), where=before > 0)
    variance = np.divide(
        efficiency * (1.0 - efficiency),
        before,
        out=np.full_like(after, np.nan),
        where=before > 0,
    )
    return efficiency, np.sqrt(np.clip(variance, 0.0, None))


def _load_tag(tag: str) -> dict:
    sample = infer_sample_from_tag(tag)
    channel = infer_channel_from_tag(tag)
    year = infer_dataset_year(tag, sample)
    fid_profile = infer_fid_profile(tag, sample)
    fiducial = resolve_fiducial_config(sample, channel, fid_profile)["expression"]
    tree_name = resolve_draw_config(sample, channel, year)["data"]["tree"]
    data_path = Path(data_output_path(tag))
    threshold_path = Path(cut_scan_dir(tag)) / "weighted_signal_efficiency" / "thresholds.json"
    if not data_path.is_file():
        raise FileNotFoundError(f"Missing scored DATA: {data_path}")
    if not threshold_path.is_file():
        raise FileNotFoundError(f"Missing weighted-efficiency thresholds: {threshold_path}")
    threshold_payload = json.loads(threshold_path.read_text())
    if threshold_payload.get("train_tag") != tag:
        raise ValueError(f"Threshold train_tag mismatch in {threshold_path}")
    if threshold_payload.get("weight_branch") != "Reweight":
        raise ValueError(f"Thresholds for {tag} are not Reweight-based")
    return {
        "tag": tag,
        "sample": sample,
        "channel": channel,
        "year": year,
        "fid_profile": fid_profile,
        "fiducial_selection": fiducial,
        "tree": tree_name,
        "data_path": str(data_path),
        "threshold_path": str(threshold_path),
        "thresholds": threshold_map(threshold_payload),
    }


def _score_branch(tree, tag: str) -> str:
    tagged = f"Prediction_{tag}"
    if tagged in tree:
        return tagged
    if "Prediction" in tree:
        return "Prediction"
    raise KeyError(f"Neither '{tagged}' nor 'Prediction' exists in {tree.object_path}")


def _analyze_year(dataset: str, labeled_tags: dict[str, str], bins: np.ndarray, chunk_size: int) -> dict:
    specs = {label: _load_tag(tag) for label, tag in labeled_tags.items()}
    reference = next(iter(specs.values()))
    for spec in specs.values():
        for key in ("sample", "channel", "year", "fid_profile", "fiducial_selection", "tree"):
            if spec[key] != reference[key]:
                raise ValueError(f"{dataset} tags do not share {key}: {spec[key]} != {reference[key]}")

    target_sets = [set(spec["thresholds"]) for spec in specs.values()]
    if any(targets != target_sets[0] for targets in target_sets[1:]):
        raise ValueError(f"Weighted-efficiency targets differ within {dataset}")
    targets = sorted(target_sets[0], reverse=True)
    counts_before = np.zeros(len(bins) - 1, dtype=np.int64)
    counts_after = {
        label: {target: np.zeros(len(bins) - 1, dtype=np.int64) for target in targets}
        for label in specs
    }

    files = {label: uproot.open(spec["data_path"]) for label, spec in specs.items()}
    try:
        trees = {label: files[label][specs[label]["tree"]] for label in specs}
        entry_counts = {label: int(tree.num_entries) for label, tree in trees.items()}
        if len(set(entry_counts.values())) != 1:
            raise ValueError(f"Scored DATA entry counts differ within {dataset}: {entry_counts}")
        score_branches = {label: _score_branch(tree, specs[label]["tag"]) for label, tree in trees.items()}
        common_columns = list(dict.fromkeys(["Bmass", *selection_columns(reference["fiducial_selection"])]))
        total_entries = next(iter(entry_counts.values()))
        for start in range(0, total_entries, chunk_size):
            stop = min(start + chunk_size, total_entries)
            arrays = next(iter(trees.values())).arrays(
                common_columns, entry_start=start, entry_stop=stop, library="np"
            )
            frame = pd.DataFrame(arrays)
            selected = apply_selection(frame, reference["fiducial_selection"], f"{dataset} fiducial selection")
            fid_mask = frame.index.isin(selected.index)
            mass = frame["Bmass"].to_numpy(dtype=float)
            mass_mask = fid_mask & (mass >= bins[0]) & (mass < bins[-1]) & np.isfinite(mass)
            counts_before += np.histogram(mass[mass_mask], bins=bins)[0]

            for label, tree in trees.items():
                score_data = tree.arrays(
                    ["Bmass", score_branches[label]], entry_start=start, entry_stop=stop, library="np"
                )
                candidate_mass = np.asarray(score_data["Bmass"], dtype=float)
                if not np.array_equal(candidate_mass, mass, equal_nan=True):
                    raise ValueError(f"Candidate ordering/Bmass mismatch for {specs[label]['tag']} at {start}:{stop}")
                score = np.asarray(score_data[score_branches[label]], dtype=float)
                for target in targets:
                    threshold = float(specs[label]["thresholds"][target]["score_threshold"])
                    accepted = mass_mask & np.isfinite(score) & (score > threshold)
                    counts_after[label][target] += np.histogram(mass[accepted], bins=bins)[0]
    finally:
        for root_file in files.values():
            root_file.close()

    rows = {}
    for label, spec in specs.items():
        rows[label] = {
            "tag": spec["tag"],
            "data_path": spec["data_path"],
            "tree": spec["tree"],
            "score_branch": score_branches[label],
            "threshold_path": spec["threshold_path"],
            "thresholds": {},
        }
        for target in targets:
            after = counts_after[label][target]
            efficiency, uncertainty = binomial_efficiency(after, counts_before)
            threshold_row = spec["thresholds"][target]
            rows[label]["thresholds"][f"{target:.2f}"] = {
                "score_threshold": float(threshold_row["score_threshold"]),
                "achieved_weighted_signal_efficiency": float(threshold_row["achieved_efficiency"]),
                "counts_after": after.tolist(),
                "data_efficiency": [None if math.isnan(x) else float(x) for x in efficiency],
                "binomial_uncertainty": [None if math.isnan(x) else float(x) for x in uncertainty],
            }
    return {
        "dataset": dataset,
        "fid_profile": reference["fid_profile"],
        "fiducial_selection": reference["fiducial_selection"],
        "input_entries": entry_counts,
        "counts_before_score_cut": counts_before.tolist(),
        "models": rows,
    }


def _plot_year_target(output_path: Path, year_result: dict, bins: np.ndarray, target: float) -> None:
    centers = 0.5 * (bins[:-1] + bins[1:])
    before = np.asarray(year_result["counts_before_score_cut"], dtype=float)
    fig, axes = plt.subplots(3, 1, figsize=(8.2, 9.5), sharex=True, gridspec_kw={"height_ratios": [1, 1, 1.15]})
    axes[0].stairs(before, bins, linewidth=1.6, color="black")
    axes[0].set_ylabel("DATA before score cut")
    axes[0].set_title(f"{year_result['dataset']}: mass response at weighted signal efficiency {target:.0%}")
    for label, model in year_result["models"].items():
        row = model["thresholds"][f"{target:.2f}"]
        after = np.asarray(row["counts_after"], dtype=float)
        efficiency = np.asarray([np.nan if x is None else x for x in row["data_efficiency"]])
        uncertainty = np.asarray([np.nan if x is None else x for x in row["binomial_uncertainty"]])
        axes[1].stairs(after, bins, linewidth=1.4, label=label)
        axes[2].errorbar(centers, efficiency, yerr=uncertainty, marker="o", markersize=2.4, linewidth=1.0, capsize=1.5, label=label)
    axes[1].set_ylabel("DATA after score cut")
    axes[1].legend(fontsize=7, frameon=False)
    axes[2].set_ylabel("after / before")
    axes[2].set_xlabel("Bmass (GeV)")
    axes[2].set_ylim(bottom=0)
    for axis in axes:
        axis.grid(alpha=0.2)
        for mass in (3.80, 3.85, 3.872, 3.89):
            axis.axvline(mass, color="gray", linestyle="--", linewidth=0.7, alpha=0.55)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison-tag", default=COMPARISON_TAG)
    parser.add_argument("--chunk-size", type=int, default=500_000)
    args = parser.parse_args()
    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be positive")

    output_dir = Path(comparison_dir(args.comparison_tag))
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing diagnostic: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{args.comparison_tag}.", dir=output_dir.parent))
    try:
        n_bins = int(round((MASS_RANGE[1] - MASS_RANGE[0]) / BIN_WIDTH))
        bins = np.linspace(MASS_RANGE[0], MASS_RANGE[1], n_bins + 1)
        results = {dataset: _analyze_year(dataset, tags, bins, args.chunk_size) for dataset, tags in GROUPS.items()}
        payload = {
            "comparison_tag": args.comparison_tag,
            "definition": "N(DATA passing matched score threshold, mass bin) / N(DATA with no score cut, mass bin)",
            "interpretation_note": "The denominator is the fiducial scored DATA sample with no requirement on Prediction; signal peaks are not subtracted.",
            "mass_range": list(MASS_RANGE),
            "bin_width": BIN_WIDTH,
            "bin_edges": bins.tolist(),
            "plot_targets": list(PLOT_TARGETS),
            "results": results,
        }
        (staging / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")
        manifest = {
            "comparison_tag": args.comparison_tag,
            "workflow": "workflows.x_sideband_mass_efficiency",
            "groups": GROUPS,
            "summary": "summary.json",
            "plots": [],
        }
        for dataset, result in results.items():
            for target in PLOT_TARGETS:
                plot_name = f"{dataset}_eff{int(round(100 * target)):02d}.pdf"
                _plot_year_target(staging / plot_name, result, bins, target)
                manifest["plots"].append(plot_name)
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        os.replace(staging, output_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(f"Wrote sideband mass-efficiency diagnostic: {output_dir}")


if __name__ == "__main__":
    main()
