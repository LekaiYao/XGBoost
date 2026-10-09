"""Run an isolated, snapshot-checked signed-sPlot RW variable comparison."""
import argparse
import csv
import hashlib
import json
import os
import platform
import shutil
import socket
import sys
from datetime import datetime
from importlib.metadata import version
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import uproot

from configs.samples import resolve_reweight_comparison_config
from utils.paths import reweighting_dir
from utils.selection import selection_columns
from utils.varsets import get_reweight_varset_columns
from workflows.reweighting.core import (
    load_tree_frame, select_frame, validate_columns, resolve_weights,
    train_folding_reweighter, predict_reweight, build_diagnostics, weight_summary,
    write_json,
)

STATUS = "preliminary descriptive training-sample closure; uncertainty not evaluated"


def fingerprint(path):
    path = Path(path)
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(f"Input changed during hashing: {path}")
    return {"path": str(path.resolve()), "bytes": after.st_size,
            "mtime_ns": after.st_mtime_ns, "sha256": digest.hexdigest()}


def grouped_cdf(values, weights):
    """Independent grouped-support CDF, including all weights at tied values."""
    order = np.argsort(values, kind="stable")
    values = np.asarray(values)[order]
    weights = np.asarray(weights, dtype=float)[order]
    points, first = np.unique(values, return_index=True)
    cumulative = np.r_[0., np.cumsum(np.add.reduceat(weights, first)) / weights.sum()]
    return points, cumulative


def independent_distance(x, y, wx, wy):
    px, fx = grouped_cdf(x, wx)
    py, fy = grouped_cdf(y, wy)
    support = np.union1d(px, py)
    return float(np.max(np.abs(fx[np.searchsorted(px, support, side="right")]
                               - fy[np.searchsorted(py, support, side="right")])))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study_tag")
    args = parser.parse_args()
    cfg = resolve_reweight_comparison_config(args.study_tag)
    root = Path(reweighting_dir(args.study_tag))
    root.mkdir(parents=True, exist_ok=True)
    with (root / "run_started.json").open("x") as stream:
        json.dump({"time": datetime.now().astimezone().isoformat(), "host": socket.gethostname(),
                   "pid": os.getpid(), "command": sys.argv}, stream, indent=2)
    provenance = root / "provenance"
    provenance.mkdir(exist_ok=True)
    resolved = lambda key: get_reweight_varset_columns(cfg["sample"], key, cfg["channel"])
    validation_variables = resolved(cfg["validation_variable_set"])
    variable_sets = {key: resolved(key) for key in cfg["variable_sets"]}
    manifest_fp = fingerprint(cfg["snapshot_manifest"])
    if manifest_fp["sha256"] != cfg["snapshot_manifest_sha256"]:
        raise RuntimeError("Frozen snapshot manifest changed")
    frozen = json.loads(Path(cfg["snapshot_manifest"]).read_text())
    source_files = frozen["files"]
    input_fingerprints = {"snapshot_manifest": manifest_fp}
    # All hashes are checked before event payload is loaded.
    for key in ("original", "source_data"):
        spec = cfg[key]
        fp = fingerprint(spec["path"])
        matches = [item for item in source_files
                   if Path(item["canonical_path"]).resolve() == Path(spec["path"]).resolve()]
        if len(matches) != 1 or fp["sha256"] != matches[0]["sha256_after"]:
            raise RuntimeError(f"Snapshot mismatch for {key}")
        expected = next(t for t in matches[0]["root"]["trees"] if t["name"] == spec["tree"])
        with uproot.open(spec["path"]) as source:
            tree = source[spec["tree"]]
            if tree.num_entries != expected["entries"]:
                raise RuntimeError(f"Entry mismatch for {key}")
            expected_types = {b["name"]: b["type"] for b in expected["branches"]}
            if tree.typenames() != expected_types:
                raise RuntimeError(f"Schema mismatch for {key}")
        input_fingerprints[key] = fp
        print(f"Snapshot PASS: {key}", flush=True)
    target_path = Path(cfg["target"]["path"])
    target_meta_path = target_path.with_suffix(".json")
    target_meta = json.loads(target_meta_path.read_text())
    delivery = target_path.parent.parent
    quality = json.loads((delivery / "validation.json").read_text())
    fit_summary = json.loads((delivery / "summary.json").read_text())
    if target_meta["status"] != "PASS" or quality["status"] != "PASS":
        raise RuntimeError("sPlot delivery did not pass its producer validation")
    for key, path in (("target", target_path), ("target_metadata", target_meta_path),
                      ("target_validation", delivery / "validation.json"),
                      ("target_summary", delivery / "summary.json"),
                      ("parameter_reference", cfg["parameter_reference"])):
        input_fingerprints[key] = fingerprint(path)
    parameters = json.loads(Path(cfg["parameter_reference"]).read_text())["parameters"]
    columns = list(dict.fromkeys(validation_variables + selection_columns(cfg["selection"])
                                  + selection_columns(cfg["target_source_selection"])))
    target_full = load_tree_frame(str(target_path), cfg["target"]["tree"],
                                  columns + [cfg["target_weight_branch"]])
    source_data = load_tree_frame(cfg["source_data"]["path"], cfg["source_data"]["tree"], columns)
    source_selected = select_frame(source_data, cfg["target_source_selection"], "sPlot source")
    if len(source_selected) != len(target_full):
        raise RuntimeError("sPlot source/cache length mismatch")
    for col in columns:
        if not np.array_equal(source_selected[col].to_numpy(), target_full[col].to_numpy()):
            raise RuntimeError(f"sPlot source/cache entry alignment mismatch: {col}")
    full_weights = resolve_weights(target_full, cfg["target_weight_branch"], "full target")
    expected_stats = target_meta["weight_statistics"]
    if len(full_weights) != expected_stats["entries"]:
        raise RuntimeError("sPlot metadata entries mismatch")
    np.testing.assert_allclose(full_weights.sum(), expected_stats["sumw"], rtol=1e-12)
    np.testing.assert_allclose(np.square(full_weights).sum(), expected_stats["sumw2"], rtol=1e-12)
    if np.count_nonzero(full_weights < 0) != expected_stats["negative_weights"]:
        raise RuntimeError("sPlot negative-weight count mismatch")
    original_full = load_tree_frame(cfg["original"]["path"], cfg["original"]["tree"], columns)
    original = select_frame(original_full, cfg["selection"], "RW MC")
    target = select_frame(target_full, cfg["selection"], "RW sPlot")
    for name, frame in (("MC", original), ("sPlot", target)):
        validate_columns(frame, columns, name)
    unit = resolve_weights(original, None, "MC")
    signed = resolve_weights(target, cfg["target_weight_branch"], "sPlot")
    print(f"Selected MC={len(original)}, sPlot={len(target)}, target Neff={weight_summary(signed)['effective_sample_size']:.3f}", flush=True)
    del source_data, source_selected, original_full, target_full
    preflight = {"status": "PASS", "config": cfg, "variable_sets": variable_sets,
                 "validation_variables": validation_variables, "parameters": parameters,
                 "input_fingerprints": input_fingerprints,
                 "target_entrywise_raw_alignment_columns": columns,
                 "weights": {"mc": weight_summary(unit), "target": weight_summary(signed)},
                 "target_fit_caveats": fit_summary["fit_quality_comparison"]["new_fit_caveats"],
                 "software": {name: version(name) for name in ("hep_ml", "numpy", "pandas", "uproot", "scikit-learn")},
                 "python": platform.python_version(), "bootstrap_replicas": 0,
                 "evaluation": "same selected samples used for training; mean vote over all five folds; not independent holdout"}
    write_json(provenance / "preflight.json", preflight)
    for source in (Path(__file__), Path("configs/samples.py"), Path("utils/varsets.py"),
                   Path("workflows/reweighting/core.py")):
        destination = provenance / "source_used" / source.name
        destination.parent.mkdir(exist_ok=True)
        shutil.copy2(source, destination)
    np.savez_compressed(root / "selected_inputs.npz", columns=np.asarray(columns),
                        mc=original[columns].to_numpy(), data=target[columns].to_numpy(),
                        mc_source_entry=original.index.to_numpy(),
                        target_cache_entry=target.index.to_numpy(), signal_sWeight=signed)
    results, rows, weights = {}, [], {}
    for varset, variables in variable_sets.items():
        print(f"Training {varset}: {variables}", flush=True)
        output = root / varset
        output.mkdir(exist_ok=False)
        model, corrected = train_folding_reweighter(original, target, variables, unit, signed, **parameters)
        if not np.isfinite(corrected).all() or np.any(corrected <= 0):
            raise RuntimeError(f"Invalid predicted weights: {varset}")
        joblib.dump(model, output / "reweighter.pkl")
        loaded = joblib.load(output / "reweighter.pkl")
        np.testing.assert_allclose(predict_reweight(loaded, original, variables, unit), corrected, rtol=1e-12)
        np.save(output / "mc_weights.npy", corrected)
        diagnostics = build_diagnostics(original, target, validation_variables, unit, corrected, signed)
        for variable in validation_variables:
            record = diagnostics["variables"][variable]
            for weight, key in ((unit, "cdf_distance_before"), (corrected, "cdf_distance_after")):
                check = independent_distance(original[variable].to_numpy(), target[variable].to_numpy(), weight, signed)
                np.testing.assert_allclose(check, record[key], rtol=0, atol=1e-10)
            rows.append({"variable_set": varset, "variable": variable,
                         "in_reweight_inputs": variable in variables,
                         "before": record["cdf_distance_before"], "after": record["cdf_distance_after"],
                         "delta_after_minus_before": -record["improvement"]})
        results[varset] = diagnostics
        weights[varset] = corrected
        write_json(output / "reweighting_manifest.json", {
            "study_tag": args.study_tag, "variable_set": varset, "variables": variables,
            "selection": cfg["selection"], "parameters": parameters,
            "inputs": input_fingerprints, "status": STATUS, "bootstrap_replicas": 0,
            "prediction": preflight["evaluation"], "artifacts": {"model": "reweighter.pkl", "weights": "mc_weights.npy"}})
        write_json(output / "diagnostics.json", diagnostics)
        print(f"Completed {varset}: {json.dumps(diagnostics['variables'])}", flush=True)
    with (root / "cdf_comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    figure, axes = plt.subplots(4, 2, figsize=(13, 15))
    for ax, variable in zip(axes.flat, validation_variables):
        curves = [(target[variable].to_numpy(), signed, "X DATA signed sPlot", "black"),
                  (original[variable].to_numpy(), unit, "MC before", "grey")]
        curves += [(original[variable].to_numpy(), weights[k], k, None) for k in variable_sets]
        for values, weight, label, color in curves:
            points, cumulative = grouped_cdf(values, weight)
            ax.step(points, cumulative[1:], where="post", label=label, color=color, linewidth=1.1, rasterized=True)
        ax.set_title(variable); ax.set_xlabel(variable); ax.set_ylabel("Normalized signed empirical CDF")
        ax.grid(alpha=.2)
    axes.flat[0].legend(fontsize=8)
    figure.suptitle("ppRef X snapshot v1: common dR <= 0.45 support\nTraining-sample closure; point estimates only", fontsize=13)
    figure.tight_layout(rect=(0, 0, 1, .96))
    figure.savefig(root / "cdf_overlays.pdf")
    figure.savefig(root / "cdf_overlays.png", dpi=140)
    plt.close(figure)
    postflight = {}
    for key, old in input_fingerprints.items():
        new = fingerprint(old["path"])
        postflight[key] = new == old
        if new != old:
            raise RuntimeError(f"Input changed during comparison: {key}")
    summary = {"status": STATUS, "study_tag": args.study_tag, "config": cfg,
               "metric": "maximum signed-weighted empirical-CDF distance over union of observed finite support",
               "metric_is_standard_ks_pvalue": False, "bootstrap_replicas": 0,
               "evaluation": preflight["evaluation"], "results": results,
               "target_fit_caveats": preflight["target_fit_caveats"]}
    write_json(root / "summary.json", summary)
    write_json(root / "validation.json", {"status": "PASS", "model_count": len(results),
               "metric_rows": len(rows), "source_target_raw_alignment": True,
               "independent_grouped_cdf_check": True, "model_reload_prediction_check": True,
               "input_postflight": postflight})
    print(f"PASS: {root / 'cdf_comparison.csv'}", flush=True)


if __name__ == "__main__":
    main()
