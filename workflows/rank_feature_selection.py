"""Rank baseline and candidates by pooled, sample-weighted train+test AUC."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import uproot
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from configs.feature_selection import resolve_feature_selection_experiment
from configs import samples as samples
from utils.varsets import get_varset_columns, infer_varset_from_tag
from utils.selection import apply_selection, selection_columns
from utils.training_weights import resolve_training_weights
from utils.paths import (model_dir, model_path, scaler_path, model_config_path,
    training_dir, selected_dir, experiment_output_dir)


def fingerprint(path):
    path = Path(path); before = path.stat(); digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(f"Input changed: {path}")
    return dict(path=str(path.resolve()), size=after.st_size,
                mtime_ns=after.st_mtime_ns, sha256=digest.hexdigest())


def ordered_ranking(rows, baseline):
    # Exact ties retain the baseline; no train/test-gap penalty is applied.
    return sorted(rows, key=lambda row: (-row["combined_auc"],
                  row["train_tag"] != baseline, row["train_tag"]))


def rank_experiment(experiment):
    cfg = resolve_feature_selection_experiment(experiment)
    baseline = cfg["baseline"]
    items = [{"train_tag": baseline, "removed_variable": ""}] + cfg["candidates"]
    models, metadata, protected = {}, {}, {}
    def protect(path):
        path = Path(path); protected[str(path)] = fingerprint(path)
    for item in items:
        tag = item["train_tag"]
        paths = [Path(model_config_path(tag)), Path(model_dir(tag))/"run_metadata.json",
                 Path(model_path(tag)), Path(scaler_path(tag)), Path(training_dir(tag))/"roc.json"]
        for path in paths: protect(path)
        models[tag] = json.loads(paths[0].read_text())
        metadata[tag] = json.loads(paths[1].read_text())
    base, bm = models[baseline], metadata[baseline]
    sample = samples.infer_sample_from_tag(baseline)
    channel = samples.infer_channel_from_tag(baseline)
    year = samples.infer_dataset_year(baseline, sample)
    train = samples.resolve_training_config(sample, channel, year,
                samples.infer_selection_profile(baseline, sample))
    assert samples.to_root_spec(train["signal"]) == bm["signal_path"]
    assert samples.to_root_spec(train["background"]) == bm["background_path"]
    assert train["signal_selection"] == bm["signal_selection"]
    assert train["background_selection"] == bm["background_selection"]
    all_features = sorted({f for c in models.values() for f in c["input_columns"]})
    weight_columns = base["signal_weight_branches"]
    sig_path = Path(selected_dir(baseline))/"MC_with_score.root"
    bkg_path = Path(selected_dir(baseline))/"DATA_with_score.root"
    for path in [sig_path, bkg_path]: protect(path)
    apply_info = json.loads((Path(selected_dir(baseline))/"batch_apply_summary.json").read_text())
    assert apply_info["apply_selection"]["expression"] == bm["signal_selection"]
    frames = []
    for path, kind in [(sig_path, "signal"), (bkg_path, "background")]:
        columns = list(dict.fromkeys(all_features + selection_columns(bm[kind+"_selection"])
                       + (weight_columns if kind == "signal" else [])))
        tree = samples.split_root_spec(bm[kind+"_path"])[1]
        with uproot.open(path) as source:
            frame = source[tree].arrays(columns, library="pd")
        frames.append(apply_selection(frame, bm[kind+"_selection"], kind))
    signal, background = frames
    features = pd.concat([signal[all_features], background[all_features]], ignore_index=True)
    labels = np.r_[np.ones(len(signal), dtype=bool), np.zeros(len(background), dtype=bool)]
    weights = np.r_[resolve_training_weights(signal, weight_columns, "signal"), np.ones(len(background))]
    indices = train_test_split(np.arange(len(labels)), train_size=base["train_fraction"],
                               random_state=base["split_random_state"])
    results = []
    for item in items:
        tag = item["train_tag"]; mc = models[tag]; meta = metadata[tag]
        for key in ["signal_path", "background_path", "signal_selection", "background_selection"]:
            assert meta[key] == bm[key], (tag, key)
        for key in ["split_random_state", "train_fraction", "signal_weight_branches", "model_params"]:
            assert mc[key] == base[key], (tag, key)
        assert mc["input_columns"] == get_varset_columns(sample, infer_varset_from_tag(tag), channel)
        if tag != baseline and cfg["operation"] == "drop1":
            assert mc["input_columns"] == [f for f in base["input_columns"] if f != item["removed_variable"]]
        model = joblib.load(model_path(tag)); scaler = joblib.load(scaler_path(tag))
        transformed = pd.DataFrame(scaler.transform(features[mc["input_columns"]]), columns=mc["trans_columns"])
        scores = model.predict_proba(transformed)[:, 1]
        assert np.isfinite(scores).all()
        recorded = json.loads((Path(training_dir(tag))/"roc.json").read_text())
        row = dict(train_tag=tag, removed_variable=item.get("removed_variable", ""),
                   feature_count=len(mc["input_columns"]),
                   combined_auc=float(roc_auc_score(labels, scores, sample_weight=weights)))
        for split, idx in zip(["train", "test"], indices):
            value = float(roc_auc_score(labels[idx], scores[idx], sample_weight=weights[idx]))
            np.testing.assert_allclose(value, recorded[split]["auc"], rtol=0, atol=1e-12)
            row[split+"_auc"] = value
        if "combined" in recorded:
            np.testing.assert_allclose(row["combined_auc"], recorded["combined"]["auc"], rtol=0, atol=1e-12)
        results.append(row)
    ranking = ordered_ranking(results, baseline)
    base_auc = next(row["combined_auc"] for row in results if row["train_tag"] == baseline)
    for rank, row in enumerate(ranking, 1):
        row["rank"] = rank; row["delta_auc_vs_baseline"] = row["combined_auc"]-base_auc
    winner = ranking[0]["train_tag"]
    best_candidate = next(row["train_tag"] for row in ranking if row["train_tag"] != baseline)
    review = {}
    for tag in dict.fromkeys([baseline, winner, best_candidate]):
        directory = Path(selected_dir(tag))/"cut_scan"
        protect(directory/"thresholds.json")
        plots = sorted(directory.glob("DATA_*_eff*.pdf"))
        assert plots, f"No mass review plots for {tag}"
        review[tag] = [str(path.resolve()) for path in plots]
    for path, original in protected.items():
        assert fingerprint(path) == original, path
    out = Path(experiment_output_dir("training", experiment)); out.mkdir(parents=True, exist_ok=True)
    payload = dict(experiment=experiment, baseline=baseline, metric="combined_weighted_auc",
        definition="AUC recomputed on pooled train+test scores; original MC weights, unit sideband weights",
        signal_entries=len(signal), background_entries=len(background), ranking=ranking,
        machine_recommendation=winner, best_candidate=best_candidate, decision_status="AWAITING_HUMAN_MASS_REVIEW",
        baseline_changed=False, validation="PASS: all recorded train/test AUCs reproduced; inputs unchanged",
        mass_review_plots=review, input_fingerprints=protected)
    (out/"ranking.json").write_text(json.dumps(payload, indent=2)+"\n")
    with (out/"ranking.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(ranking[0])); writer.writeheader(); writer.writerows(ranking)
    lines = [f"# {experiment}", "", "Status: AWAITING_HUMAN_MASS_REVIEW", "",
        f"Baseline: `{baseline}`", f"Machine recommendation: `{winner}`", "",
        "Metric: pooled train+test weighted AUC; baseline has not been changed.", "",
        "|Rank|Tag|Removed variable|Combined AUC|Delta|", "|---|---|---|---|---|"]
    lines += [f"|{r['rank']}|{r['train_tag']}|{r['removed_variable']}|{r['combined_auc']:.8f}|{r['delta_auc_vs_baseline']:+.8f}|" for r in ranking]
    for tag, plots in review.items():
        lines += ["", f"Mass review: {tag}"] + [f"- [{Path(p).name}]({p})" for p in plots]
    (out/"ranking.md").write_text("\n".join(lines)+"\n")
    print(json.dumps({k: payload[k] for k in ["machine_recommendation", "decision_status", "ranking"]}, indent=2))
    from utils.feature_selection_report import update_feature_selection_report
    update_feature_selection_report(experiment)
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment")
    rank_experiment(parser.parse_args().experiment)
