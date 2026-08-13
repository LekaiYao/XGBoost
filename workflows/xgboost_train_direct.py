import argparse
import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import uproot
from sklearn.metrics import auc, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from configs.samples import (
    bd_pbpb_precut_paths,
    infer_channel_from_tag,
    infer_dataset_token_from_tag,
    infer_dataset_year,
    infer_fid_profile,
    infer_reweight_profile,
    infer_sample_from_tag as infer_sample_from_config,
    infer_selection_profile,
    resolve_training_config,
    resolve_training_reweight_config,
    split_root_spec,
    supports_bd_pbpb_precut,
    to_root_spec,
)
from configs.direct_xgb_settings import DIRECT_XGB_PARAMS
from utils.paths import (
    condor_feature_importance_cumulative_path,
    condor_feature_importance_path,
    condor_model_config_path,
    condor_model_dir,
    condor_model_path,
    condor_scaler_path,
    condor_training_dir,
    condor_training_history_path,
    condor_training_ks_curve_path,
    condor_training_ks_path,
    condor_training_logloss_path,
    condor_training_score_path,
    ensure_dir,
)
from utils.run_metadata import save_run_metadata
from utils.selection import apply_selection, selection_columns
from utils.training_weights import (
    balanced_scale_pos_weight,
    resolve_training_weights,
    weighted_ks_curve,
)
from utils.varsets import get_varset_columns, infer_sample_from_tag, infer_varset_from_tag


def build_roc_payload(y_true, score, sample_weight=None):
    fpr, tpr, thresholds = roc_curve(y_true, score, sample_weight=sample_weight)
    roc_auc = auc(fpr, tpr)
    return {
        "auc": float(roc_auc),
        "threshold": [float(x) for x in thresholds],
        "tpr": [float(x) for x in tpr],
        "fpr": [float(x) for x in fpr],
        "background_rejection": [float(1.0 - x) for x in fpr],
        "signal_efficiency": [float(x) for x in tpr],
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Direct XGBoost training without Optuna")
    parser.add_argument("train_tag")
    parser.add_argument("--dataset-year", choices=["2023", "2024"], default=None)
    parser.add_argument("--selection-profile", default=None)
    parser.add_argument("--use-precut", type=int, choices=[0, 1], default=0)
    return parser.parse_args()


def save_feature_importance(model, feature_names, train_tag):
    importance_pairs = sorted(zip(feature_names, model.feature_importances_), key=lambda x: x[1], reverse=True)
    with open(condor_feature_importance_path(train_tag), "w") as f:
        json.dump(
            [{"rank": i, "feature": n, "importance": float(s)} for i, (n, s) in enumerate(importance_pairs, 1)],
            f,
            indent=2,
        )

    names = [n for n, _ in importance_pairs]
    scores = np.array([float(s) for _, s in importance_pairs])
    cum = np.cumsum(scores) / np.sum(scores) * 100.0
    plt.figure(figsize=(8, 5))
    plt.plot(names, cum, color="black", linewidth=1, alpha=0.6)
    plt.scatter(names, cum, color="tab:blue", s=45)
    plt.axhline(95.0, color="tab:red", linestyle="--", linewidth=1.5)
    plt.ylim(0, 105)
    plt.xticks(rotation=25, ha="right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(condor_feature_importance_cumulative_path(train_tag))
    plt.close()


def main():
    args = parse_args()
    train_tag = args.train_tag

    sample = infer_sample_from_config(train_tag)
    channel = infer_channel_from_tag(train_tag)
    dataset_token = infer_dataset_token_from_tag(train_tag)
    dataset_year = args.dataset_year or infer_dataset_year(train_tag, sample)
    selection_profile = args.selection_profile or infer_selection_profile(train_tag, sample)
    fid_profile = infer_fid_profile(train_tag, sample)
    reweight_profile = infer_reweight_profile(train_tag)
    train_cfg = resolve_training_config(sample, channel, dataset_year, selection_profile)
    reweight_cfg = resolve_training_reweight_config(
        sample, channel, dataset_year, reweight_profile, selection_profile, fid_profile
    )

    sample_key = infer_sample_from_tag(train_tag)
    feature_set_tag = infer_varset_from_tag(train_tag, sample=sample_key)
    if feature_set_tag is None:
        raise ValueError(f"Unable to infer feature set from train_tag: {train_tag}")
    input_columns = get_varset_columns(sample_key, feature_set_tag, channel=channel)
    trans_columns = [f"{col}_trans" for col in input_columns]

    sig_path = to_root_spec(train_cfg["signal"])
    bkg_path = to_root_spec(train_cfg["background"])
    signal_selection = train_cfg["signal_selection"]
    background_selection = train_cfg["background_selection"]
    dataset_source = train_cfg["dataset_source"]

    if reweight_cfg["signal"] is not None:
        sig_path = to_root_spec(reweight_cfg["signal"])
        dataset_source = f"{dataset_source}_signal_override"

    if args.use_precut:
        if not supports_bd_pbpb_precut(train_tag):
            raise ValueError(
                f"--use-precut only supports Bd_pb23/Bd_pb24 single-DAG tags, got '{train_tag}'."
            )
        precut_paths = bd_pbpb_precut_paths(train_tag)
        train_background_path = precut_paths["train_background"]
        if not train_background_path.exists():
            raise FileNotFoundError(f"Missing precut training background file: {train_background_path}")
        _, bkg_tree = split_root_spec(bkg_path)
        bkg_path = f"{train_background_path}:{bkg_tree}"
        dataset_source = f"{dataset_source}_precut_local"

    print(f"Train tag: {train_tag}")
    print(f"Dataset source: {dataset_source}")
    print(f"Selection profile: {selection_profile}")
    print(f"Reweight profile: {reweight_profile}")

    signal_read_columns = list(
        dict.fromkeys(
            input_columns
            + selection_columns(signal_selection)
            + ([reweight_cfg["weight_branch"]] if reweight_cfg["weight_branch"] else [])
        )
    )
    background_read_columns = list(
        dict.fromkeys(input_columns + selection_columns(background_selection))
    )
    print(f"Signal read columns ({len(signal_read_columns)}): {signal_read_columns}")
    print(f"Background read columns ({len(background_read_columns)}): {background_read_columns}")

    ak_sig = uproot.concatenate(
        sig_path, expressions=signal_read_columns, library="pd"
    )
    ak_bkg = uproot.concatenate(
        bkg_path, expressions=background_read_columns, library="pd"
    )

    ak_sig = apply_selection(ak_sig, signal_selection, "signal_selection")
    ak_bkg = apply_selection(ak_bkg, background_selection, "background_selection")
    is_weighted_training = reweight_cfg["weight_branch"] is not None
    if is_weighted_training:
        signal_weight = resolve_training_weights(
            ak_sig, reweight_cfg["weight_branch"], "signal sample"
        )
        background_weight = resolve_training_weights(
            ak_bkg, None, "background sample"
        )
    else:
        signal_weight = None
        background_weight = None

    ak_sig = ak_sig[input_columns].copy()
    ak_bkg = ak_bkg[input_columns].copy()
    ak_sig["is_sig"] = True
    ak_bkg["is_sig"] = False
    ak_sig["is_bkg"] = False
    ak_bkg["is_bkg"] = True
    if is_weighted_training:
        ak_sig["_training_weight"] = signal_weight
        ak_bkg["_training_weight"] = background_weight

    df_raw = pd.concat([ak_sig, ak_bkg], axis=0, ignore_index=True)
    X_raw = df_raw[input_columns]
    y = df_raw[["is_sig", "is_bkg"]]
    if is_weighted_training:
        X_train_raw, X_test_raw, y_train, y_test, weight_train, weight_test = train_test_split(
            X_raw, y, df_raw["_training_weight"], train_size=0.75, random_state=42
        )
    else:
        X_train_raw, X_test_raw, y_train, y_test = train_test_split(
            X_raw, y, train_size=0.75, random_state=42
        )
        weight_train = None
        weight_test = None
    scaler = StandardScaler()
    scaler.fit(
        X_train_raw,
        sample_weight=weight_train.to_numpy(dtype=float) if is_weighted_training else None,
    )
    X_train = pd.DataFrame(scaler.transform(X_train_raw), columns=trans_columns, index=X_train_raw.index)
    X_test = pd.DataFrame(scaler.transform(X_test_raw), columns=trans_columns, index=X_test_raw.index)

    n_sig_train = int(y_train["is_sig"].sum())
    n_bkg_train = int(y_train["is_bkg"].sum())
    if n_sig_train <= 0:
        raise ValueError("No signal events available in training split after selection.")
    if is_weighted_training:
        scale_pos_weight = balanced_scale_pos_weight(
            y_train["is_sig"].to_numpy(), weight_train.to_numpy()
        )
    else:
        scale_pos_weight = float(n_bkg_train / n_sig_train)

    params = {
        "booster": "gbtree",
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "random_state": 42,
        "n_jobs": 4,
        "scale_pos_weight": scale_pos_weight,
        **DIRECT_XGB_PARAMS[dataset_token][channel],
    }
    xgbc = XGBClassifier(**params)
    fit_kwargs = {
        "eval_set": [
            (X_train, y_train["is_sig"].to_numpy()),
            (X_test, y_test["is_sig"].to_numpy()),
        ],
        "verbose": False,
    }
    if is_weighted_training:
        fit_kwargs["sample_weight"] = weight_train.to_numpy()
        fit_kwargs["sample_weight_eval_set"] = [
            weight_train.to_numpy(),
            weight_test.to_numpy(),
        ]
    xgbc.fit(X_train, y_train["is_sig"].to_numpy(), **fit_kwargs)
    evals_result = xgbc.evals_result()

    train_score_xgb = xgbc.predict_proba(X_train)
    test_score_xgb = xgbc.predict_proba(X_test)
    train_score_xgb_all = train_score_xgb[:, 1]
    test_score_xgb_sig = test_score_xgb[y_test["is_sig"]][:, 1]
    test_score_xgb_bkg = test_score_xgb[y_test["is_bkg"]][:, 1]
    test_score_xgb_all = test_score_xgb[:, 1]
    train_y_true = y_train["is_sig"].astype(int).to_numpy()
    test_y_true = y_test["is_sig"].astype(int).to_numpy()
    train_roc = build_roc_payload(
        train_y_true,
        train_score_xgb_all,
        weight_train.to_numpy() if is_weighted_training else None,
    )
    test_roc = build_roc_payload(
        test_y_true,
        test_score_xgb_all,
        weight_test.to_numpy() if is_weighted_training else None,
    )
    efficiency_reference = train_cfg.get("efficiency_reference_signal")
    efficiency_reference_weight = train_cfg.get(
        "efficiency_reference_weight_branch"
    )
    if efficiency_reference is None and reweight_cfg["signal"] is not None:
        efficiency_reference = reweight_cfg["signal"]
        efficiency_reference_weight = reweight_cfg["weight_branch"]

    ensure_dir(condor_model_dir(train_tag))
    ensure_dir(condor_training_dir(train_tag))
    joblib.dump(xgbc, condor_model_path(train_tag))
    joblib.dump(scaler, condor_scaler_path(train_tag))

    with open(condor_model_config_path(train_tag), "w") as f:
        json.dump(
            {
                "input_columns": input_columns,
                "trans_columns": trans_columns,
                "model_params": params,
                "reweight_profile": reweight_profile,
                "signal_input_override": reweight_cfg["signal"] is not None,
                "signal_weight_branch": reweight_cfg["weight_branch"],
                "signal_path": sig_path,
                "split_policy": "candidate_level_random_split_preliminary",
                "split_random_state": 42,
                "train_fraction": 0.75,
                "scaler_fit_scope": "training_subset_only",
                "efficiency_reference_signal": (
                    to_root_spec(efficiency_reference)
                    if efficiency_reference is not None
                    else None
                ),
                "efficiency_reference_weight_branch": efficiency_reference_weight,
            },
            f,
            indent=2,
        )

    score_plot_path = condor_training_score_path(train_tag)
    plt.figure(figsize=(6, 6))
    test_is_sig = y_test["is_sig"].to_numpy(dtype=bool)
    test_weights = (
        weight_test.to_numpy(dtype=float) if is_weighted_training else None
    )
    plt.hist(
        test_score_xgb_sig,
        weights=test_weights[test_is_sig] if test_weights is not None else None,
        label="signal",
        histtype="step",
        bins=np.linspace(0, 1, 100),
        density=True,
    )
    plt.hist(
        test_score_xgb_bkg,
        weights=test_weights[~test_is_sig] if test_weights is not None else None,
        label="background",
        histtype="step",
        bins=np.linspace(0, 1, 100),
        density=True,
    )
    plt.xlabel("Score (Prob. from XGBoost Prediction)")
    plt.ylabel("(Bin Width)$^{-1}$")
    plt.legend()
    plt.xlim(0, 1)
    plt.savefig(score_plot_path)
    plt.close()

    roc_json_path = f"{condor_training_dir(train_tag)}/roc.json"
    with open(roc_json_path, "w") as f:
        json.dump(
            {
                "mode": "direct_xgboost",
                "train": train_roc,
                "test": test_roc,
            },
            f,
            indent=2,
        )

    roc_plot_path = f"{condor_training_dir(train_tag)}/roc.pdf"
    plt.figure(figsize=(6, 6))
    plt.plot(
        train_roc["signal_efficiency"],
        train_roc["background_rejection"],
        linewidth=2,
        label=f"Train AUC = {train_roc['auc']:.4f}",
    )
    plt.plot(
        test_roc["signal_efficiency"],
        test_roc["background_rejection"],
        linewidth=2,
        label=f"Test AUC = {test_roc['auc']:.4f}",
    )
    plt.xlabel("Signal efficiency")
    plt.ylabel("Background rejection")
    plt.title("ROC Curve")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(roc_plot_path)
    plt.close()

    save_feature_importance(xgbc, input_columns, train_tag)

    # ---- overtraining diagnostics: logloss history ----
    train_logloss = evals_result["validation_0"]["logloss"]
    test_logloss = evals_result["validation_1"]["logloss"]
    n_estimators_actual = len(train_logloss)

    # Apply CMS AN style for publication-quality figures
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

    logloss_plot_path = condor_training_logloss_path(train_tag)
    plt.figure(figsize=(7, 5))
    plt.plot(range(1, n_estimators_actual + 1), train_logloss, label="Train", color="tab:blue")
    plt.plot(range(1, n_estimators_actual + 1), test_logloss, label="Test", color="tab:orange")
    plt.xlabel("Boosting round")
    plt.ylabel("Log-loss")
    plt.grid(alpha=0.25, linestyle="--", linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(logloss_plot_path)
    plt.close()
    print(f"Logloss plot saved to: {logloss_plot_path}")

    training_history_path = condor_training_history_path(train_tag)
    with open(training_history_path, "w") as f:
        json.dump(
            {
                "train_logloss": [float(x) for x in train_logloss],
                "test_logloss": [float(x) for x in test_logloss],
                "best_iteration": int(getattr(xgbc, "best_iteration", n_estimators_actual)),
                "eval_metric": params.get("eval_metric", "logloss"),
            },
            f,
            indent=2,
        )
    print(f"Training history saved to: {training_history_path}")

    # ---- overtraining diagnostics: KS curves ----
    train_ks = weighted_ks_curve(
        y_train["is_sig"].to_numpy(),
        train_score_xgb_all,
        weight_train.to_numpy() if is_weighted_training else np.ones(len(y_train)),
    )
    test_ks = weighted_ks_curve(
        y_test["is_sig"].to_numpy(),
        test_score_xgb_all,
        weight_test.to_numpy() if is_weighted_training else np.ones(len(y_test)),
    )

    ks_plot_path = condor_training_ks_curve_path(train_tag)
    plt.figure(figsize=(7, 5))
    # Train set: blue family
    plt.plot(
        train_ks["score_thresholds"], train_ks["sig_cdf"],
        color="tab:blue", linestyle="-",
        label="Train signal",
    )
    plt.plot(
        train_ks["score_thresholds"], train_ks["bkg_cdf"],
        color="tab:blue", linestyle="--",
        label="Train background",
    )
    # Test set: orange family
    plt.plot(
        test_ks["score_thresholds"], test_ks["sig_cdf"],
        color="tab:orange", linestyle="-",
        label="Test signal",
    )
    plt.plot(
        test_ks["score_thresholds"], test_ks["bkg_cdf"],
        color="tab:orange", linestyle="--",
        label="Test background",
    )
    # Annotate KS values in a text box
    _ks_text = (
        f"Train KS = {train_ks['ks_stat']:.3f}\n"
        f"Test KS  = {test_ks['ks_stat']:.3f}"
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
    print(f"KS curve plot saved to: {ks_plot_path}")

    ks_json_path = condor_training_ks_path(train_tag)
    with open(ks_json_path, "w") as f:
        json.dump({"train": train_ks, "test": test_ks}, f, indent=2)
    print(f"KS data saved to: {ks_json_path}")

    # Restore original rcParams
    plt.rcParams.update(_saved_rc)

    save_run_metadata(
        train_tag=train_tag,
        training_script="workflows/xgboost_train_direct.py",
        signal_path=sig_path,
        background_path=bkg_path,
        signal_selection=train_cfg["signal_selection"],
        background_selection=train_cfg["background_selection"],
        input_columns=input_columns,
        trans_columns=trans_columns,
        pos_weight=scale_pos_weight,
        fixed_model_params=params,
        best_model_params=params,
        is_optuna=False,
        optimization_metric="direct training test AUC",
        best_objective_value=float(test_roc["auc"]),
        train_fraction=0.75,
        val_fraction=0.0,
        test_fraction=0.25,
        notes={
            "training_mode": "direct_xgboost",
            "dataset_source": dataset_source,
            "dataset_year": dataset_year,
            "selection_profile": selection_profile,
            "sample": sample,
            "channel": channel,
            "train_auc": float(train_roc["auc"]),
            "test_auc": float(test_roc["auc"]),
            "roc_json_path": roc_json_path,
            "roc_plot_path": roc_plot_path,
            "n_sig_train": n_sig_train,
            "n_bkg_train": n_bkg_train,
            "scale_pos_weight": scale_pos_weight,
            "reweight_profile": reweight_profile,
            "signal_input_override": reweight_cfg["signal"] is not None,
            "signal_weight_branch": reweight_cfg["weight_branch"],
            "signal_weight_sum": (
                float(signal_weight.sum())
                if is_weighted_training
                else float(len(ak_sig))
            ),
            "signal_weight_entries": (
                int(len(signal_weight))
                if is_weighted_training
                else int(len(ak_sig))
            ),
        },
    )

    print(f"Direct XGBoost training complete: {train_tag}, test AUC={test_roc['auc']:.4f}")


if __name__ == "__main__":
    main()
