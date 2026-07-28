import os
import re

OUTPUT_ROOT_DIR = "output"
MODELS_DIR = os.path.join(OUTPUT_ROOT_DIR, "models")
TRAINING_DIR = os.path.join(OUTPUT_ROOT_DIR, "training")
SHAP_DIR = os.path.join(OUTPUT_ROOT_DIR, "shap")
SELECTED_DIR = os.path.join(OUTPUT_ROOT_DIR, "selected")
REWEIGHTING_DIR = os.path.join(OUTPUT_ROOT_DIR, "reweighting")


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def resolve_existing(*candidates):
    for path in candidates:
        if path and os.path.exists(path):
            return path
    return candidates[0] if candidates else None


def split_version_suffix(train_tag):
    match = re.match(r"(.+)_((?:\d+)?v)(\d+)$", train_tag)
    if not match:
        return None
    base, token, version_text = match.groups()
    return base, token, int(version_text)


def group_base_tag(train_tag):
    parsed = split_version_suffix(train_tag)
    if not parsed:
        return train_tag
    base, token, _ = parsed
    if token == "v":
        return base
    return f"{base}_{token}"


def train_group_tag(train_tags):
    if not train_tags:
        raise ValueError("No train tags provided")
    base_tags = [group_base_tag(tag) for tag in train_tags]
    if len(set(base_tags)) != 1:
        raise ValueError(f"Train tags do not belong to a single group: {train_tags}")
    return base_tags[0] if len(train_tags) > 1 else train_tags[0]


def train_batch_tag(train_tag):
    parsed = split_version_suffix(train_tag)
    if not parsed:
        return train_tag
    base_tag, token, version = parsed
    start = ((version - 1) // 10) * 10 + 1
    end = start + 9
    if token == "v":
        return f"{base_tag}_v{start}_v{end}"
    return f"{base_tag}_{token}{start}_{token}{end}"


def model_dir(train_tag):
    return os.path.join(MODELS_DIR, train_tag)


def training_dir(train_tag):
    return os.path.join(TRAINING_DIR, train_tag)


def condor_model_dir(train_tag):
    return os.path.join(MODELS_DIR, train_tag)


def condor_training_dir(train_tag):
    return os.path.join(TRAINING_DIR, train_tag)


def shap_dir(train_tag):
    return os.path.join(SHAP_DIR, train_tag)


def selected_dir(train_tag):
    return os.path.join(SELECTED_DIR, train_tag)


def reweighting_dir(reweight_tag):
    return os.path.join(REWEIGHTING_DIR, reweight_tag)


def reweighter_model_path(reweight_tag):
    return os.path.join(reweighting_dir(reweight_tag), "reweighter.pkl")


def reweighting_manifest_path(reweight_tag):
    return os.path.join(reweighting_dir(reweight_tag), "reweighting_manifest.json")


def reweighting_diagnostics_path(reweight_tag):
    return os.path.join(reweighting_dir(reweight_tag), "diagnostics.json")


def reweighting_domain_closure_path(reweight_tag):
    return os.path.join(reweighting_dir(reweight_tag), "domain_classifier_holdout.json")


def reweighted_root_path(reweight_tag, input_path):
    stem = os.path.splitext(os.path.basename(str(input_path)))[0]
    return os.path.join(reweighting_dir(reweight_tag), f"{stem}_with_reweight.root")


def cut_scan_dir(train_tag):
    return os.path.join(selected_dir(train_tag), "cut_scan")


def model_path(train_tag):
    return os.path.join(model_dir(train_tag), "xgb_model.pkl")


def scaler_path(train_tag):
    return os.path.join(model_dir(train_tag), "scaler.pkl")


def model_config_path(train_tag):
    return os.path.join(model_dir(train_tag), "model_config.json")


def training_score_path(train_tag):
    return os.path.join(training_dir(train_tag), "xgb_score.pdf")


def condor_model_path(train_tag):
    return os.path.join(condor_model_dir(train_tag), "xgb_model.pkl")


def condor_scaler_path(train_tag):
    return os.path.join(condor_model_dir(train_tag), "scaler.pkl")


def condor_model_config_path(train_tag):
    return os.path.join(condor_model_dir(train_tag), "model_config.json")


def condor_training_score_path(train_tag):
    return os.path.join(condor_training_dir(train_tag), "xgb_score.pdf")


def feature_importance_path(train_tag):
    return os.path.join(training_dir(train_tag), "feature_importance.json")


def feature_importance_cumulative_path(train_tag):
    return os.path.join(training_dir(train_tag), "feature_importance_cumulative.pdf")


def condor_feature_importance_path(train_tag):
    return os.path.join(condor_training_dir(train_tag), "feature_importance.json")


def condor_feature_importance_cumulative_path(train_tag):
    return os.path.join(condor_training_dir(train_tag), "feature_importance_cumulative.pdf")


def condor_training_logloss_path(train_tag):
    return os.path.join(condor_training_dir(train_tag), "logloss.pdf")


def condor_training_history_path(train_tag):
    return os.path.join(condor_training_dir(train_tag), "training_history.json")


def condor_training_ks_curve_path(train_tag):
    return os.path.join(condor_training_dir(train_tag), "ks_curve.pdf")


def condor_training_ks_path(train_tag):
    return os.path.join(condor_training_dir(train_tag), "ks.json")


def shap_importance_path(train_tag):
    return os.path.join(shap_dir(train_tag), "shap_importance.json")


def shap_importance_fraction_path(train_tag):
    return os.path.join(shap_dir(train_tag), "shap_importance_fraction.json")


def shap_summary_path(train_tag):
    return os.path.join(shap_dir(train_tag), "shap_summary.pdf")


def shap_bar_path(train_tag):
    return os.path.join(shap_dir(train_tag), "shap_bar.pdf")


def shap_cumulative_path(train_tag):
    return os.path.join(shap_dir(train_tag), "shap_cumulative.pdf")


def mc_output_path(train_tag):
    return os.path.join(selected_dir(train_tag), "MC_with_score.root")


def data_output_path(train_tag):
    return os.path.join(selected_dir(train_tag), "DATA_with_score.root")


def legacy_model_path(train_tag):
    return model_path(train_tag)


def legacy_scaler_path(train_tag):
    return scaler_path(train_tag)


def legacy_model_config_path(train_tag):
    return model_config_path(train_tag)


def legacy_training_score_path(train_tag):
    return training_score_path(train_tag)


def legacy_feature_importance_path(train_tag):
    return feature_importance_path(train_tag)


def legacy_feature_importance_cumulative_path(train_tag):
    return feature_importance_cumulative_path(train_tag)


def legacy_shap_importance_path(train_tag):
    return shap_importance_path(train_tag)


def legacy_shap_importance_fraction_path(train_tag):
    return shap_importance_fraction_path(train_tag)


def legacy_shap_summary_path(train_tag):
    return shap_summary_path(train_tag)


def legacy_shap_bar_path(train_tag):
    return shap_bar_path(train_tag)


def legacy_shap_cumulative_path(train_tag):
    return shap_cumulative_path(train_tag)


def legacy_mc_output_path(train_tag):
    return mc_output_path(train_tag)


def legacy_data_output_path(train_tag):
    return data_output_path(train_tag)


def legacy_cut_scan_dir(train_tag):
    return cut_scan_dir(train_tag)


def resolve_model_path(train_tag):
    return resolve_existing(condor_model_path(train_tag), model_path(train_tag))


def resolve_scaler_path(train_tag):
    return resolve_existing(condor_scaler_path(train_tag), scaler_path(train_tag))


def resolve_model_config_path(train_tag):
    return resolve_existing(condor_model_config_path(train_tag), model_config_path(train_tag))


def resolve_data_output_path(train_tag):
    return data_output_path(train_tag)


def resolve_mc_output_path(train_tag):
    return mc_output_path(train_tag)
