import os
import re

XGB_OUTPUT_DIR = "xgb_output"
SELECTED_EVENTS_DIR = "selected_events"


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def resolve_existing(*candidates):
    for path in candidates:
        if path and os.path.exists(path):
            return path
    return candidates[0] if candidates else None


def group_base_tag(train_tag):
    match = re.match(r"(.+)_v\d+$", train_tag)
    return match.group(1) if match else train_tag


def train_group_tag(train_tags):
    if not train_tags:
        raise ValueError("No train tags provided")
    base_tags = [group_base_tag(tag) for tag in train_tags]
    if len(set(base_tags)) != 1:
        raise ValueError(f"Train tags do not belong to a single group: {train_tags}")
    return base_tags[0] if len(train_tags) > 1 else train_tags[0]


def train_batch_tag(train_tag):
    match = re.match(r"(.+)_v(\d+)$", train_tag)
    if not match:
        return train_tag
    base_tag, version_text = match.groups()
    version = int(version_text)
    start = ((version - 1) // 10) * 10 + 1
    end = start + 9
    return f"{base_tag}_v{start}_v{end}"


def model_dir(train_tag):
    return os.path.join(XGB_OUTPUT_DIR, "models", train_tag)


def training_dir(train_tag):
    return os.path.join(XGB_OUTPUT_DIR, "training", train_tag)


def condor_model_dir(train_tag):
    return os.path.join(XGB_OUTPUT_DIR, "models", train_batch_tag(train_tag), train_tag)


def condor_training_dir(train_tag):
    return os.path.join(XGB_OUTPUT_DIR, "training", train_batch_tag(train_tag), train_tag)


def shap_dir(train_tag):
    return os.path.join(XGB_OUTPUT_DIR, "shap", train_tag)


def selected_dir(train_tag):
    return os.path.join(SELECTED_EVENTS_DIR, train_tag)


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
    return os.path.join(XGB_OUTPUT_DIR, f"xgb_model_{train_tag}.pkl")


def legacy_scaler_path(train_tag):
    return os.path.join(XGB_OUTPUT_DIR, f"scaler_{train_tag}.pkl")


def legacy_model_config_path(train_tag):
    return os.path.join(XGB_OUTPUT_DIR, f"model_config_{train_tag}.json")


def legacy_training_score_path(train_tag):
    return os.path.join(XGB_OUTPUT_DIR, f"xgb_score_{train_tag}.pdf")


def legacy_feature_importance_path(train_tag):
    return os.path.join(XGB_OUTPUT_DIR, f"feature_importance_{train_tag}.json")


def legacy_feature_importance_cumulative_path(train_tag):
    return os.path.join(XGB_OUTPUT_DIR, f"feature_importance_cumulative_{train_tag}.pdf")


def legacy_shap_importance_path(train_tag):
    return os.path.join(XGB_OUTPUT_DIR, f"shap_importance_{train_tag}.json")


def legacy_shap_importance_fraction_path(train_tag):
    return os.path.join(XGB_OUTPUT_DIR, f"shap_importance_fraction_{train_tag}.json")


def legacy_shap_summary_path(train_tag):
    return os.path.join(XGB_OUTPUT_DIR, f"shap_summary_{train_tag}.pdf")


def legacy_shap_bar_path(train_tag):
    return os.path.join(XGB_OUTPUT_DIR, f"shap_bar_{train_tag}.pdf")


def legacy_shap_cumulative_path(train_tag):
    return os.path.join(XGB_OUTPUT_DIR, f"shap_cumulative_{train_tag}.pdf")


def legacy_mc_output_path(train_tag):
    return os.path.join(SELECTED_EVENTS_DIR, f"MC_with_score_{train_tag}.root")


def legacy_data_output_path(train_tag):
    return os.path.join(SELECTED_EVENTS_DIR, f"DATA_with_score_{train_tag}.root")


def legacy_cut_scan_dir(train_tag):
    return os.path.join(SELECTED_EVENTS_DIR, f"{train_tag}_pdf")


def resolve_model_path(train_tag):
    return resolve_existing(condor_model_path(train_tag), model_path(train_tag), legacy_model_path(train_tag))


def resolve_scaler_path(train_tag):
    return resolve_existing(condor_scaler_path(train_tag), scaler_path(train_tag), legacy_scaler_path(train_tag))


def resolve_model_config_path(train_tag):
    return resolve_existing(condor_model_config_path(train_tag), model_config_path(train_tag), legacy_model_config_path(train_tag))


def resolve_data_output_path(train_tag):
    return resolve_existing(data_output_path(train_tag), legacy_data_output_path(train_tag))


def resolve_mc_output_path(train_tag):
    return resolve_existing(mc_output_path(train_tag), legacy_mc_output_path(train_tag))
