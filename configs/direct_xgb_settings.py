# Direct (non-Optuna) XGBoost settings by sample + version.
#
# Structure:
#   DIRECT_XGB_PARAMS[sample][model_version]
#
# Each version is a complete, independent parameter set. Do not inherit from or
# partially override another version. Dataset and channel differences outside
# these fixed model parameters remain resolved by configs/samples.py and
# utils/varsets.py.


REQUIRED_DIRECT_XGB_FIELDS = frozenset(
    {
        "n_estimators",
        "learning_rate",
        "max_depth",
        "min_child_weight",
        "subsample",
        "colsample_bytree",
        "gamma",
        "reg_alpha",
        "reg_lambda",
        "max_delta_step",
    }
)


DIRECT_XGB_PARAMS = {
    "pbpb": {
        1: {
            "n_estimators": 1000,
            "learning_rate": 0.03,
            "max_depth": 4,
            "min_child_weight": 8,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "gamma": 1.0,
            "reg_alpha": 0.5,
            "reg_lambda": 6.0,
            "max_delta_step": 1.0,
        },
    },
    "pp": {
        1: {
            "n_estimators": 900,
            "learning_rate": 0.04,
            "max_depth": 4,
            "min_child_weight": 6,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "gamma": 0.5,
            "reg_alpha": 0.3,
            "reg_lambda": 4.0,
            "max_delta_step": 0.0,
        },
    },
}


def resolve_direct_xgb_params(sample: str, model_version: int) -> dict:
    try:
        params = DIRECT_XGB_PARAMS[sample][model_version]
    except KeyError as exc:
        available = tuple(sorted(DIRECT_XGB_PARAMS.get(sample, {})))
        raise ValueError(
            "No complete direct-XGBoost configuration for "
            f"sample='{sample}', xgb_v{model_version}. "
            f"Available versions: {available}"
        ) from exc

    fields = set(params)
    missing = REQUIRED_DIRECT_XGB_FIELDS - fields
    unknown = fields - REQUIRED_DIRECT_XGB_FIELDS
    if missing or unknown:
        raise ValueError(
            "Invalid direct-XGBoost configuration for "
            f"sample='{sample}', xgb_v{model_version}: "
            f"missing={tuple(sorted(missing))}, unknown={tuple(sorted(unknown))}"
        )
    return dict(params)
