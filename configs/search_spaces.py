# Central hyperparameter configuration.
# - OPTUNA_SPACES: search space used by workflows/condor_optuna_XGBoost.py
# - DIRECT_XGB_PARAMS: fixed params used by workflows/xgboost_train_direct.py


def int_range(low, high):
    return {"type": "int", "low": low, "high": high}


def float_range(low, high, log=False):
    cfg = {"type": "float", "low": low, "high": high}
    if log:
        cfg["log"] = True
    return cfg


OPTUNA_SPACES = {
    "pbpb": {
        "n_estimators": int_range(500, 1800),
        "learning_rate": float_range(0.01, 0.08, log=True),
        "max_depth": int_range(2, 6),
        "min_child_weight": int_range(2, 20),
        "subsample": float_range(0.65, 0.95),
        "colsample_bytree": float_range(0.65, 0.95),
        "gamma": float_range(0.0, 4.0),
        "reg_alpha": float_range(0.0, 3.0),
        "reg_lambda": float_range(1.0, 12.0),
        "max_delta_step": float_range(0.0, 4.0),
    },
    "pp": {
        "n_estimators": int_range(400, 1600),
        "learning_rate": float_range(0.01, 0.10, log=True),
        "max_depth": int_range(2, 6),
        "min_child_weight": int_range(1, 15),
        "subsample": float_range(0.65, 1.0),
        "colsample_bytree": float_range(0.65, 1.0),
        "gamma": float_range(0.0, 3.0),
        "reg_alpha": float_range(0.0, 2.0),
        "reg_lambda": float_range(1.0, 10.0),
        "max_delta_step": float_range(0.0, 3.0),
    },
}


DIRECT_XGB_PARAMS = {
    "pbpb": {
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
    "pp": {
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
}
