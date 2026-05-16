# Central hyperparameter configuration.


def int_range(low, high):
    return {"type": "int", "low": low, "high": high}


def float_range(low, high, log=False):
    cfg = {"type": "float", "low": low, "high": high}
    if log:
        cfg["log"] = True
    return cfg


_pbpb_optuna = {
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
}
_pp_optuna = {
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
}

OPTUNA_SPACES = {
    "pbpb": {"X": dict(_pbpb_optuna), "Bu": dict(_pbpb_optuna), "Bd": dict(_pbpb_optuna), "Bs": dict(_pbpb_optuna)},
    "pp": {"X": dict(_pp_optuna), "Bu": dict(_pp_optuna), "Bd": dict(_pp_optuna), "Bs": dict(_pp_optuna)},
}

_pbpb_direct = {
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
}
_pp_direct = {
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
}

DIRECT_XGB_PARAMS = {
    "pbpb": {"X": dict(_pbpb_direct), "Bu": dict(_pbpb_direct), "Bd": dict(_pbpb_direct), "Bs": dict(_pbpb_direct)},
    "pp": {"X": dict(_pp_direct), "Bu": dict(_pp_direct), "Bd": dict(_pp_direct), "Bs": dict(_pp_direct)},
}
