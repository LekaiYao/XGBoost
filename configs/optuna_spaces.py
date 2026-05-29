# Optuna search spaces (dataset + channel + space version).
#
# Structure:
#   OPTUNA_SPACES[dataset_token][channel][space_version]
# Example:
#   OPTUNA_SPACES["pb23"]["X"]["v1"]
#
# To add v2 later (manual):
#   1) copy *_v1 block -> *_v2
#   2) update values in *_v2
#   3) register "v2" under the same dataset/channel entry


def int_range(low, high):
    return {"type": "int", "low": low, "high": high}


def float_range(low, high, log=False):
    cfg = {"type": "float", "low": low, "high": high}
    if log:
        cfg["log"] = True
    return cfg


def _clone_space(space):
    return {k: dict(v) for k, v in space.items()}


# Base values (can currently stay the same across many keys).
_PBPB_BASE_V1 = {
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

_PP_BASE_V1 = {
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


# Explicit dataset+channel blocks (v1).
PB23_X_V1 = {
    "n_estimators": int_range(800, 4000),
    "learning_rate": float_range(0.02, 0.12, log=True),
    "max_depth": int_range(2, 5),
    "min_child_weight": float_range(10.0, 200.0, log=True),
    "subsample": float_range(0.60, 0.95),
    "colsample_bytree": float_range(0.60, 0.95),
    "gamma": float_range(0.0, 20.0),
    "reg_alpha": float_range(1e-3, 20.0, log=True),
    "reg_lambda": float_range(3.0, 300.0, log=True),
    "max_delta_step": float_range(1.0, 8.0),
}
PB23_BU_V1 = _clone_space(_PBPB_BASE_V1)
PB23_BD_V1 = _clone_space(_PBPB_BASE_V1)
PB23_BS_V1 = _clone_space(_PBPB_BASE_V1)

PB24_X_V1 = {
    "n_estimators": int_range(800, 4000),
    "learning_rate": float_range(0.02, 0.12, log=True),
    "max_depth": int_range(2, 5),
    "min_child_weight": float_range(10.0, 200.0, log=True),
    "subsample": float_range(0.60, 0.95),
    "colsample_bytree": float_range(0.60, 0.95),
    "gamma": float_range(0.0, 20.0),
    "reg_alpha": float_range(1e-3, 20.0, log=True),
    "reg_lambda": float_range(3.0, 300.0, log=True),
    "max_delta_step": float_range(1.0, 8.0),
}
PB24_BU_V1 = _clone_space(_PBPB_BASE_V1)
PB24_BD_V1 = _clone_space(_PBPB_BASE_V1)
PB24_BS_V1 = _clone_space(_PBPB_BASE_V1)

PP24_X_V1 = _clone_space(_PP_BASE_V1)
PP24_BU_V1 = _clone_space(_PP_BASE_V1)
PP24_BD_V1 = _clone_space(_PP_BASE_V1)
PP24_BS_V1 = _clone_space(_PP_BASE_V1)


OPTUNA_SPACES = {
    "pb23": {
        "X": {"v1": PB23_X_V1},
        "Bu": {"v1": PB23_BU_V1},
        "Bd": {"v1": PB23_BD_V1},
        "Bs": {"v1": PB23_BS_V1},
    },
    "pb24": {
        "X": {"v1": PB24_X_V1},
        "Bu": {"v1": PB24_BU_V1},
        "Bd": {"v1": PB24_BD_V1},
        "Bs": {"v1": PB24_BS_V1},
    },
    "pp24": {
        "X": {"v1": PP24_X_V1},
        "Bu": {"v1": PP24_BU_V1},
        "Bd": {"v1": PP24_BD_V1},
        "Bs": {"v1": PP24_BS_V1},
    },
}


def _none_training_options():
    return {"early_stopping_rounds": None}


OPTUNA_TRAINING_OPTIONS = {
    "pb23": {
        "X": {"v1": {"early_stopping_rounds": 100}},
        "Bu": {"v1": _none_training_options()},
        "Bd": {"v1": _none_training_options()},
        "Bs": {"v1": _none_training_options()},
    },
    "pb24": {
        "X": {"v1": {"early_stopping_rounds": 100}},
        "Bu": {"v1": _none_training_options()},
        "Bd": {"v1": _none_training_options()},
        "Bs": {"v1": _none_training_options()},
    },
    "pp24": {
        "X": {"v1": _none_training_options()},
        "Bu": {"v1": _none_training_options()},
        "Bd": {"v1": _none_training_options()},
        "Bs": {"v1": _none_training_options()},
    },
}
