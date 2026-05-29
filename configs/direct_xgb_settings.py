# Direct (non-Optuna) XGBoost settings by dataset + channel.
#
# Structure:
#   DIRECT_XGB_PARAMS[dataset_token][channel]


_PBPB_DIRECT_BASE = {
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

_PP_DIRECT_BASE = {
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
    "pb23": {
        "X": dict(_PBPB_DIRECT_BASE),
        "Bu": dict(_PBPB_DIRECT_BASE),
        "Bd": dict(_PBPB_DIRECT_BASE),
        "Bs": dict(_PBPB_DIRECT_BASE),
    },
    "pb24": {
        "X": dict(_PBPB_DIRECT_BASE),
        "Bu": dict(_PBPB_DIRECT_BASE),
        "Bd": dict(_PBPB_DIRECT_BASE),
        "Bs": dict(_PBPB_DIRECT_BASE),
    },
    "pp24": {
        "X": dict(_PP_DIRECT_BASE),
        "Bu": dict(_PP_DIRECT_BASE),
        "Bd": dict(_PP_DIRECT_BASE),
        "Bs": dict(_PP_DIRECT_BASE),
    },
}

