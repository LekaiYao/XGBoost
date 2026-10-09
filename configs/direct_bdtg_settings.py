"""Fixed, user-approved TMVA BDTG settings; no parameter search."""
from copy import deepcopy
BDTG_PARAMS = {1: dict(NTrees=1000, Shrinkage=0.03, MaxDepth=4,
    MinNodeSize="1%", UseBaggedBoost=True, BaggedSampleFraction=0.8,
    nCuts=20, BoostType="Grad", PruneMethod="NoPruning",
    UseRandomisedTrees=False, SkipNormalization=True)}
SPLIT_SEED = 42
TRAIN_FRACTION = 0.75

def resolve_bdtg_params(version):
    if version not in BDTG_PARAMS:
        raise ValueError(f"Unknown BDTG version {version}")
    return deepcopy(BDTG_PARAMS[version])

def tmva_options(params):
    return "!H:!V:" + ":".join(f"{k}={str(v).lower() if isinstance(v, bool) else v}" for k,v in params.items())
