#!/bin/bash
set -euo pipefail
cd /eos/home-l/leyao/pbpb_work/X_analysis/XGBoost
source .venv/bin/activate
export PYTHONNOUSERSITE=1
exec .venv/bin/python - "$1" <<'PYRANK'
import sys
from configs.feature_selection import resolve_feature_selection_experiment
name = sys.argv[1]
if resolve_feature_selection_experiment(name)["operation"] == "add1":
    from workflows.rank_add1_auc import main
else:
    from workflows.rank_feature_selection import rank_experiment as main
main(name)
PYRANK
