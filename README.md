# XGBoost

This directory contains a small XGBoost-based workflow for training, hyperparameter tuning, score application, and mass-shape plotting for the X(3872) analysis.

The repository is currently stored under EOS for versioning and artifact sharing. The HTCondor helper files are kept here as examples/documentation, but are not meant to be launched directly from this EOS path. For actual batch running, copy or mirror the workflow to an AFS path first.

## Repository Structure

- `XGBoost.py`: baseline training script.
- `optuna_XGBoost.py`: Optuna-based hyperparameter scan and final training.
- `apply.py`: applies a trained model to MC and DATA ROOT ntuples and writes `xgb_score`.
- `draw.py`: scans score cuts and produces `Bmass` plots from scored DATA.
- `shap_importance.py`: computes SHAP feature rankings, normalized fractions, and summary plots from a trained model.
- `run.sh`: example batch entrypoint used by HTCondor after relocating the workflow to AFS.
- `submit.sub`: example HTCondor submission file kept for later AFS-side usage.
- `requirements.txt`: Python dependencies for the local virtual environment.

## Workflow

1. Train a baseline model:

```bash
python3 XGBoost.py <train_tag>
```

2. Train with Optuna tuning:

```bash
python3 optuna_XGBoost.py <train_tag>
```

3. Apply the trained model:

```bash
python3 apply.py <train_tag>
```

4. Draw score-cut mass distributions:

```bash
python3 draw.py <train_tag>
```

5. Compute SHAP importance, normalized fractions, and summary plots:

```bash
python3 shap_importance.py <train_tag> [max_events]
```

## Input Data

The scripts currently read ROOT ntuples from:

- `SIG_PATH = /eos/home-l/leyao/pbpb_work/X_analysis/ppRef24/flat_ntmix_ppRef_MC.root:ntmix`
- `BKG_PATH = /eos/home-l/leyao/pbpb_work/X_analysis/ppRef24/flat_ntmix_ppRef_DATA.root:ntmix`

Signal events are selected with `isX3872 == 1`, while background is taken from `Bmass` sidebands.

## Features

The current model uses four input variables:

- `Btrk1dR`
- `Btrk2dR`
- `BtrkPtimb`
- `Bchi2Prob`

They are standardized with `StandardScaler` before training and inference.

After training, both `XGBoost.py` and `optuna_XGBoost.py` print the feature importance ranking in descending order and save it to:

- `xgb_output/feature_importance_<train_tag>.json`
- `xgb_output/feature_importance_cumulative_<train_tag>.pdf`
- `xgb_output/shap_importance_<train_tag>.json`
- `xgb_output/shap_importance_fraction_<train_tag>.json`
- `xgb_output/shap_summary_<train_tag>.pdf`
- `xgb_output/shap_bar_<train_tag>.pdf`
- `xgb_output/shap_cumulative_<train_tag>.pdf`

`apply.py` writes scored ROOT outputs to:

- `selected_events/MC_with_score_<train_tag>.root`
- `selected_events/DATA_with_score_<train_tag>.root`

`draw.py` reads `selected_events/DATA_with_score_<train_tag>.root` and saves score-scan mass plots to:

- `selected_events/<train_tag>_pdf/`

## Local Python Environment

Use a project-local virtual environment named `.venv/`.

Recommended interpreter:

- Base interpreter location: `/usr/bin/python3.9`
- Python version: `3.9.x`
- `include-system-site-packages = false`

The repository dependencies are listed in `requirements.txt`:

- `uproot`
- `awkward`
- `pandas`
- `numpy`
- `matplotlib`
- `scikit-learn`
- `xgboost`
- `optuna`
- `joblib`
- `shap`

To create the environment manually:

```bash
python3.9 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

## Batch Running

These files are included as reference only while the repository lives on EOS. HTCondor on this setup should be launched from an AFS location instead.


`run.sh` activates `.venv/` and launches:

```bash
python3 optuna_XGBoost.py ${train_tag}
```

The provided `submit.sub` submits one Condor job with a configurable `train_tag`.
