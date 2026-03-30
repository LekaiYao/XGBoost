# XGBoost

This directory contains a small XGBoost-based workflow for training, hyperparameter tuning, score application, and mass-shape plotting for the X(3872) analysis.

## Repository Structure

- `XGBoost.py`: baseline training script.
- `optuna_XGBoost.py`: Optuna-based hyperparameter scan and final training.
- `apply.py`: applies a trained model to MC and DATA ROOT ntuples and writes `xgb_score`.
- `draw.py`: scans score cuts and produces `Bmass` plots from scored DATA.
- `run.sh`: batch entrypoint used by HTCondor.
- `submit.sub`: example HTCondor submission file.
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

To create the environment manually:

```bash
python3.9 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

## Batch Running

`run.sh` activates `.venv/` and launches:

```bash
python3 optuna_XGBoost.py ${train_tag}
```

The provided `submit.sub` submits one Condor job with a configurable `train_tag`.
