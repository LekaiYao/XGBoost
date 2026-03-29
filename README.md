# XGBoost

This directory contains a small XGBoost-based workflow for training, hyperparameter tuning, score application, and mass-shape plotting for the X(3872) analysis.

## Repository Structure

- `XGBoost.py`: baseline training script.
- `optuna_XGBoost.py`: Optuna-based hyperparameter scan and final training.
- `apply.py`: applies a trained model to MC and DATA ROOT ntuples and writes `xgb_score`.
- `draw.py`: scans score cuts and produces `Bmass` plots from scored DATA.
- `run.sh`: batch entrypoint used by HTCondor.
- `submit.sub`: example HTCondor submission file.

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

The current model uses three input variables:

- `Btrk1dR`
- `Btrk2dR`
- `BtrkPtimb`

They are standardized with `StandardScaler` before training and inference.

## Local Python Environment

This folder contains a local virtual environment in `myenv/`. If you prefer the more common naming convention, it plays the same role as a project-local `.venv/`.

Environment details from `myenv/pyvenv.cfg`:

- Base interpreter location: `/usr/bin`
- Python version: `3.9.25`
- `include-system-site-packages = false`

The code imports and therefore requires at least the following Python packages:

- `uproot`
- `awkward`
- `pandas`
- `numpy`
- `matplotlib`
- `scikit-learn`
- `xgboost`
- `optuna`
- `joblib`

To recreate a similar environment manually:

```bash
python3.9 -m venv .venv
source .venv/bin/activate
pip install uproot awkward pandas numpy matplotlib scikit-learn xgboost optuna joblib
```

## Batch Running

`run.sh` activates `myenv/` and launches:

```bash
python3 optuna_XGBoost.py ${train_tag}
```

The provided `submit.sub` submits one Condor job with a configurable `train_tag`.
