# XGBoost

This directory contains a small XGBoost-based workflow for training, hyperparameter tuning, score application, and mass-shape plotting for the X(3872) analysis.

The repository is currently stored under EOS for versioning and artifact sharing. The HTCondor helper files are kept here as examples/documentation, but are not meant to be launched directly from this EOS path. For actual batch running, copy or mirror the workflow to an AFS path first.

## Repository Structure

- `XGBoost.py`: baseline training script.
- `optuna_XGBoost.py`: Optuna-based hyperparameter scan and final training.
- `apply.py`: applies a trained model to MC and DATA ROOT ntuples and writes `xgb_score`.
- `draw.py`: scans score cuts and produces `Bmass` plots from scored DATA.
- `shap_importance.py`: computes SHAP feature rankings, normalized fractions, and summary plots from a trained model.
- `paths.py`: shared output-path helpers for the organized directory layout and legacy fallback.
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

## Output Layout

New runs are written into the organized layout below:

```text
xgb_output/
  models/<train_tag>/
    xgb_model.pkl
    scaler.pkl
    model_config.json

  training/<train_tag>/
    xgb_score.pdf
    feature_importance.json
    feature_importance_cumulative.pdf

  shap/<train_tag>/
    shap_importance.json
    shap_importance_fraction.json
    shap_summary.pdf
    shap_bar.pdf
    shap_cumulative.pdf

selected_events/<train_tag>/
  MC_with_score.root
  DATA_with_score.root
  cut_scan/
    X_cut010.pdf
    X_cut020.pdf
    ...
```

The read-side scripts keep compatibility with the older flat layout. In particular, `apply.py` and `shap_importance.py` can still load models from the old `xgb_output/xgb_model_<train_tag>.pkl` style layout, and `draw.py` can still read `selected_events/DATA_with_score_<train_tag>.root` if needed.


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

Training outputs are saved to:

- `xgb_output/training/<train_tag>/feature_importance.json`
- `xgb_output/training/<train_tag>/feature_importance_cumulative.pdf`
- `xgb_output/training/<train_tag>/xgb_score.pdf`

Model artifacts are saved to:

- `xgb_output/models/<train_tag>/xgb_model.pkl`
- `xgb_output/models/<train_tag>/scaler.pkl`
- `xgb_output/models/<train_tag>/model_config.json`

SHAP outputs are saved to:

- `xgb_output/shap/<train_tag>/shap_importance.json`
- `xgb_output/shap/<train_tag>/shap_importance_fraction.json`
- `xgb_output/shap/<train_tag>/shap_summary.pdf`
- `xgb_output/shap/<train_tag>/shap_bar.pdf`
- `xgb_output/shap/<train_tag>/shap_cumulative.pdf`

Scored ROOT outputs are saved to:

- `selected_events/<train_tag>/MC_with_score.root`
- `selected_events/<train_tag>/DATA_with_score.root`

Cut-scan plots are saved to:

- `selected_events/<train_tag>/cut_scan/`

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
