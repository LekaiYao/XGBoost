# XGBoost

This directory contains the current XGBoost workflow for the PbPb mass-spectrum analysis, including:

- single-model training
- Optuna hyperparameter scans
- grouped score application
- grouped cut-scan plotting
- Condor submission wrappers
- hyperparameter-design notes

The repository code lives under EOS, while the actual HTCondor submission wrappers must be launched from AFS.

## AFS and EOS

Current split:

- EOS repo path:
  `/eos/home-l/leyao/pbpb_work/X_analysis/XGBoost`
- AFS Condor submission path:
  `/afs/cern.ch/user/l/leyao/private/pbpb_work/X_analysis/XGBoost`

The current Condor pattern is:

1. submit from AFS
2. run AFS-side wrapper scripts such as `run.sh` or `run_batch_compare.sh`
3. `cd` into the EOS repository
4. activate `.venv`
5. execute the EOS-side Python scripts

## Main scripts

- `XGBoost.py`
  Baseline single-model training script.
- `optuna_XGBoost.py`
  Legacy local Optuna scan script.
- `condor_optuna_XGBoost.py`
  Current main Condor training script. Supports named search-space presets such as `v11`, `v21`, `v31`, etc.
- `apply.py`
  Applies one trained model to MC and DATA and writes `xgb_score`.
- `draw.py`
  Draws cut-scan mass plots for one scored `train_tag` from `selected_events/<train_tag>/DATA_with_score.root`.
- `batch_apply_scores.py`
  Loads multiple trained models from one group and writes all scores into one grouped ROOT output.
- `batch_draw_scores.py`
  Legacy grouped draw script that reads one grouped DATA ROOT and writes plots into each `train_tag` output directory.
- `shap_importance.py`
  SHAP feature-importance workflow for trained models.

## Condor wrapper scripts

AFS-side Condor helpers:

- `run.sh`
  Wrapper for training jobs.
- `submit.sub`
  Training submission file. The queued `train_tag` values determine which search-space presets are launched.
- `run_batch_compare.sh`
  Wrapper for grouped apply plus grouped draw.
- `submit_batch_compare.sub`
  Batch compare submission file. Now supports explicit version ranges.
- `submit_batch_compare_4v1.sub`
  Example batch compare submit file for the `pb23_4v_o100` line.
- `submit_batch_compare_p2s.sub`
  Example batch compare submit file for the `pb23p2s_4v2_o50` line.

## Naming rule

Current training tags follow:

`<sample>_<feature-version>_<objective-version>o<optuna-trials>_v<search-space-version>`

Examples:

- `pb23_4v2_o50_v1`
- `pb23_4v_o100_v7`
- `pb23p2s_4v2_o50_v10`
- `pb23v2_4v2_o50_v11`
- `pb23v2_4v2_o100_v21`
- `pb23v2_4v2_o100_v40`
- `pb23v4_4v2_3o200_v1`
- `pb23v4_7v_3o200_v28`

Interpretation:

- `pb23`, `pb23p2s`, `pb23v2`:
  sample line
- `4v`, `4v2`, `5v`, `7v`:
  input-variable version
- `3o200`:
  objective version `3` with `200` Optuna trials
- `o50`, `o100`:
  Optuna trial count
- `vN`:
  search-space preset version

Compatibility note:

- Older historical trainings may still use tags without the explicit objective-version field, for example `..._o100_v21`
- New trainings should use the explicit form `..._<objective-version>o<trials>_vN`

## Current pb23v3_7v training setup

The current `pb23v3_7v` Condor training line uses:

- signal MC:
  `/eos/user/h/hmarques/RUN3_Data_MC_sharing/X3872/PbPb23/flat_ntmix_PbPb23_MC.root:ntmix`
- background DATA:
  `/eos/user/h/hmarques/RUN3_Data_MC_sharing/X3872/PbPb23/flat_ntmix_PbPb23_DATA0.root:ntmix`
- signal selection:
  `isX3872 == 1`
- background selection:
  `(3.744 < Bmass < 3.802) or (3.942 < Bmass < 4.00)`
- input variables:
  - `Btrk1dR`
  - `Btrk2dR`
  - `Btrk2Pt`
  - `BtrkPtimb`
  - `Bchi2Prob`
  - `Balpha`
  - `Bnorm_trk1Dxy`

Other current training details:

- inputs are standardized with `StandardScaler`
- class imbalance is handled with `scale_pos_weight = n_bkg / n_sig`
- train/validation/test splitting is stratified
- `random_state = 42`
- the Optuna objective is based on validation-set cut scanning and maximization of weighted `S/sqrt(S+B)`
- `S` is scaled by `3491.0 / 70439.0`
- `B` is scaled by `signal-window-width / sideband-width`

## Search-space history

The running record of hyperparameter-design choices is stored in:

- `notes/optuna_search_space_history_pb23v2_4v2.md`
- `notes/optuna_search_space_history_pb23v2_5v.md`
- `notes/optuna_search_space_history_pb23v3_7v.md`

These files currently summarize:

- `pb23_4v2_o50_v1-v10`
- `pb23_4v_o100_v1-v10`
- `pb23p2s_4v2_o50_v1-v10`
- `pb23v2_4v2_o50_v11-v20`
- `pb23v2_4v2_o100_v21-v30`
- `pb23v2_4v2_o100_v31-v40`
- `pb23v2_5v_o100_v1-v40`
- `pb23v3_7v_o100_v1-v50`

Future training batches should continue to append new sections to their corresponding history files.

## Plotting conventions

All current mass-spectrum plotting scripts are standardized to:

- mass range:
  `3.62 < Bmass < 4.0`
- histogram bin width:
  `0.01`
- additional cut:
  `BQvalue < 0.13`
- output naming:
  `DATA_cutXXX.pdf` or `DATA_cutXXXX.pdf`

The old `X_cut...pdf` naming should no longer be used in the active plotting scripts.

## Typical workflows

### Single training

```bash
python3 XGBoost.py <train_tag>
```

### Legacy local Optuna

```bash
python3 optuna_XGBoost.py <train_tag>
```

### Current Condor-style Optuna training

Local direct run for debugging:

```bash
OPTUNA_N_TRIALS=100 python3 condor_optuna_XGBoost.py <train_tag> <search_space_tag>
```

Typical Condor run:

1. edit AFS `submit.sub`
2. set the queued `train_tag, search_space_tag`
3. run:

```bash
condor_submit submit.sub
```

### Single-model score application

```bash
python3 apply.py <train_tag>
```

### Single-model drawing

```bash
python3 draw.py <train_tag>
```

### Grouped score application

```bash
python3 batch_apply_scores.py [--output-tag <output_tag>] <train_tag1> <train_tag2> ...
```

This writes grouped outputs to:

- `selected_events/<output_tag>/MC_with_score.root`
- `selected_events/<output_tag>/DATA_with_score.root`

If `--output-tag` is omitted, `<output_tag>` defaults to `<group_tag>`.

### Batch compare through Condor

Current `run_batch_compare.sh` accepts:

```bash
run_batch_compare.sh <group_tag> <version_start> <version_end>
```

For example, this expands:

- `pb23v2_4v2_o50 11 20`

into:

- `pb23v2_4v2_o50_v11`
- ...
- `pb23v2_4v2_o50_v20`

and then runs:

- `batch_apply_scores.py`
- `batch_draw_scores.py`

The grouped ROOT outputs are written under a version-range-specific directory:

- `selected_events/<group_tag>_v<version_start>_v<version_end>/`

For example:

- `selected_events/pb23v2_4v2_o100_v21_v30/`
- `selected_events/pb23v2_4v2_o100_v31_v40/`

## Output layout

Grouped score-application outputs are stored in:

- `selected_events/<group_tag>_v<start>_v<end>/`

Inside each grouped directory:

- grouped ROOT files stay directly under the group directory:
  - `MC_with_score.root`
  - `DATA_with_score.root`
- grouped cut-scan plots are written directly under:
  - `selected_events/<grouped_output_tag>/<train_tag>/`

Examples:

- `selected_events/pb23v4_5v_3o200_v61_v70/DATA_with_score.root`
- `selected_events/pb23v4_5v_3o200_v61_v70/pb23v4_5v_3o200_v61/DATA_fid_cut090.pdf`

For the Condor workflow, model and training artifacts are now also grouped by version range:

- `xgb_output/models/<grouped_output_tag>/<train_tag>/`
- `xgb_output/training/<grouped_output_tag>/<train_tag>/`

Examples:

- `xgb_output/models/pb23v4_5v_3o200_v61_v70/pb23v4_5v_3o200_v61/xgb_model.pkl`
- `xgb_output/models/pb23v4_5v_3o200_v61_v70/pb23v4_5v_3o200_v61/run_metadata.json`
- `xgb_output/training/pb23v4_5v_3o200_v61_v70/pb23v4_5v_3o200_v61/xgb_score.pdf`
- `xgb_output/training/pb23v4_5v_3o200_v61_v70/pb23v4_5v_3o200_v61/test_roc.pdf`

This grouped `xgb_output` layout applies to Condor training only.

Local single-model workflows remain unchanged:

- `xgb_output/models/<train_tag>/`
- `xgb_output/training/<train_tag>/`

```text
xgb_output/
  models/<train_tag>/
    xgb_model.pkl
    scaler.pkl
    model_config.json
    run_metadata.json

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
```

Scored ROOT outputs and cut scans are stored in:

```text
selected_events/<train_tag>/
  MC_with_score.root
  DATA_with_score.root
  cut_scan/
    DATA_cut000.pdf
    DATA_cut050.pdf
    ...

selected_events/<group_tag>/
  MC_with_score.root
  DATA_with_score.root
```

## Local Python environment

Use the project-local `.venv`.

Typical setup:

```bash
python3.9 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

Main dependencies:

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
