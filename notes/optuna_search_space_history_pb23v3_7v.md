# pb23v3_7v Optuna Search-Space History

This file tracks the planned and executed hyperparameter-space batches for the
`pb23v3_7v` training line.

Current line definition:

- sample tag:
  `pb23v3`
- feature tag:
  `7v`
- Optuna trials:
  `o100`
- signal selection:
  `isX3872 == 1`
- background selection:
  `(3.744 < Bmass < 3.802) or (3.942 < Bmass < 4.00)`
- input columns:
  - `Btrk1dR`
  - `Btrk2dR`
  - `Btrk2Pt`
  - `BtrkPtimb`
  - `Bchi2Prob`
  - `Balpha`
  - `Bnorm_trk1Dxy`

## pb23v3_7v_o100_v1-v50

The `v1-v50` plan is intentionally organized into 5 style families instead of
one broad mixed scan.

### v1-v10: BDT-like stable shallow trees

Goal:

- stay close to traditional HEP BDT behavior
- test whether smoother, shallower boosted trees reproduce the literature trend
- reduce learning of local sideband fluctuations

Typical features:

- `max_depth = 2-4`
- moderate to strong regularization
- moderate sampling
- some presets include `max_delta_step` for extra stability under imbalance

### v11-v20: low-learning-rate, many-tree smooth ranking

Goal:

- test whether background suppression needs slower and smoother boosting
- improve ordering of complex combinatorial background without aggressive local fits

Typical features:

- low `learning_rate`
- high `n_estimators`
- shallow to medium depth
- balanced regularization

### v21-v30: medium/deep aggressive separation

Goal:

- test whether previous scans were still underfitting
- allow stronger nonlinearity and more aggressive background reordering

Typical features:

- `max_depth = 4-8`
- medium to high `learning_rate`
- looser `min_child_weight` and `gamma`
- weaker regularization in part of the group

### v31-v40: deep but constrained models

Goal:

- allow complex PbPb background structure
- keep deep trees under control so they do not simply memorize sidebands

Typical features:

- `max_depth = 5-8`
- stronger `min_child_weight`, `gamma`, `reg_alpha`, `reg_lambda`
- nonzero `max_delta_step` appears more often

### v41-v50: decorrelation and strong stochasticity

Goal:

- test whether stronger row/column randomness improves generalization
- probe whether the model can avoid locking onto local sideband-specific structures

Typical features:

- wide `subsample` / `colsample_bytree` variation
- `colsample_bylevel` introduced explicitly
- some asymmetric row-vs-column sampling setups
- one final wildcard space (`v50`) covers a broad mixed region

Future `pb23v3_7v` batches should append new sections here rather than create a
new file.

## pb23v3_7v_o100_v51-v60

This block is a focused refinement family built from the historical convergence
pattern of `pb23v2_4v2_o100_v31`.

Observed high-value region from that earlier line:

- `n_estimators` near the upper boundary, around `2050-2200`
- `learning_rate` around `0.016-0.019`
- `max_depth = 3`
- `min_child_weight` high, around `24-26`
- `subsample` and `colsample_bytree` both around `0.51-0.53`
- `gamma` around `4.0-4.3`
- `reg_alpha` around `5.0-5.5`
- `reg_lambda` around `13-14`

These `v51-v60` presets therefore do not explore a new style family. They zoom
in on one specific shallow-tree, strongly regularized, low-sampling region that
showed late-trial convergence and stable high objective values.

The 10 presets cover:

- `v51-v53`
  tight refinements around the historical optimum
- `v54-v55`
  slight widening in `max_depth`, `min_child_weight`, and learning rate
- `v56-v57`
  very tight high-regularization refinements around the apparent best basin
- `v58-v60`
  wider guard-band scans around the same basin, to test whether the optimum is
  robust or sitting on a narrow edge
