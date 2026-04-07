# Optuna search-space history for current Condor training batches

This file is the running record for the Optuna search-space design used in the current Condor training batches in this repository.

Maintenance rule:

- Every new training batch added under the `pb23v2_4v2` naming line should append one new section here.
- Each section should record the naming range, trial count, and the intended physics or model-style purpose of the search spaces.

## Batch summary

- `pb23_4v2_o50_v1-v10`: original broad scan with mixed conservative and aggressive XGBoost styles.
- `pb23_4v_o100_v1-v10`: same original 10-preset style family, but run with `o100`.
- `pb23p2s_4v2_o50_v1-v10`: same original 10-preset style family, but used in the `pb23p2s` training line.
- `pb23v2_4v2_o50_v11-v20`: first redesign after switching to the updated sample definition, using a split between robust and more aggressive background-suppression styles.
- `pb23v2_4v2_o100_v21-v30`: second redesign with a wider exploration in both strongly regularized and strongly expressive directions.
- `pb23v2_4v2_o100_v31-v40`: third redesign that intentionally probes more extreme and asymmetric model styles not covered well by the previous batches.

## v1-v10

Naming:

- `pb23_4v2_o50_v1` to `pb23_4v2_o50_v10`

High-level intent:

- Establish an initial 10-preset survey around one common physics goal: suppress combinatorial background and make the target mass peak more visible.

Design logic:

- `v1-v2`: conservative and stable starting points.
- `v3-v4`: stronger expression and more aggressive learning.
- `v5`: stronger regularization to reduce overfitting.
- `v6`: high-sampling stable-statistics style.
- `v7`: low learning rate and many trees, testing slow convergence.
- `v8`: the most aggressive preset in the original batch.
- `v9`: deep trees with stronger restrictions.
- `v10`: fewer trees and higher learning rate, testing faster response.

What this batch mainly tested:

- Whether the problem was primarily limited by model capacity.
- Whether deeper or more aggressive XGBoost settings could outperform the conservative BDT-like regime.
- Whether background suppression improved in the mass spectrum without obvious overtraining.

## v11-v20

Naming:

- `pb23v2_4v2_o50_v11` to `pb23v2_4v2_o50_v20`

Sample context:

- Uses the updated `pb23v2` data definition.
- Uses `isX3872 == 1` as signal.
- Uses background sidebands `(3.75 < Bmass < 3.83) or (3.91 < Bmass < 4.00)`.
- Uses `PbPb23_DATA0.root` as the background input sample for training.

High-level intent:

- Move away from the original all-in-one mixed scan and explicitly test two style families:
  one robust family and one more aggressive family.

Design logic:

- `v11-v15`: robust family.
  Shallow to medium trees, stronger regularization, more restricted subsampling.
  These presets are meant to reduce the chance that the model learns unstable local background structure.
- `v16-v20`: aggressive background-suppression family.
  Moderate to deeper trees, looser penalties, or larger effective boosting capacity.
  These presets are meant to test whether the previous scans were still underfitting the combinatorial background ordering problem.

Preset-by-preset intent:

- `v11`: conservative shallow and regularized baseline.
- `v12`: slow-learning stable booster with many trees.
- `v13`: moderate-capacity balanced preset.
- `v14`: deeper but still regularized.
- `v15`: strongest-regularization robust endpoint.
- `v16`: moderately aggressive with high event usage.
- `v17`: many-tree low-learning-rate aggressive scan.
- `v18`: faster-learning deeper model.
- `v19`: deeper trees but strongly constrained.
- `v20`: fast compact aggressive preset.

What this batch mainly tested:

- Whether separating “stable” and “aggressive” styles gives a clearer answer than the original mixed scan.
- Whether better post-cut mass-shape behavior comes from controlling variance or from increasing model flexibility.

## v21-v30

Naming:

- `pb23v2_4v2_o100_v21` to `pb23v2_4v2_o100_v30`

High-level intent:

- Keep the same physics inputs and training sample definition as `v11-v20`, but expand the search in directions that were not fully covered before.
- Increase Optuna statistics from `50` to `100` trials so each preset is probed more meaningfully.

Design logic:

- `v21-v23`: stronger regularization and stronger stochasticity.
- `v24-v26`: higher model capacity for more aggressive background suppression.
- `v27-v29`: mixed styles that probe specific tradeoffs between depth, regularization, and subsampling.
- `v30`: broad balanced catch-all preset.

Preset-by-preset intent:

- `v21`: ultra-conservative.
  Very shallow trees, high `min_child_weight`, strong `gamma`, strong L1/L2, low row and column sampling.
  Tests whether very heavy regularization improves mass-shape stability.
- `v22`: slow boosted stable model.
  Many trees, very low learning rate, moderate depth.
  Tests whether smoother rank building improves background flattening.
- `v23`: stochastic mid-capacity model.
  Moderate depth with stronger subsampling.
  Tests whether decorrelation from randomization helps suppress combinatorial structure.
- `v24`: aggressive high-depth fast learner.
  Tests whether more local nonlinear structure is needed to suppress the exponential-like background.
- `v25`: deeper but slower and broader than `v24`.
  Tests a more controlled aggressive regime.
- `v26`: high-statistics boosting with near-full sampling.
  Tests whether many trees with small learning rate and high sample usage outperform more stochastic settings.
- `v27`: deep but strongly constrained.
  Tests whether high-capacity trees help only when node-growth penalties are also strong.
- `v28`: most aggressive preset in this batch.
  High depth and high learning rate.
  Included to test whether the current setup is still underfitting the background-ordering problem.
- `v29`: deep trees with strong stochastic sampling.
  Tests whether deep structure plus aggressive decorrelation gives a better compromise between suppression and shape stability.
- `v30`: broad balanced preset.
  Acts as a hedge against overcommitting to any one style.

What this batch mainly tested:

- Whether the current setup still has unexplored upside on the aggressive side.
- Whether stronger regularization than `v11-v20` gives more physical post-cut mass-shape behavior.
- Whether broader Optuna coverage at `100` trials changes the ranking of the preset families.

## v31-v40

Naming:

- `pb23v2_4v2_o100_v31` to `pb23v2_4v2_o100_v40`

High-level intent:

- Push farther into deliberately unusual or asymmetric hyperparameter regions instead of only refining the earlier balanced scans.
- Test whether the missing gain comes from one specific ingredient: very low sampling, extremely strong regularization, very deep slow learners, shallow fast learners, or very wide unconstrained scans.

Design logic:

- `v31-v32`: shallow-tree extremes.
- `v33-v34`: deep-tree extremes, one loose and one strongly constrained.
- `v35-v39`: asymmetric sampling and regularization tests.
- `v40`: very wide wildcard preset.

Preset-by-preset intent:

- `v31`: maximum conservatism.
  Extremely shallow, very low sampling, strong gamma and strong L1/L2.
  Tests whether strong decorrelation and heavy penalties are required for stable background shaping.
- `v32`: shallow but fast.
  Shallow trees with high learning rate and high sample usage.
  Tests whether the problem is mostly ranking-simple and benefits from fast coarse separation.
- `v33`: deep and slow.
  Many trees, low learning rate, deep structure, weak penalties.
  Tests whether the analysis is still underfitting fine nonlinear background structure.
- `v34`: deep but tightly constrained.
  Same general deep-tree direction as `v33`, but with high child-weight and strong penalties.
  Tests whether deep trees help only if node growth is heavily controlled.
- `v35`: medium depth with extremely low row and column sampling.
  Tests whether strong stochasticity removes mass-correlated local structures more effectively.
- `v36`: fast deep learner with reduced sampling.
  Tests an intentionally aggressive background-suppression regime.
- `v37`: near-full row usage but very low column usage.
  Tests whether feature-level randomness matters more than event-level randomness for this 4-variable setup.
- `v38`: high child-weight and large L2 with near-full sampling.
  Tests whether suppressing local fluctuations while keeping sample statistics high improves the post-cut mass shape.
- `v39`: deep trees with low row sampling but high column retention.
  Tests whether tree depth plus row stochasticity is a useful compromise.
- `v40`: wide wildcard preset.
  Covers a broad mixed region to catch useful combinations missed by the more theme-driven presets.

What this batch mainly tested:

- Whether any single hyperparameter axis is the main limiter: depth, sampling, regularization, or learning rate.
- Whether asymmetric sampling patterns can improve the background shape more than the symmetric sampling ranges used before.
- Whether more extreme presets outperform the previously balanced searches in terms of validation `S/sqrt(S+B)` and eventual mass-spectrum behavior.
