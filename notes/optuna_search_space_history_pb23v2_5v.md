# Optuna search-space history for pb23v2_5v training batches

This file records the search-space design used for the `pb23v2_5v` Condor training line.

## Batch summary

- `pb23v2_5v_o100_v1-v40`: same 40-preset family previously developed for the `pb23v2_4v2` line, now rerun with one additional physics input variable.

## Variable-set change

Compared with the earlier `pb23v2_4v2` training line, this `5v` line adds:

- `Btrk2dR`

So the full input list is:

- `Btrk1dR`
- `Btrk2dR`
- `Btrk2Pt`
- `BtrkPtimb`
- `Bchi2Prob`

## Search-space family

This batch keeps the same preset philosophy as the earlier `pb23v2_4v2` line:

- `v1-v10`: original broad mixed scan
- `v11-v20`: robust versus aggressive redesign
- `v21-v30`: wider `o100` exploration
- `v31-v40`: more extreme and asymmetric scans

The purpose of rerunning them under `5v` is to test whether the added `Btrk2dR` information improves background suppression and post-cut mass-shape behavior without changing the established hyperparameter families.
