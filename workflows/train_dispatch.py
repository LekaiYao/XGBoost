import argparse
import os
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("train_tag")
    parser.add_argument("optuna_n_trials", type=int)
    parser.add_argument("stage_group", nargs="?", default="")
    parser.add_argument("resume_flag", nargs="?", default="0")
    parser.add_argument("dataset_year", nargs="?", default="")
    parser.add_argument("selection_profile", nargs="?", default="")
    args = parser.parse_args()

    dataset_year = "" if args.dataset_year == "__EMPTY__" else args.dataset_year
    selection_profile = "" if args.selection_profile == "__EMPTY__" else args.selection_profile

    env = os.environ.copy()
    env["OPTUNA_N_TRIALS"] = str(args.optuna_n_trials)

    if "_xgb_" in args.train_tag:
        cmd = [sys.executable, "-m", "workflows.xgboost_train_direct", args.train_tag]
        if dataset_year:
            cmd += ["--dataset-year", dataset_year]
        if selection_profile:
            cmd += ["--selection-profile", selection_profile]
    elif args.stage_group.startswith("v") or args.stage_group.startswith("3v"):
        cmd = [sys.executable, "workflows/condor_optuna_XGBoost.py", args.train_tag]
        if dataset_year:
            cmd += ["--dataset-year", dataset_year]
        if selection_profile:
            cmd += ["--selection-profile", selection_profile]
    else:
        cmd = [sys.executable, "workflows/staged_optuna_pipeline.py", args.train_tag]
        if args.stage_group:
            cmd += ["--stage-group", args.stage_group]
        if args.resume_flag == "1":
            cmd.append("--resume")
        if dataset_year:
            cmd += ["--dataset-year", dataset_year]

    raise SystemExit(subprocess.call(cmd, env=env))


if __name__ == "__main__":
    main()
