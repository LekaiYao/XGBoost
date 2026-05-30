import argparse
import os
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("train_tag")
    parser.add_argument("optuna_n_trials", type=int)
    args = parser.parse_args()

    env = os.environ.copy()
    env["OPTUNA_N_TRIALS"] = str(args.optuna_n_trials)

    if "_xgb_" in args.train_tag:
        cmd = [sys.executable, "-m", "workflows.xgboost_train_direct", args.train_tag]
    else:
        cmd = [sys.executable, "-m", "workflows.condor_optuna_XGBoost", args.train_tag]

    raise SystemExit(subprocess.call(cmd, env=env))


if __name__ == "__main__":
    main()
