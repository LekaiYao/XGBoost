import argparse
import hashlib
import shlex
import subprocess
import sys
from pathlib import Path

from configs.samples import resolve_training_config, resolve_training_reweight_config
from utils.paths import reweighted_root_path, reweighting_dir
from utils.varsets import get_reweight_varset_columns


REPO_ROOT = Path(__file__).resolve().parents[2]

JOBS = {
    "Psi2S_pb23_R5_migrate_v1": {
        "source_reweight_tag": "Psi2S_pp24_R5_rw_v1",
        "channel": "Psi2S",
        "apply_sample": "pbpb",
        "apply_dataset_year": "2023",
        "apply_selection_profile": "pb23_v2",
        "apply_fid_profile": "pb23_fid2",
        "training_reweight_profile": "rwr5v1",
    },
    "Psi2S_pb23_R6range4_migrate_v1": {
        "source_reweight_tag": "Psi2S_pp24_R6range4_rw_v1",
        "channel": "Psi2S",
        "apply_sample": "pbpb",
        "apply_dataset_year": "2023",
        "apply_selection_profile": "pb23_v1",
        "apply_fid_profile": "pb23_fid1",
        "training_reweight_profile": "rwr6range4v1",
    },
    "X_pb23_R6range5_migrate_v1": {
        "source_reweight_tag": "X_pp24_xsplot_R6range5_rw_v1",
        "channel": "X",
        "apply_sample": "pbpb",
        "apply_dataset_year": "2023",
        "apply_selection_profile": "pb23_v3",
        "apply_fid_profile": "pb23_fid3",
        "training_reweight_profile": "rwr6range5v1",
    },
    "X_pb23_R5v3_migrate_v1": {
        "source_reweight_tag": "X_pp24_xsplot_R5v3_rw_v1",
        "channel": "X",
        "apply_sample": "pbpb",
        "apply_dataset_year": "2023",
        "apply_selection_profile": "pb23_v4",
        "apply_fid_profile": "pb23_fid4",
        "training_reweight_profile": "rwr5v3v1",
    },
    "X_pp24_xsplot_R5v3_rw_v1": {
        "sample": "pp",
        "channel": "X",
        "dataset_year": "2024",
        "selection_profile": "pp24_v5",
        "variable_set": "R5v3",
        "validation_variable_set": "R8",
        "target_root": (
            "/eos/home-l/leyao/pbpb_work/X_analysis/Analysis_CODES/plotER/Validation/results/"
            "ppRef_X_r5_splot/SignalWeight_TTree_ppRef_X_r5_fiducial_splot_ntmix_X3872.root"
        ),
        "target_tree": "ntmix_X3872_sWeight",
        "target_weight_branch": "signal_sWeight",
        "apply_sample": "pbpb",
        "apply_dataset_year": "2024",
        "apply_selection_profile": "pb24_v20",
        "apply_fid_profile": "pb24_fid20",
        "training_reweight_profile": "rwr5v3v1",
    },
    "X_pp24_xsplot_R6range5_rw_v1": {
        "sample": "pp",
        "channel": "X",
        "dataset_year": "2024",
        "selection_profile": "pp24_v6",
        "variable_set": "R6",
        "validation_variable_set": "R8",
        "target_root": (
            "/eos/home-l/leyao/pbpb_work/X_analysis/Analysis_CODES/plotER/Validation/results/"
            "ppRef_X_r5_splot/SignalWeight_TTree_ppRef_X_r5_fiducial_splot_ntmix_X3872.root"
        ),
        "target_tree": "ntmix_X3872_sWeight",
        "target_weight_branch": "signal_sWeight",
        "apply_sample": "pbpb",
        "apply_dataset_year": "2024",
        "apply_selection_profile": "pb24_v19",
        "apply_fid_profile": "pb24_fid19",
        "training_reweight_profile": "rwr6range5v1",
    },
    "Psi2S_pp24_R6range4_rw_v1": {
        "sample": "pp",
        "channel": "Psi2S",
        "dataset_year": "2024",
        "selection_profile": "pp24_v1",
        "variable_set": "R6",
        "validation_variable_set": "R8",
        "target_root": (
            "/eos/home-l/leyao/pbpb_work/X_analysis/Analysis_CODES/plotER/Validation/WEIGHTS/"
            "SignalWeight_TTree_ppRef_ntmix_PSI2S_PSI2S_btrk2dr_v2.root"
        ),
        "target_tree": "ntmix_PSI2S_sWeight",
        "target_weight_branch": "signal_sWeight",
        "apply_sample": "pbpb",
        "apply_dataset_year": "2024",
        "apply_selection_profile": "pb24_v1",
        "apply_fid_profile": "pb24_fid1",
        "training_reweight_profile": "rwr6range4v1",
    },
    "Psi2S_pp24_R5_rw_v1": {
        "sample": "pp",
        "channel": "Psi2S",
        "dataset_year": "2024",
        "selection_profile": "pp24_v2",
        "variable_set": "R5",
        "validation_variable_set": "R8",
        "target_root": (
            "/eos/home-l/leyao/pbpb_work/X_analysis/Analysis_CODES/plotER/Validation/WEIGHTS/"
            "SignalWeight_TTree_ppRef_ntmix_PSI2S_PSI2S_btrk2dr_v2.root"
        ),
        "target_tree": "ntmix_PSI2S_sWeight",
        "target_weight_branch": "signal_sWeight",
        "apply_sample": "pbpb",
        "apply_dataset_year": "2024",
        "apply_selection_profile": "pb24_v2",
        "apply_fid_profile": "pb24_fid2",
        "training_reweight_profile": "rwr5v1",
    },
}


def resolve_training_job(reweight_tag):
    spec = JOBS.get(reweight_tag)
    if spec is None:
        raise ValueError(
            f"Unsupported configured reweight job '{reweight_tag}'. "
            f"Expected one of {tuple(JOBS)}."
        )
    if "source_reweight_tag" in spec:
        raise ValueError(f"Configured job '{reweight_tag}' is apply-only")
    return spec


def configured_apply_jobs(reweight_tag):
    training_spec = resolve_training_job(reweight_tag)
    jobs = []
    if "apply_dataset_year" in training_spec:
        jobs.append((reweight_tag, training_spec))
    jobs.extend(
        (job_tag, spec)
        for job_tag, spec in JOBS.items()
        if spec.get("source_reweight_tag") == reweight_tag
    )
    return sorted(jobs, key=lambda item: item[1]["apply_dataset_year"])


def validation_splot_spec(training_spec):
    return training_spec.get(
        "validation_splot",
        {
            "path": training_spec.get("target_root"),
            "tree": training_spec.get("target_tree"),
            "weight_branch": training_spec.get("target_weight_branch"),
        },
    )


def source_fingerprint():
    paths = [
        REPO_ROOT / "configs/samples.py",
        REPO_ROOT / "utils/varsets.py",
        REPO_ROOT / "workflows/reweighting/train_reweighter.py",
        Path(__file__).resolve(),
    ]
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.relative_to(REPO_ROOT).as_posix().encode())
        digest.update(path.read_bytes())
    head = subprocess.check_output(
        ["git", "rev-parse", "--short=8", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()
    return head, digest.hexdigest()[:12]


def build_train_command(reweight_tag):
    spec = resolve_training_job(reweight_tag)
    training = resolve_training_config(
        spec["sample"], spec["channel"], spec["dataset_year"], spec["selection_profile"]
    )
    get_reweight_varset_columns(
        spec["sample"], spec["variable_set"], spec["channel"]
    )
    get_reweight_varset_columns(
        spec["sample"], spec["validation_variable_set"], spec["channel"]
    )
    head, source_hash = source_fingerprint()
    source = training["signal"]
    return [
        sys.executable,
        "-m",
        "workflows.reweighting.train_reweighter",
        reweight_tag,
        "--original-root", source["path"],
        "--original-tree", source["tree"],
        "--target-root", spec["target_root"],
        "--target-tree", spec["target_tree"],
        "--target-weight-branch", spec["target_weight_branch"],
        "--selection", training["signal_selection"],
        "--variable-set", spec["variable_set"],
        "--validation-variable-set", spec["validation_variable_set"],
        "--sample", spec["sample"],
        "--channel", spec["channel"],
        "--n-folds", "5",
        "--random-state", "42",
        "--n-estimators", "40",
        "--learning-rate", "0.2",
        "--max-depth", "3",
        "--min-samples-leaf", "200",
        "--loss-regularization", "5.0",
        "--subsample", "0.8",
        "--physics-status",
        f"preliminary_{reweight_tag.lower()}_point_estimate_no_bootstrap_head{head}_source{source_hash}",
    ]


def build_apply_command(reweight_tag):
    spec = JOBS[reweight_tag]
    model_tag = spec.get("source_reweight_tag", reweight_tag)
    training = resolve_training_config(
        spec["apply_sample"], spec["channel"],
        spec["apply_dataset_year"], spec["apply_selection_profile"],
    )
    profile = resolve_training_reweight_config(
        spec["apply_sample"], spec["channel"], spec["apply_dataset_year"],
        spec["training_reweight_profile"], spec["apply_selection_profile"],
        spec["apply_fid_profile"],
    )
    source = training["signal"]
    expected_output = REPO_ROOT / reweighted_root_path(model_tag, source["path"])
    configured_output = REPO_ROOT / profile["signal"]["path"]
    if expected_output != configured_output:
        raise ValueError(
            f"Apply output/profile mismatch: {expected_output} != {configured_output}"
        )
    output_columns = [
        "Bmass", "Bchi2Prob", "Btrk1dR", "Btrk2dR", "BtrkPtimb",
        "Btrk1Pt", "Btrk2Pt", "BtktkvProb", "Bcos_dtheta", "Btktkpt",
        "Bpt", "By", "BQvalue",
    ]
    return [
        sys.executable,
        "-m",
        "workflows.reweighting.apply_reweighter",
        model_tag,
        "--input-root", source["path"],
        "--input-tree", source["tree"],
        "--output-tree", profile["signal"]["tree"],
        "--selection", training["signal_selection"],
        "--output-columns", ",".join(output_columns),
    ]


def main():
    parser = argparse.ArgumentParser(description="Run a frozen, configured reweighting job.")
    parser.add_argument("reweight_tag", choices=tuple(JOBS))
    parser.add_argument("--stage", choices=("train", "apply"), default="train")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    command = (
        build_train_command(args.reweight_tag)
        if args.stage == "train"
        else build_apply_command(args.reweight_tag)
    )
    spec = JOBS[args.reweight_tag]
    model_tag = spec.get("source_reweight_tag", args.reweight_tag)
    output_dir = REPO_ROOT / reweighting_dir(model_tag)
    if "target_root" in spec:
        target = Path(spec["target_root"])
        if not target.is_file():
            raise FileNotFoundError(f"Missing validated target artifact: {target}")
    if args.stage == "train" and output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite reweighting output: {output_dir}")
    if args.stage == "apply":
        output = REPO_ROOT / reweighted_root_path(
            model_tag,
            resolve_training_config(
                JOBS[args.reweight_tag]["apply_sample"],
                JOBS[args.reweight_tag]["channel"],
                JOBS[args.reweight_tag]["apply_dataset_year"],
                JOBS[args.reweight_tag]["apply_selection_profile"],
            )["signal"]["path"],
        )
        if output.exists() or output.with_suffix(".manifest.json").exists():
            raise FileExistsError(f"Refusing to overwrite reweighted output: {output}")
    if args.dry_run:
        print(shlex.join(command))
        return
    subprocess.run(command, cwd=REPO_ROOT, check=True)


if __name__ == "__main__":
    main()
