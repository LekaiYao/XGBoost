#!/usr/bin/env python3
import argparse
import json
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Set

OUTPUT_CATEGORIES = ("models", "training", "selected", "shap")
CHANNEL_RE = r"(X|Bu|Bd|Bs)"
DATASET_RE = r"(pp24|pb23|pb24)"
VARSET_RE = r"([0-9]+v[0-9]+)"
SINGLE_TAG_RE = re.compile(
    rf"^{CHANNEL_RE}_{DATASET_RE}_v(\d+)_fid(\d+)_{VARSET_RE}_xgb_v(\d+)$"
)
OPTUNA_TAG_RE = re.compile(
    rf"^{CHANNEL_RE}_{DATASET_RE}_v(\d+)_fid(\d+)_{VARSET_RE}_(\d+)o(\d+)_v(\d+)$"
)


def format_bytes(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(num_bytes)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{num_bytes} B"


def collect_train_tags(output_dir: Path) -> Set[str]:
    tags: Set[str] = set()
    for category in OUTPUT_CATEGORIES:
        root = output_dir / category
        if not root.is_dir():
            continue
        for p in root.iterdir():
            if p.is_dir():
                tags.add(p.name)
    return tags


def match_workflow(tag: str, workflow_type: str) -> bool:
    if workflow_type == "single":
        return bool(SINGLE_TAG_RE.fullmatch(tag))
    if workflow_type == "optuna":
        return bool(OPTUNA_TAG_RE.fullmatch(tag))
    return bool(SINGLE_TAG_RE.fullmatch(tag) or OPTUNA_TAG_RE.fullmatch(tag))


def move_dir(src: Path, dst: Path, dry_run: bool) -> bool:
    if not src.exists():
        return False
    if dry_run:
        return True
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst))
    return True


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Archive output/{models,training,selected,shap} into output/backup_outdate/<timestamp>. "
            "Before moving selected/<tag>, delete selected/<tag>/DATA_with_score.root."
        )
    )
    parser.add_argument("--output-dir", default="output", help="Output root (default: output)")
    parser.add_argument(
        "--backup-subdir",
        default="backup_outdate",
        help="Backup folder name under output root (default: backup_outdate)",
    )
    parser.add_argument(
        "--delete-selected-data-root",
        action="store_true",
        default=True,
        help="Delete selected/<tag>/DATA_with_score.root before moving (default: enabled)",
    )
    parser.add_argument(
        "--keep-selected-data-root",
        action="store_true",
        help="Do not delete selected/<tag>/DATA_with_score.root before moving",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print operations only")
    parser.add_argument(
        "--workflow-type",
        choices=("all", "single", "optuna"),
        default="all",
        help="Archive only selected workflow tags by naming rule (default: all)",
    )
    parser.add_argument(
        "--afs-dir",
        default="/afs/cern.ch/user/l/leyao/private/pbpb_work/X_analysis/XGBoost",
        help="AFS submit root used by clear_dag_locks.sh",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    if not output_dir.is_dir():
        raise FileNotFoundError(f"output directory not found: {output_dir}")

    backup_root = output_dir / args.backup_subdir
    stamp = time.strftime("%Y%m%d_%H%M%S")
    backup_dir = backup_root / stamp

    all_tags = sorted(collect_train_tags(output_dir))
    train_tags = [t for t in all_tags if match_workflow(t, args.workflow_type)]
    skipped_tags = [t for t in all_tags if t not in train_tags]
    moved = []
    removed_data_roots = []
    removed_bytes = 0
    lock_cleanup_results = {}
    lock_cleanup_script = Path(__file__).resolve().parent / "clear_dag_locks.sh"

    delete_selected_data_root = args.delete_selected_data_root and not args.keep_selected_data_root

    for tag in train_tags:
        lock_cmd = [str(lock_cleanup_script)]
        if args.dry_run:
            lock_cmd.append("--dry-run")
        lock_cmd.extend([tag, args.afs_dir])
        proc = subprocess.run(lock_cmd, check=False, capture_output=True, text=True)
        lock_cleanup_results[tag] = {
            "returncode": proc.returncode,
            "stdout": proc.stdout.strip(),
            "stderr": proc.stderr.strip(),
        }

        selected_data_root = output_dir / "selected" / tag / "DATA_with_score.root"
        if delete_selected_data_root and selected_data_root.exists() and selected_data_root.is_file():
            size = selected_data_root.stat().st_size
            removed_data_roots.append(str(selected_data_root.relative_to(output_dir)))
            removed_bytes += size
            if not args.dry_run:
                selected_data_root.unlink()

        for category in OUTPUT_CATEGORIES:
            src = output_dir / category / tag
            dst = backup_dir / category / tag
            if move_dir(src, dst, args.dry_run):
                moved.append((str(src.relative_to(output_dir)), str(dst.relative_to(output_dir))))

    summary = {
        "output_dir": str(output_dir),
        "backup_dir": str(backup_dir),
        "dry_run": args.dry_run,
        "workflow_type": args.workflow_type,
        "train_tags_total_detected": len(all_tags),
        "train_tags_total_matched": len(train_tags),
        "train_tags_to_move": train_tags,
        "train_tags_skipped": skipped_tags,
        "removed_data_with_score_count": len(removed_data_roots),
        "removed_data_with_score_bytes": removed_bytes,
        "removed_data_with_score_human": format_bytes(removed_bytes),
        "removed_data_with_score_files": removed_data_roots,
        "moved_entries": moved,
        "dag_resubmit_cleanup": {
            "script": str(lock_cleanup_script),
            "afs_dir": args.afs_dir,
            "results": lock_cleanup_results,
        },
    }

    print("=== Archive Summary ===")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
