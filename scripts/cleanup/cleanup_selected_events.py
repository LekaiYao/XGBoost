#!/usr/bin/env python3
import argparse
import time
from pathlib import Path


def format_bytes(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(num_bytes)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{num_bytes} B"

def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Delete DATA_with_score.root and MC_with_score.root inside output/selected "
            "subdirectories older than N hours."
        )
    )
    parser.add_argument(
        "--selected-dir",
        default="output/selected",
        help="Path to selected output directory (default: output/selected)",
    )
    parser.add_argument(
        "--hours",
        type=int,
        default=120,
        help="Delete score ROOT files inside folders older than this many hours (default: 120)",
    )
    args = parser.parse_args()

    selected_dir = Path(args.selected_dir).resolve()
    if not selected_dir.is_dir():
        raise FileNotFoundError(f"selected directory not found: {selected_dir}")

    now = time.time()
    cutoff_ts = now - args.hours * 3600
    target_names = {"DATA_with_score.root", "MC_with_score.root"}

    deleted_files = []
    deleted_bytes = 0

    for child in sorted(selected_dir.iterdir()):
        if not child.is_dir():
            continue
        if child.stat().st_mtime >= cutoff_ts:
            continue

        for path in sorted(child.iterdir()):
            if not path.is_file():
                continue
            if path.name not in target_names:
                continue
            size = path.stat().st_size
            path.unlink()
            deleted_files.append(path)
            deleted_bytes += size

    print("=== Cleanup Summary ===")
    print(f"selected_dir: {selected_dir}")
    print(f"folders_older_than_{args.hours}h_scanned_for_score_roots")
    print(f"deleted_score_root_files: {len(deleted_files)}")
    for path in deleted_files:
        print(f"  deleted: {path.relative_to(selected_dir)}")
    print(f"freed_space_from_deleted_root_files: {deleted_bytes} bytes ({format_bytes(deleted_bytes)})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
