#!/usr/bin/env python3
import argparse
import shutil
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


def unique_destination(base_dir: Path, name: str) -> Path:
    dst = base_dir / name
    if not dst.exists():
        return dst
    stamp = time.strftime("%Y%m%d_%H%M%S")
    idx = 1
    while True:
        candidate = base_dir / f"{name}__moved_{stamp}_{idx}"
        if not candidate.exists():
            return candidate
        idx += 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Move selected_events subdirs older than N days to pre_obj, "
            "then delete .root files larger than threshold in pre_obj."
        )
    )
    parser.add_argument(
        "--selected-events-dir",
        default="selected_events",
        help="Path to selected_events directory",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=5,
        help="Move folders older than this many days (default: 5)",
    )
    parser.add_argument(
        "--root-threshold-mb",
        type=int,
        default=500,
        help="Delete .root files larger than this size in MB under pre_obj (default: 500)",
    )
    args = parser.parse_args()

    selected_events_dir = Path(args.selected_events_dir).resolve()
    pre_obj_dir = selected_events_dir / "pre_obj"

    if not selected_events_dir.is_dir():
        raise FileNotFoundError(f"selected_events directory not found: {selected_events_dir}")

    pre_obj_dir.mkdir(parents=True, exist_ok=True)

    now = time.time()
    cutoff_ts = now - args.days * 24 * 3600
    threshold_bytes = args.root_threshold_mb * 1024 * 1024

    moved_count = 0
    moved_names = []

    for child in sorted(selected_events_dir.iterdir()):
        if not child.is_dir():
            continue
        if child.name == "pre_obj":
            continue
        if child.stat().st_mtime >= cutoff_ts:
            continue

        dst = unique_destination(pre_obj_dir, child.name)
        shutil.move(str(child), str(dst))
        moved_count += 1
        moved_names.append((child.name, dst.name))

    deleted_files = 0
    deleted_bytes = 0

    for path in pre_obj_dir.rglob("*.root"):
        if not path.is_file():
            continue
        size = path.stat().st_size
        mtime = path.stat().st_mtime
        if size > threshold_bytes and mtime < cutoff_ts:
            path.unlink()
            deleted_files += 1
            deleted_bytes += size

    print("=== Cleanup Summary ===")
    print(f"selected_events: {selected_events_dir}")
    print(f"pre_obj: {pre_obj_dir}")
    print(f"moved_folders_older_than_{args.days}d: {moved_count}")
    if moved_names:
        for src_name, dst_name in moved_names:
            if src_name == dst_name:
                print(f"  moved: {src_name}")
            else:
                print(f"  moved: {src_name} -> {dst_name}")
    print(f"deleted_large_root_files_older_than_{args.days}d(>{args.root_threshold_mb}MB): {deleted_files}")
    print(f"freed_space_from_deleted_root_files: {deleted_bytes} bytes ({format_bytes(deleted_bytes)})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
