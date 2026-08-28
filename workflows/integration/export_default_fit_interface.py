#!/usr/bin/env python3
"""Export the current paired-year fit interface after ML draw completion."""

from __future__ import annotations

import argparse
import fcntl
from pathlib import Path
from typing import Optional

from configs.samples import infer_channel_from_tag
from configs.year_pairings import resolve_year_pairing_for_tag
from utils.paths import resolve_model_config_path, selected_dir
from workflows.integration.export_psi2s_simultaneous_year_fit_manifest import (
    export_manifest as export_psi2s_manifest,
)
from workflows.integration.export_x_simultaneous_year_mc_shape_manifest import (
    export_manifest as export_x_manifest,
)


def _required_paths(train_tag: str) -> tuple[Path, ...]:
    selected = Path(selected_dir(train_tag)).resolve()
    return (
        Path(resolve_model_config_path(train_tag)).resolve(),
        selected / "DATA_with_score.root",
        selected / "MC_with_score.root",
        selected / "cut_scan/weighted_signal_efficiency/thresholds.json",
    )


def export_default_fit_interface(train_tag: str) -> Optional[Path]:
    pairing = resolve_year_pairing_for_tag(train_tag)
    if pairing is None:
        print(f"SKIP fit interface: no year pairing configured for {train_tag}")
        return None

    missing_current = [path for path in _required_paths(train_tag) if not path.is_file()]
    if missing_current:
        raise FileNotFoundError(
            f"Current tag is missing required post-draw artifacts: {missing_current}"
        )

    peer_tag = next(tag for tag in pairing["tags"].values() if tag != train_tag)
    missing_peer = [path for path in _required_paths(peer_tag) if not path.is_file()]
    if missing_peer:
        print(
            f"SKIP fit interface: paired tag {peer_tag} is not ready; "
            f"missing {missing_peer}"
        )
        return None

    anchor = pairing["anchor_train_tag"]
    output_directory = Path(selected_dir(anchor)).resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    lock_path = output_directory / ".fit_interface_export.lock"
    with lock_path.open("a+") as lock_stream:
        fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX)
        channel = infer_channel_from_tag(anchor)
        if channel == "X":
            output = export_x_manifest(anchor)
        elif channel == "Psi2S":
            output = export_psi2s_manifest(anchor)
        else:
            print(f"SKIP fit interface: unsupported paired channel {channel}")
            return None
    print(f"FIT_INTERFACE_READY {output}")
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("train_tag")
    args = parser.parse_args()
    export_default_fit_interface(args.train_tag)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
