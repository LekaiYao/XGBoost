#!/usr/bin/env python3
"""Prepare the confirmed psi(2S)-anchored PbPb24 X injection-yield contract."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import uproot


RHO_PBPB = 1.08
TARGETS = (0.2, 0.3, 0.4, 0.5)
REFERENCE_TREE = "ntmix_X3872"
REFERENCE_WEIGHT = "Reweight"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_x_efficiencies(path: Path) -> dict[float, dict]:
    payload = json.loads(path.read_text())
    return {
        round(float(row["target_efficiency"]), 10): row
        for row in payload["thresholds"]
        if round(float(row["target_efficiency"]), 10) in TARGETS
    }


def validate_reference(path: Path) -> dict:
    with uproot.open(path) as root_file:
        tree = root_file[REFERENCE_TREE]
        missing = [name for name in ("Bmass", "Prediction", REFERENCE_WEIGHT) if name not in tree.keys()]
        if missing:
            raise ValueError(f"Missing reference branches in {path}: {missing}")
        arrays = tree.arrays(["Bmass", "Prediction", REFERENCE_WEIGHT], library="np")
        entries = int(tree.num_entries)
    for name, values in arrays.items():
        if not np.all(np.isfinite(values)):
            raise ValueError(f"Non-finite {name} in {path}")
    if np.any(arrays[REFERENCE_WEIGHT] <= 0):
        raise ValueError(f"Non-positive {REFERENCE_WEIGHT} in {path}")
    return {
        "path": str(path),
        "tree": REFERENCE_TREE,
        "weight_branch": REFERENCE_WEIGHT,
        "entries": entries,
        "sha256": sha256(path),
        "weight_sum": float(np.sum(arrays[REFERENCE_WEIGHT], dtype=np.float64)),
        "weight_min": float(np.min(arrays[REFERENCE_WEIGHT])),
        "weight_max": float(np.max(arrays[REFERENCE_WEIGHT])),
    }


def prepare(h010_summary: Path, output_dir: Path) -> Path:
    rows = json.loads(h010_summary.read_text())
    if len(rows) != 8:
        raise ValueError(f"Expected 8 H010 rows, got {len(rows)}")

    output_rows = []
    references = {}
    for row in rows:
        tag = row["train_tag"]
        target = round(float(row["target_x_efficiency"]), 10)
        if target not in TARGETS:
            raise ValueError(f"Unexpected X target efficiency: {target}")
        threshold_path = Path("output/selected") / tag / "cut_scan/weighted_signal_efficiency/thresholds.json"
        x_row = load_x_efficiencies(threshold_path).get(target)
        if x_row is None:
            raise ValueError(f"Missing X efficiency {target} for {tag}")
        if abs(float(x_row["score_threshold"]) - float(row["score_threshold"])) > 1e-12:
            raise ValueError(f"H010/threshold score mismatch for {row['key']}")

        reference_path = Path("output/selected") / tag / "REFERENCE_MC_with_score.root"
        references.setdefault(tag, validate_reference(reference_path))
        psi_yield = float(row["raw_yield"])
        psi_error = float(row["raw_yield_error"])
        psi_efficiency = float(row["psi2s_score_efficiency"])
        x_efficiency = float(x_row["achieved_efficiency"])
        scale = RHO_PBPB * x_efficiency / psi_efficiency
        central = psi_yield * scale
        statistical = psi_error * scale
        scenarios = {
            "background_only": 0.0,
            "psi_fit_minus_1sigma": max(0.0, central - statistical),
            "central": central,
            "psi_fit_plus_1sigma": central + statistical,
        }
        output_rows.append(
            {
                "key": row["key"],
                "model_type": row["model_type"],
                "train_tag": tag,
                "target_x_efficiency": target,
                "score_threshold": float(row["score_threshold"]),
                "x_achieved_weighted_score_efficiency": x_efficiency,
                "psi2s_score_efficiency": psi_efficiency,
                "psi2s_raw_yield": psi_yield,
                "psi2s_raw_yield_fit_error": psi_error,
                "yield_scale_factor": scale,
                "expected_x_after_score": central,
                "expected_x_after_score_psi_fit_error": statistical,
                "asimov_injection_yields": scenarios,
                "selection": row["selection"],
                "data_path": row["data_path"],
                "data_tree": row["data_tree"],
                "threshold_source": str(threshold_path),
                "threshold_source_sha256": sha256(threshold_path),
                "reference_signal_key": tag,
                "h010_diagnostic_flags": row.get("diagnostic_flags", []),
            }
        )

    keys = {(row["model_type"], row["target_x_efficiency"]) for row in output_rows}
    expected_keys = {(model, target) for model in ("weighted", "unweighted") for target in TARGETS}
    if keys != expected_keys:
        raise ValueError(f"Incomplete model/target grid: {keys}")

    payload = {
        "schema_version": 1,
        "status": "fast_test_assumptions_confirmed",
        "definition": "N_X_after = N_psi_raw * rho_PbPb * epsilon_X_score / epsilon_psi_score",
        "physics_inputs": {
            "rho_pbpb_central": RHO_PBPB,
            "pre_score_acceptance_efficiency_ratio_x_over_psi2s": 1.0,
            "prompt_fraction_ratio_psi2s_over_x": 1.0,
            "propagated_uncertainty": "H010 psi(2S) raw-yield fit uncertainty only",
            "rho_uncertainty_propagated": False,
        },
        "toy_plan": {
            "fast_stage_toys_per_point": 200,
            "toy_scenarios": ["background_only", "central"],
            "asimov_scenarios": [
                "background_only",
                "psi_fit_minus_1sigma",
                "central",
                "psi_fit_plus_1sigma",
            ],
            "expanded_stage_toys_per_point": 1000,
            "expanded_stage_requires_new_user_confirmation": True,
        },
        "h010_source": {"path": str(h010_summary), "sha256": sha256(h010_summary)},
        "reference_signals": references,
        "working_points": output_rows,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "expected_yields.json"
    output_path.write_text(json.dumps(payload, indent=2) + "\n")
    manifest = {
        "schema_version": 1,
        "status": "validated",
        "expected_yields": str(output_path),
        "expected_yields_sha256": sha256(output_path),
        "working_point_count": len(output_rows),
        "model_types": ["weighted", "unweighted"],
        "target_x_efficiencies": list(TARGETS),
        "notes": [
            "Fast-test assumptions only; not a final physics expectation.",
            "Reference X templates use the common reweighted signal and Reweight branch.",
            "No signal injection or fit is performed by this producer.",
        ],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--h010-summary",
        type=Path,
        default=Path("../Analysis_CODES/fitER/results/PbPb_H010_score_working_points/fit_summary.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/signal_injection/pbpb24_ratio_psi2s_anchor_v1"),
    )
    args = parser.parse_args()
    print(prepare(args.h010_summary, args.output_dir))


if __name__ == "__main__":
    main()
