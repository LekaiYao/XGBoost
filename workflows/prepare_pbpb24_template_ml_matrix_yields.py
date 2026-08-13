#!/usr/bin/env python3
"""Prepare the confirmed weighted/unit-template x weighted/unweighted-ML injection contract."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import uproot

from configs.samples import infer_fid_profile, infer_sample_from_tag, resolve_fiducial_config
from utils.selection import apply_selection, selection_columns


RHO_PBPB = 1.08
TARGETS = (0.2, 0.3, 0.4, 0.5)
TEMPLATE_MODES = ("weighted", "unweighted")
REFERENCE_TREE = "ntmix_X3872"
REFERENCE_WEIGHT = "Reweight"
FIT_RANGE = (3.8, 3.94)
BIN_WIDTH = 0.005


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def reference_efficiencies(path: Path, selection: str, threshold: float) -> dict[str, dict]:
    columns = list(dict.fromkeys(["Prediction", REFERENCE_WEIGHT, *selection_columns(selection)]))
    with uproot.open(path) as root_file:
        tree = root_file[REFERENCE_TREE]
        missing = [name for name in columns if name not in tree.keys()]
        if missing:
            raise ValueError(f"Missing branches in {path}: {missing}")
        frame = tree.arrays(columns, library="pd")
        entries = int(tree.num_entries)
    frame = apply_selection(frame, selection, "fiducial selection")
    scores = frame["Prediction"].to_numpy(dtype=float)
    weights = frame[REFERENCE_WEIGHT].to_numpy(dtype=float)
    if not np.all(np.isfinite(scores)) or not np.all(np.isfinite(weights)) or np.any(weights <= 0):
        raise ValueError(f"Invalid score/weight values in {path}")
    passed = scores > threshold
    return {
        "weighted": {
            "efficiency": float(np.sum(weights[passed]) / np.sum(weights)),
            "denominator": float(np.sum(weights)),
            "numerator": float(np.sum(weights[passed])),
            "event_weight": REFERENCE_WEIGHT,
        },
        "unweighted": {
            "efficiency": float(np.count_nonzero(passed) / len(scores)),
            "denominator": int(len(scores)),
            "numerator": int(np.count_nonzero(passed)),
            "event_weight": "unit",
        },
        "entries_total": entries,
        "entries_fiducial": int(len(scores)),
    }


def prepare(h010_summary: Path, output_dir: Path) -> Path:
    h010_rows = json.loads(h010_summary.read_text())
    if len(h010_rows) != 8:
        raise ValueError(f"Expected 8 H010 rows, got {len(h010_rows)}")
    points = []
    references = {}
    for h010 in h010_rows:
        tag = h010["train_tag"]
        target = round(float(h010["target_x_efficiency"]), 10)
        if target not in TARGETS:
            raise ValueError(f"Unexpected target: {target}")
        sample = infer_sample_from_tag(tag)
        fid_profile = infer_fid_profile(tag, sample)
        fid_selection = resolve_fiducial_config(sample, "X", fid_profile)["expression"]
        reference = Path("output/selected") / tag / "REFERENCE_MC_with_score.root"
        references.setdefault(
            tag,
            {"path": str(reference), "tree": REFERENCE_TREE, "sha256": sha256(reference)},
        )
        threshold = float(h010["score_threshold"])
        efficiencies = reference_efficiencies(reference, fid_selection, threshold)
        psi_yield = float(h010["raw_yield"])
        psi_error = float(h010["raw_yield_error"])
        psi_efficiency = float(h010["psi2s_score_efficiency"])
        pre_score_x = psi_yield * RHO_PBPB / psi_efficiency
        pre_score_error = psi_error * RHO_PBPB / psi_efficiency
        for template in TEMPLATE_MODES:
            x_efficiency = efficiencies[template]["efficiency"]
            central = pre_score_x * x_efficiency
            statistical = pre_score_error * x_efficiency
            points.append(
                {
                    "key": f"{template}_template__{h010['key']}",
                    "template_type": template,
                    "ml_type": h010["model_type"],
                    "train_tag": tag,
                    "target_x_efficiency": target,
                    "score_threshold": threshold,
                    "fiducial_selection": fid_selection,
                    "comparison": "Prediction > score_threshold",
                    "template_event_weight": efficiencies[template]["event_weight"],
                    "template_score_efficiency": x_efficiency,
                    "template_efficiency_numerator": efficiencies[template]["numerator"],
                    "template_efficiency_denominator": efficiencies[template]["denominator"],
                    "reference_entries_total": efficiencies["entries_total"],
                    "reference_entries_fiducial": efficiencies["entries_fiducial"],
                    "psi2s_raw_yield": psi_yield,
                    "psi2s_raw_yield_fit_error": psi_error,
                    "psi2s_score_efficiency": psi_efficiency,
                    "expected_x_pre_score": pre_score_x,
                    "expected_x_pre_score_psi_fit_error": pre_score_error,
                    "expected_x_after_score": central,
                    "expected_x_after_score_psi_fit_error": statistical,
                    "asimov_injection_yields": {
                        "background_only": 0.0,
                        "psi_fit_minus_1sigma": max(0.0, central - statistical),
                        "central": central,
                        "psi_fit_plus_1sigma": central + statistical,
                    },
                    "data_path": h010["data_path"],
                    "data_tree": h010["data_tree"],
                    "reference_signal_key": tag,
                }
            )

    expected = {(t, m, e) for t in TEMPLATE_MODES for m in ("weighted", "unweighted") for e in TARGETS}
    actual = {(p["template_type"], p["ml_type"], p["target_x_efficiency"]) for p in points}
    if actual != expected or len(points) != 16:
        raise ValueError(f"Incomplete template/ML/target grid: {actual}")
    payload = {
        "schema_version": 2,
        "status": "confirmed_template_ml_matrix_fast_test",
        "definition": "N_X_after = (N_psi_raw * rho_PbPb / epsilon_psi_score) * epsilon_X(template,ML)",
        "physics_inputs": {
            "rho_pbpb_central": RHO_PBPB,
            "pre_score_acceptance_efficiency_ratio_x_over_psi2s": 1.0,
            "prompt_fraction_ratio_psi2s_over_x": 1.0,
            "propagated_uncertainty": "H010 psi(2S) raw-yield fit uncertainty only",
            "rho_uncertainty_propagated": False,
        },
        "fit_contract": {
            "mass_range": list(FIT_RANGE),
            "bin_width": BIN_WIDTH,
            "mass_bins": int(round((FIT_RANGE[1] - FIT_RANGE[0]) / BIN_WIDTH)),
            "generation_and_fit_use_matching_template": True,
        },
        "toy_plan": {
            "toys_per_ensemble": 200,
            "ensembles_per_point": ["background_only", "central"],
            "asimov_scenarios": ["background_only", "psi_fit_minus_1sigma", "central", "psi_fit_plus_1sigma"],
            "matrix_points": 16,
            "total_asimov": 64,
            "total_toys": 6400,
        },
        "h010_source": {"path": str(h010_summary), "sha256": sha256(h010_summary)},
        "reference_signals": references,
        "points": sorted(points, key=lambda p: (p["template_type"], p["ml_type"], p["target_x_efficiency"])),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "expected_yields.json"
    output.write_text(json.dumps(payload, indent=2) + "\n")
    manifest = {
        "schema_version": 2,
        "status": "validated",
        "expected_yields": str(output),
        "expected_yields_sha256": sha256(output),
        "matrix_point_count": 16,
        "template_types": list(TEMPLATE_MODES),
        "ml_types": ["weighted", "unweighted"],
        "target_x_efficiencies": list(TARGETS),
        "mass_range": list(FIT_RANGE),
        "mass_bins": 28,
        "notes": [
            "Weighted and unit templates use the same event base; only event weights differ.",
            "Each generation template is fitted with its matching signal template.",
            "No injection or fit is performed by this producer.",
        ],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h010-summary", type=Path, default=Path("../Analysis_CODES/fitER/results/PbPb_H010_score_working_points/fit_summary.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("output/signal_injection/pbpb24_template_ml_matrix_v2"))
    args = parser.parse_args()
    print(prepare(args.h010_summary, args.output_dir))


if __name__ == "__main__":
    main()
