#!/usr/bin/env python3
"""Generate a human-readable summary for every tag in output/selected."""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from configs.samples import (  # noqa: E402
    infer_channel_from_tag,
    infer_dataset_year,
    infer_fid_profile,
    infer_reweight_profile,
    infer_sample_from_tag,
    infer_selection_profile,
    resolve_fiducial_config,
    resolve_training_config,
    resolve_training_reweight_config,
)
from utils.varsets import get_varset_columns, infer_varset_from_tag  # noqa: E402


def load_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"WARNING: cannot read {path}: {exc}", file=sys.stderr)
        return {}
    return payload if isinstance(payload, dict) else {}


def one_line(value) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(", ", ": "))


def warn_difference(tag: str, field: str, recorded, current) -> None:
    if recorded is not None and current is not None and recorded != current:
        print(
            f"WARNING: {tag}: recorded {field} differs from the current resolver; "
            "the recorded training/apply value is reported.",
            file=sys.stderr,
        )


def resolve_current_tag(tag: str) -> dict:
    sample = infer_sample_from_tag(tag)
    channel = infer_channel_from_tag(tag)
    year = infer_dataset_year(tag, sample)
    selection_profile = infer_selection_profile(tag, sample)
    fid_profile = infer_fid_profile(tag, sample)
    varset = infer_varset_from_tag(tag, sample)
    reweight_profile = infer_reweight_profile(tag)
    training = resolve_training_config(sample, channel, year, selection_profile)
    fid = resolve_fiducial_config(sample, channel, fid_profile)
    variables = get_varset_columns(sample, varset, channel)
    reweight = resolve_training_reweight_config(
        sample,
        channel,
        year,
        reweight_profile,
        selection_profile,
        fid_profile,
    )
    return {
        "sample": sample,
        "channel": channel,
        "year": year,
        "selection_profile": selection_profile,
        "fid_profile": fid_profile,
        "varset": varset,
        "reweight_profile": reweight_profile,
        "signal_selection": training["signal_selection"],
        "background_selection": training["background_selection"],
        "fiducial_selection": fid["expression"],
        "variables": variables,
        "reweight": reweight,
    }


def numeric_constant(node: ast.AST):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if (
        isinstance(node, ast.UnaryOp)
        and isinstance(node.op, ast.USub)
        and isinstance(node.operand, ast.Constant)
        and isinstance(node.operand.value, (int, float))
    ):
        return -node.operand.value
    return None


def format_number(value) -> str:
    if value is None:
        return "inf"
    return f"{value:g}" if isinstance(value, float) else str(value)


def _operand(node: ast.AST):
    if isinstance(node, ast.Name):
        return "variable", node.id
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "abs"
        and len(node.args) == 1
        and isinstance(node.args[0], ast.Name)
    ):
        return "absolute", node.args[0].id
    value = numeric_constant(node)
    return ("number", value) if value is not None else None


def _merge_interval(clause: dict, variable: str, lower=None, upper=None) -> dict:
    merged = dict(clause)
    old_lower, old_upper = merged.get(variable, (None, None))
    if lower is not None:
        old_lower = lower if old_lower is None else max(old_lower, lower)
    if upper is not None:
        old_upper = upper if old_upper is None else min(old_upper, upper)
    if old_lower is not None and old_upper is not None and old_lower > old_upper:
        raise ValueError(f"Contradictory bounds for {variable}: {old_lower}, {old_upper}")
    merged[variable] = (old_lower, old_upper)
    return merged


def _comparison_interval(left: ast.AST, operator: ast.AST, right: ast.AST):
    left_value = _operand(left)
    right_value = _operand(right)
    if left_value is None or right_value is None:
        return None

    symbol = {
        ast.Gt: ">",
        ast.GtE: ">",
        ast.Lt: "<",
        ast.LtE: "<",
        ast.Eq: "=",
    }.get(type(operator))
    if symbol is None:
        return None
    inverse = {">": "<", "<": ">", "=": "="}

    if left_value[0] in ("variable", "absolute") and right_value[0] == "number":
        kind, variable = left_value
        value = right_value[1]
    elif right_value[0] in ("variable", "absolute") and left_value[0] == "number":
        kind, variable = right_value
        value = left_value[1]
        symbol = inverse[symbol]
    else:
        return None

    if kind == "absolute":
        if symbol == "<" and value >= 0:
            return variable, -value, value
        if symbol == "=" and value == 0:
            return variable, 0, 0
        return None
    if symbol == ">":
        return variable, value, None
    if symbol == "<":
        return variable, None, value
    return variable, value, value


def _expression_clauses(node: ast.AST) -> list[dict] | None:
    """Convert a selection to disjunctive interval clauses.

    Open/closed comparison boundaries intentionally share the same [min,max] display.
    """
    if isinstance(node, ast.BoolOp):
        children = [_expression_clauses(value) for value in node.values]
        if any(child is None for child in children):
            return None
        if isinstance(node.op, ast.Or):
            return [clause for child in children for clause in child]
        clauses = [{}]
        for child in children:
            combined = []
            for left_clause in clauses:
                for right_clause in child:
                    merged = dict(left_clause)
                    for variable, (lower, upper) in right_clause.items():
                        merged = _merge_interval(merged, variable, lower, upper)
                    combined.append(merged)
            clauses = combined
        return clauses

    if isinstance(node, ast.Compare):
        clause = {}
        operands = [node.left, *node.comparators]
        for left, operator, right in zip(operands, node.ops, operands[1:]):
            interval = _comparison_interval(left, operator, right)
            if interval is None:
                return None
            variable, lower, upper = interval
            clause = _merge_interval(clause, variable, lower, upper)
        return [clause]
    return None


def selection_clauses(expression: str) -> list[dict] | None:
    if not expression:
        return None
    normalized = expression.replace("&&", " and ").replace("||", " or ")
    try:
        return _expression_clauses(ast.parse(normalized, mode="eval").body)
    except (SyntaxError, ValueError):
        return None


def canonical_clauses(clauses: list[dict] | None):
    if clauses is None:
        return None
    return sorted(
        tuple(sorted((variable, lower, upper) for variable, (lower, upper) in clause.items()))
        for clause in clauses
    )


def format_interval(bounds: tuple) -> str:
    lower, upper = bounds
    low_text = f"-{format_number(None)}" if lower is None else format_number(lower)
    high_text = format_number(upper)
    return f"[{low_text},{high_text}]"


def format_clause_lines(clause: dict, indent="  ") -> list[str]:
    return [
        f"{indent}{variable}:{format_interval(bounds)}"
        for variable, bounds in clause.items()
    ]


def format_selection_block(label: str, expression: str) -> list[str]:
    clauses = selection_clauses(expression)
    lines = [f"{label}:"]
    if clauses is None:
        return lines + [f"  expression: {expression}"]
    if len(clauses) == 1:
        return lines + format_clause_lines(clauses[0])
    for index, clause in enumerate(clauses, start=1):
        lines.append(f"  branch {index}:")
        lines.extend(format_clause_lines(clause, indent="    "))
    return lines


def format_background_block(expression: str, signal_expression: str) -> list[str]:
    background = selection_clauses(expression)
    signal = selection_clauses(signal_expression)
    if background is None or signal is None or len(signal) != 1:
        return format_selection_block("background_selection", expression)

    signal_clause = signal[0]
    residuals = []
    for clause in background:
        residuals.append(
            {
                variable: bounds
                for variable, bounds in clause.items()
                if signal_clause.get(variable) != bounds
            }
        )
    lines = ["background_selection:", "  signal_selection &"]
    if len(residuals) == 1:
        if not residuals[0]:
            return ["background_selection: same as signal_selection"]
        return lines + format_clause_lines(residuals[0])

    variables = {variable for clause in residuals for variable in clause}
    if len(variables) == 1 and all(len(clause) == 1 for clause in residuals):
        variable = next(iter(variables))
        ordered = sorted(
            (clause[variable] for clause in residuals),
            key=lambda bounds: (
                float("-inf") if bounds[0] is None else bounds[0],
                float("inf") if bounds[1] is None else bounds[1],
            ),
        )
        ranges = " U ".join(format_interval(bounds) for bounds in ordered)
        return lines + [f"  {variable}:{ranges}"]
    for index, clause in enumerate(residuals, start=1):
        lines.append(f"  branch {index}:")
        lines.extend(format_clause_lines(clause, indent="    "))
    return lines


def find_reweight_manifest(signal_path: str) -> tuple[dict, Path | None]:
    if not signal_path:
        return {}, None
    file_part = signal_path.split(":", 1)[0]
    root_path = Path(file_part)
    if not root_path.is_absolute():
        root_path = REPO_ROOT / root_path
    directory = root_path.parent
    for name in ("reweighting_manifest.json", "manifest.json"):
        candidate = directory / name
        payload = load_json(candidate)
        if payload:
            return payload, candidate
    return {}, None


def signal_mc_description(current: dict, manifest: dict, signal_path: str, weight_branch) -> str:
    dataset = "PbPb" if current["sample"] == "pbpb" else "ppRef"
    year = str(current["year"])[-2:]
    if weight_branch is None:
        particle = "psi(2S)" if "PSI2S" in signal_path.upper() else current["channel"]
        return f"Unweighted {dataset}{year} {particle} MC"
    reweight_tag = manifest.get("reweight_tag")
    if reweight_tag:
        match = re.search(r"_xsplot_(.+?)_rw_v\d+$", reweight_tag)
        if match:
            return (
                f"{dataset}{year} {current['channel']} MC with ppRef-derived "
                f"{match.group(1)} weights"
            )
        return f"{dataset}{year} {current['channel']} MC with weights from {reweight_tag}"
    return f"{dataset}{year} {current['channel']} MC with the recorded reweighting profile"


def build_entry(tag: str, selected_dir: Path) -> list[str]:
    current = resolve_current_tag(tag)
    model_dir = REPO_ROOT / "output" / "models" / tag
    model_config = load_json(model_dir / "model_config.json")
    run_metadata = load_json(model_dir / "run_metadata.json")
    apply_summary = load_json(selected_dir / tag / "batch_apply_summary.json")

    recorded_input = apply_summary.get("input_selection", {})
    recorded_draw = apply_summary.get("draw_selection", {})
    recorded_varset = apply_summary.get("training_varset", {})
    notes = run_metadata.get("notes", {})

    signal_selection = (
        run_metadata.get("signal_selection")
        or recorded_input.get("signal_selection")
        or current["signal_selection"]
    )
    background_selection = (
        run_metadata.get("background_selection")
        or recorded_input.get("background_selection")
        or current["background_selection"]
    )
    fid_profile = (
        recorded_draw.get("fid_profile")
        or notes.get("fid_profile")
        or current["fid_profile"]
    )
    fiducial_selection = (
        recorded_draw.get("fiducial_cut", {}).get("expression")
        or current["fiducial_selection"]
    )
    varset = (
        run_metadata.get("feature_set_tag")
        or recorded_varset.get("varset_tag")
        or current["varset"]
    )
    variables = (
        model_config.get("input_columns")
        or run_metadata.get("input_columns")
        or recorded_varset.get("columns")
        or current["variables"]
    )

    warn_difference(tag, "signal_selection", signal_selection, current["signal_selection"])
    warn_difference(tag, "background_selection", background_selection, current["background_selection"])
    warn_difference(tag, "fid_profile", fid_profile, current["fid_profile"])
    warn_difference(tag, "fiducial_selection", fiducial_selection, current["fiducial_selection"])
    warn_difference(tag, "varset", varset, current["varset"])
    warn_difference(tag, "variables", variables, current["variables"])

    lines = [tag]
    lines.extend(format_selection_block("signal_selection", signal_selection))
    lines.extend(format_background_block(background_selection, signal_selection))
    fid_clauses = selection_clauses(fiducial_selection)
    signal_clauses = selection_clauses(signal_selection)
    if (
        fid_clauses is not None
        and signal_clauses is not None
        and canonical_clauses(fid_clauses) == canonical_clauses(signal_clauses)
    ):
        lines.append(f"{fid_profile}: same as signal_selection")
    else:
        lines.extend(format_selection_block(fid_profile, fiducial_selection))
    lines.append(f"{varset}: {one_line(variables)}")

    reweight_profile = model_config.get("reweight_profile") or current["reweight_profile"]
    if reweight_profile != "rw0":
        signal_path = model_config.get("signal_path") or run_metadata.get("signal_path") or ""
        weight_branch = (
            model_config.get("signal_weight_branch")
            or run_metadata.get("notes", {}).get("signal_weight_branch")
            or current["reweight"].get("weight_branch")
        )
        manifest, manifest_path = find_reweight_manifest(signal_path) if weight_branch else ({}, None)
        if weight_branch and not manifest:
            print(
                f"WARNING: {tag}: no reweighting manifest found for {signal_path}",
                file=sys.stderr,
            )
        reweight_selection = manifest.get("selection", "")
        lines.append(f"{reweight_profile}:")
        lines.append(
            "  signal_MC: "
            + signal_mc_description(current, manifest, signal_path, weight_branch)
        )
        if weight_branch:
            reweight_lines = format_selection_block(
                "reweighting_range", reweight_selection
            )
            lines.extend(f"  {line}" for line in reweight_lines)
        else:
            lines.append("  reweighting_range: not applicable")
        final_lines = format_selection_block("final_ML_range", signal_selection)
        lines.extend(f"  {line}" for line in final_lines)
        if manifest_path:
            warn_difference(
                tag,
                "reweight profile",
                reweight_profile,
                current["reweight_profile"],
            )
    return lines


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--selected-dir",
        type=Path,
        default=REPO_ROOT / "output" / "selected",
        help="Directory whose immediate subdirectories are summarized.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "docs" / "selected_tag_summary.md",
        help="Generated Markdown document.",
    )
    parser.add_argument(
        "--tag",
        action="append",
        help="Summarize only this tag. May be supplied more than once.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    selected_dir = args.selected_dir.resolve()
    output_path = args.output.resolve()
    if not selected_dir.is_dir():
        raise SystemExit(f"Selected directory does not exist: {selected_dir}")

    available_tags = {path.name for path in selected_dir.iterdir() if path.is_dir()}
    if args.tag:
        missing = sorted(set(args.tag) - available_tags)
        if missing:
            raise SystemExit(f"Selected tag directories do not exist: {missing}")
        tags = sorted(set(args.tag))
    else:
        tags = sorted(available_tags)
    entries = []
    failures = []
    for tag in tags:
        try:
            entries.append("\n".join(build_entry(tag, selected_dir)))
        except (KeyError, TypeError, ValueError) as exc:
            failures.append((tag, str(exc)))
            print(f"ERROR: {tag}: {exc}", file=sys.stderr)

    if failures:
        raise SystemExit(
            f"Refusing to write an incomplete summary: {len(failures)} of {len(tags)} tags failed."
        )

    header = (
        "# Selected-tag summary\n\n"
        "Generated by `python3 scripts/summarize_selected_tags.py`. Do not edit manually.\n\n"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(header + "\n\n".join(entries) + "\n", encoding="utf-8")
    print(f"Wrote {len(entries)} tags to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
