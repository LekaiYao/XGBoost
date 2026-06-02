#!/usr/bin/env python3
import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from configs.samples import (
    infer_channel_from_tag,
    infer_dataset_year,
    infer_fid_profile,
    infer_sample_from_tag,
    infer_selection_profile,
    resolve_apply_config,
    resolve_draw_config,
    resolve_fiducial_config,
    resolve_training_config,
)


DEFAULT_PUNZI_A = 2.0
DEFAULT_PUNZI_B = 5.0
DEFAULT_OUTPUT_DIR = "./opt_plots"
DEFAULT_REF_SCORE_CUT = 0.6

MASS_RANGE_DEFAULTS = {
    "Bu": "(Bmass > 5.05 && Bmass < 5.55)",
    "Bs": "(Bmass > 5.1 && Bmass < 5.7)",
    "Bd": "(Bmass > 5.1 && Bmass < 5.7)",
    "X": "(Bmass > 3.6 && Bmass < 4.0)",
}

BIN_WIDTH_DEFAULTS = {
    "Bu": 0.01,
    "Bs": 0.005,
    "Bd": 0.005,
    "X": 0.01,
}


def _mass_windows_from_background_selection(expr: str):
    if not expr:
        raise ValueError("background_selection is empty; cannot extract Bmass sidebands.")

    s = expr.replace("&&", " and ").replace("||", " or ")
    num = r"([0-9]*\.?[0-9]+)"
    pat_pairs = [
        # Python chained comparisons: a < Bmass < b / a > Bmass > b
        re.compile(rf"{num}\s*<\s*Bmass\s*<\s*{num}", re.IGNORECASE),
        re.compile(rf"{num}\s*>\s*Bmass\s*>\s*{num}", re.IGNORECASE),
        re.compile(rf"Bmass\s*>\s*{num}\s*(?:and)\s*Bmass\s*<\s*{num}", re.IGNORECASE),
        re.compile(rf"Bmass\s*<\s*{num}\s*(?:and)\s*Bmass\s*>\s*{num}", re.IGNORECASE),
        re.compile(rf"\(\s*Bmass\s*>\s*{num}\s*\)\s*(?:and)\s*\(\s*Bmass\s*<\s*{num}\s*\)", re.IGNORECASE),
        re.compile(rf"\(\s*Bmass\s*<\s*{num}\s*\)\s*(?:and)\s*\(\s*Bmass\s*>\s*{num}\s*\)", re.IGNORECASE),
    ]

    windows = []
    for pat in pat_pairs:
        for a, b in pat.findall(s):
            lo_f, hi_f = min(float(a), float(b)), max(float(a), float(b))
            if hi_f > lo_f:
                windows.append((lo_f, hi_f))

    uniq = []
    seen = set()
    for lo, hi in windows:
        key = (round(lo, 8), round(hi, 8))
        if key not in seen:
            seen.add(key)
            uniq.append((lo, hi))

    if not uniq:
        raise ValueError(f"No Bmass sideband windows found in background_selection: {expr}")
    return sorted(uniq, key=lambda x: x[0])


def _fmt_window(lo: float, hi: float) -> str:
    return f"(Bmass > {lo:.6g} && Bmass < {hi:.6g})"


def _to_root_expr(expr: str) -> str:
    out = expr
    out = re.sub(r"\band\b", "&&", out)
    out = re.sub(r"\bor\b", "||", out)
    out = re.sub(r"\bnot\b", "!", out)
    return out


def _build_profile_from_tag(train_tag: str):
    sample = infer_sample_from_tag(train_tag)
    channel = infer_channel_from_tag(train_tag)
    dataset_year = infer_dataset_year(train_tag, sample)
    selection_profile = infer_selection_profile(train_tag, sample)
    fid_profile = infer_fid_profile(train_tag, sample)

    training_cfg = resolve_training_config(sample, channel, dataset_year, selection_profile)
    apply_cfg = resolve_apply_config(sample, channel, dataset_year)
    draw_cfg = resolve_draw_config(sample, channel, dataset_year)
    fid_cfg = resolve_fiducial_config(sample, channel, fid_profile)

    windows = _mass_windows_from_background_selection(training_cfg["background_selection"])
    total_width = sum(hi - lo for lo, hi in windows)

    # Macro requires low/high sidebands. If only one sideband exists, set low side to zero-width.
    if len(windows) == 1:
        lo, hi = windows[0]
        sideband_low = _fmt_window(lo, lo)
        sideband_high = _fmt_window(lo, hi)
    else:
        sideband_low = _fmt_window(*windows[0])
        sideband_high = _fmt_window(*windows[-1])

    mc_tree_name = apply_cfg["mc"][0]["tree"]
    data_tree_name = apply_cfg["data"][0]["tree"]
    pre_cut = fid_cfg.get("expression")
    if not pre_cut:
        raise ValueError(f"fid profile '{fid_profile}' has empty expression.")
    pre_cut = _to_root_expr(pre_cut)

    if channel in ("Bu", "Bd"):
        fs_low, fs_high = 5.2, 5.36
    elif channel == "Bs":
        fs_low, fs_high = 5.3, 5.46
    elif channel == "X":
        if len(windows) < 2:
            raise ValueError(
                "X channel requires two Bmass sideband windows in background_selection to infer fsRegion."
            )
        # Use the gap between low and high sidebands as the signal-region proxy.
        fs_low = windows[0][1]
        fs_high = windows[-1][0]
        if fs_high <= fs_low:
            raise ValueError(
                f"Invalid inferred fsRegion for X channel: ({fs_low}, {fs_high}) from windows={windows}"
            )
    else:
        raise ValueError(f"Unsupported channel for optimization: {channel}")
    fs_region = _fmt_window(fs_low, fs_high)
    fs_width = fs_high - fs_low
    mass_range_expr = MASS_RANGE_DEFAULTS[channel]
    bin_width = BIN_WIDTH_DEFAULTS[channel]

    profile = {
        "system": "PbPb" if sample == "pbpb" else "pp",
        "dataPath": f"../../XGBoost/output/selected/{train_tag}/DATA_with_score.root",
        "mcPath": f"../../XGBoost/output/selected/{train_tag}/MC_with_score.root",
        "dataTreeName": data_tree_name,
        "mcTreeName": mc_tree_name,
        "scoreVar": "xgb_score",
        "preCut": pre_cut,
        "sidebandLow": sideband_low,
        "sidebandHigh": sideband_high,
        "mass_range": mass_range_expr,
        "bin_width": f"{bin_width:.8g}",
        "fsRegion": fs_region,
        "refScoreCut": f"{DEFAULT_REF_SCORE_CUT:.1f}",
        "signalWidth": f"{fs_width:.8g}",
        "sidebandWidth": f"{total_width:.8g}",
        "punziA": f"{DEFAULT_PUNZI_A:.1f}",
        "punziB": f"{DEFAULT_PUNZI_B:.1f}",
        "outputDir": DEFAULT_OUTPUT_DIR,
        "fileNamePattern": f"punzi_{train_tag}.pdf",
    }
    return profile


def _render_section(section_name: str, profile: dict) -> str:
    lines = [f"[{section_name}]"]
    order = [
        "system",
        "dataPath",
        "mcPath",
        "dataTreeName",
        "mcTreeName",
        "scoreVar",
        "preCut",
        "sidebandLow",
        "sidebandHigh",
        "mass_range",
        "bin_width",
        "fsRegion",
        "refScoreCut",
        "signalWidth",
        "sidebandWidth",
        "punziA",
        "punziB",
        "outputDir",
        "fileNamePattern",
    ]
    for key in order:
        lines.append(f"{key}={profile[key]}")
    lines.append("")
    return "\n".join(lines)


def _upsert_section(conf_path: Path, section_name: str, section_text: str):
    if conf_path.exists():
        content = conf_path.read_text()
    else:
        content = ""

    section_re = re.compile(
        rf"^\[{re.escape(section_name)}\]\n(?:.*\n)*?(?=^\[|\Z)",
        re.MULTILINE,
    )
    if section_re.search(content):
        new_content = section_re.sub(section_text + "\n", content)
    else:
        if content and not content.endswith("\n"):
            content += "\n"
        if content and not content.endswith("\n\n"):
            content += "\n"
        new_content = content + section_text + "\n"

    conf_path.write_text(new_content)


def _run_macro(selectioner_dir: Path, conf_name: str, profile_name: str, macro_name: str):
    cmd = [
        "root",
        "-l",
        "-b",
        "-q",
        f'{macro_name}("{conf_name}","{profile_name}")',
    ]
    subprocess.run(cmd, cwd=str(selectioner_dir), check=True)


def main():
    parser = argparse.ArgumentParser(
        description="Generate/update optimalCUT.conf profile from train_tag and optionally run ROOT macro."
    )
    parser.add_argument("train_tag", help="single DAG train tag, used directly as profile name")
    parser.add_argument(
        "--selectioner-dir",
        default="../Analysis_CODES/selectionER",
        help="Path to selectionER directory (default: ../Analysis_CODES/selectionER)",
    )
    parser.add_argument(
        "--conf-name",
        default="optimalCUT.conf",
        help="Config file name under selectionER dir",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Run both optimalCUT_punzi.C and optimalCUT_fom.C after updating config",
    )
    parser.add_argument(
        "--run_punzi",
        action="store_true",
        help="Run only optimalCUT_punzi.C after updating config",
    )
    parser.add_argument(
        "--run_fom",
        action="store_true",
        help="Run only optimalCUT_fom.C after updating config",
    )
    args = parser.parse_args()

    selectioner_dir = Path(args.selectioner_dir).resolve()
    conf_path = selectioner_dir / args.conf_name

    profile = _build_profile_from_tag(args.train_tag)
    section_text = _render_section(args.train_tag, profile)
    _upsert_section(conf_path, args.train_tag, section_text)

    print(f"Updated config: {conf_path}")
    print(f"Profile: [{args.train_tag}]")
    print(f"dataPath={profile['dataPath']}")
    print(f"mcPath={profile['mcPath']}")
    print(f"preCut={profile['preCut']}")
    print(f"sidebandLow={profile['sidebandLow']}")
    print(f"sidebandHigh={profile['sidebandHigh']}")
    print(f"mass_range={profile['mass_range']}, bin_width={profile['bin_width']}")
    print(f"fsRegion={profile['fsRegion']}, refScoreCut={profile['refScoreCut']}")
    print(f"signalWidth={profile['signalWidth']}, sidebandWidth={profile['sidebandWidth']}")

    if args.run:
        _run_macro(selectioner_dir, args.conf_name, args.train_tag, "optimalCUT_punzi.C")
        _run_macro(selectioner_dir, args.conf_name, args.train_tag, "optimalCUT_fom.C")
        print("Optimization run completed: punzi + fom.")
    elif args.run_punzi:
        _run_macro(selectioner_dir, args.conf_name, args.train_tag, "optimalCUT_punzi.C")
        print("Optimization run completed: punzi.")
    elif args.run_fom:
        _run_macro(selectioner_dir, args.conf_name, args.train_tag, "optimalCUT_fom.C")
        print("Optimization run completed: fom.")


if __name__ == "__main__":
    main()
