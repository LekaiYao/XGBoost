#!/usr/bin/env python3
import argparse
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.paths import shap_importance_fraction_path
from utils.varsets import infer_channel_from_tag, infer_sample_from_tag

VARSETS_PATH = Path(__file__).resolve().parents[1] / "utils" / "varsets.py"
DICT_NAME_MAP = {
    ("pbpb", "X"): "PBPB_VARSETS_X",
    ("pbpb", "Bu"): "PBPB_VARSETS_BU",
    ("pbpb", "Bd"): "PBPB_VARSETS_BD",
    ("pbpb", "Bs"): "PBPB_VARSETS_BS",
    ("pp", "X"): "PP_VARSETS_X",
    ("pp", "Bu"): "PP_VARSETS_BU",
    ("pp", "Bd"): "PP_VARSETS_BD",
    ("pp", "Bs"): "PP_VARSETS_BS",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create a new varset in utils/varsets.py from SHAP cumulative 95% features."
    )
    parser.add_argument("train_tag", help="Training tag with existing SHAP output")
    parser.add_argument("--varsets-path", default=str(VARSETS_PATH))
    return parser.parse_args()


def load_shap_top95_features(train_tag: str) -> list[str]:
    shap_path = Path(shap_importance_fraction_path(train_tag))
    if not shap_path.exists():
        raise FileNotFoundError(f"Missing SHAP importance fraction file: {shap_path}")

    with open(shap_path) as f:
        rows = json.load(f)

    if not rows:
        raise ValueError(f"Empty SHAP importance fraction file: {shap_path}")

    selected = []
    first_over_95 = None
    for row in rows:
        feature = row["feature"]
        cumulative_percent = float(row["cumulative_percent"])
        if cumulative_percent <= 95.0:
            selected.append(feature)
        elif first_over_95 is None:
            first_over_95 = feature
            break

    if first_over_95 is not None:
        selected.append(first_over_95)
    if not selected:
        selected.append(rows[0]["feature"])
    return selected


def resolve_dict_name(train_tag: str) -> str:
    sample = infer_sample_from_tag(train_tag)
    channel = infer_channel_from_tag(train_tag)
    try:
        return DICT_NAME_MAP[(sample, channel)]
    except KeyError as exc:
        raise ValueError(f"Unsupported sample/channel for tag '{train_tag}': {sample}/{channel}") from exc


def find_block_bounds(text: str, dict_name: str) -> tuple[int, int]:
    marker = f"{dict_name} = {{"
    start = text.find(marker)
    if start < 0:
        raise ValueError(f"Dictionary block '{dict_name}' not found in {VARSETS_PATH}")

    brace_start = text.find("{", start)
    depth = 0
    for idx in range(brace_start, len(text)):
        char = text[idx]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return brace_start, idx
    raise ValueError(f"Could not match braces for dictionary block '{dict_name}'")


def next_varset_name(block_text: str, feature_count: int) -> str:
    pattern = re.compile(rf'"{feature_count}v(\d+)"\s*:')
    versions = [int(match.group(1)) for match in pattern.finditer(block_text)]
    next_index = max(versions) + 1 if versions else 1
    return f"{feature_count}v{next_index}"


def insert_varset(varsets_path: Path, dict_name: str, varset_name: str, features: list[str]) -> None:
    text = varsets_path.read_text()
    brace_start, brace_end = find_block_bounds(text, dict_name)
    block_text = text[brace_start:brace_end + 1]

    if re.search(rf'"{re.escape(varset_name)}"\s*:', block_text):
        raise ValueError(f"Varset '{varset_name}' already exists in block '{dict_name}'")

    entry = f'    "{varset_name}": {json.dumps(features)},\n'
    updated_block = block_text[:-1] + entry + "}"
    updated_text = text[:brace_start] + updated_block + text[brace_end + 1:]
    varsets_path.write_text(updated_text)


def main():
    args = parse_args()
    train_tag = args.train_tag
    varsets_path = Path(args.varsets_path)

    features = load_shap_top95_features(train_tag)
    dict_name = resolve_dict_name(train_tag)
    text = varsets_path.read_text()
    brace_start, brace_end = find_block_bounds(text, dict_name)
    block_text = text[brace_start:brace_end + 1]
    varset_name = next_varset_name(block_text, len(features))

    insert_varset(varsets_path, dict_name, varset_name, features)

    print(f"train_tag={train_tag}")
    print(f"dict_name={dict_name}")
    print(f"new_varset={varset_name}")
    print(f"feature_count={len(features)}")
    print("features=")
    for feature in features:
        print(f"  - {feature}")


if __name__ == "__main__":
    main()
