"""Complete four-MC ROOT copies with Score and GEN plus fiducial DATA."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

import joblib
import numpy as np
import pandas as pd
import uproot

from configs.samples import (
    infer_sample_from_tag, infer_channel_from_tag, infer_dataset_year,
    infer_selection_profile, infer_fid_profile, resolve_training_config,
    resolve_fiducial_config, resolve_full_apply_config,
)
from utils.paths import (
    root_scored_dir, resolve_model_path, resolve_scaler_path,
    resolve_model_config_path,
)
from utils.selection import apply_selection, selection_columns


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("train_tag")
    scope = parser.add_mutually_exclusive_group()
    scope.add_argument("--include-data", dest="include_data", action="store_true",
                       help="Include filtered DATA (default; retained for compatibility)")
    scope.add_argument("--mc-only", dest="include_data", action="store_false",
                       help="Explicitly restrict full APPLY to four MC samples")
    scope.add_argument("--only-data", action="store_true",
                       help="Write only filtered DATA; preserve existing MC and full summary")
    parser.set_defaults(include_data=True)
    args = parser.parse_args()
    tag = args.train_tag
    sample = infer_sample_from_tag(tag)
    channel = infer_channel_from_tag(tag)
    year = infer_dataset_year(tag, sample)
    training = resolve_training_config(sample, channel, year, infer_selection_profile(tag, sample))
    selection = resolve_fiducial_config(sample, channel, infer_fid_profile(tag, sample))["expression"]
    if selection != training.get("apply_selection", training["signal_selection"]):
        raise ValueError("Full APPLY fiducial does not match configured apply selection")
    config = json.loads(Path(resolve_model_config_path(tag)).read_text())
    from utils.model_backend import load_model, score_transformed
    from utils.tagging import model_kind
    is_bdtg = model_kind(tag) == "bdtg"
    model = load_model(tag, config)
    scaler = joblib.load(resolve_scaler_path(tag))
    features = config["input_columns"]
    transformed_columns = config["trans_columns"]
    output_dir = Path(root_scored_dir(tag))
    output_dir.mkdir(parents=True, exist_ok=True)
    macro = Path(__file__).resolve().parents[1] / "utils/full_apply_root.C"
    records = []
    for key, spec in resolve_full_apply_config(sample, channel, year).items():
        if (spec["is_data"] and not args.include_data) or (args.only_data and not spec["is_data"]):
            continue
        score_selection = selection
        if not spec["is_data"] and training.get("apply_mc_selection"):
            score_selection = f"({selection}) and ({training['apply_mc_selection']})"
        elif key == "x_prompt" and training.get("apply_signal_selection"):
            score_selection = f"({selection}) and ({training['apply_signal_selection']})"
        columns = list(dict.fromkeys(features + selection_columns(score_selection)))
        destination = output_dir / Path(spec["path"]).name
        if (args.only_data or is_bdtg) and destination.exists():
            raise FileExistsError(f"DATA output already exists: {destination}")
        temporary = destination.with_suffix(".root.tmp")
        with uproot.open(spec["path"]) as source_file:
            source_branches = source_file[spec["tree"]].keys()
            if is_bdtg and "Score_raw" in source_branches:
                raise ValueError(f"Score_raw already exists in {spec['path']}")
            branch_renames = {"Score": "Score_source"} if "Score" in source_branches else {}
            if branch_renames and "Score_source" in source_branches:
                raise ValueError(f"Both Score and Score_source exist in {spec['path']}")
        print(f"Full APPLY {key}: {spec['path']} -> {destination}", flush=True)
        try:
            with tempfile.TemporaryDirectory(prefix="full_apply_") as scratch:
                scores_path = Path(scratch) / "scores.bin"
                raw_path = Path(scratch) / "raw.bin"
                entries = selected_entries = 0
                with scores_path.open("wb") as score_file, raw_path.open("wb") as raw_file:
                    for arrays in uproot.iterate(
                        f"{spec['path']}:{spec['tree']}", expressions=columns,
                        step_size="100 MB", library="np",
                    ):
                        frame = pd.DataFrame(arrays)
                        selected = apply_selection(frame, score_selection, "full APPLY scoring selection")
                        scores = np.full(len(frame), -1, dtype=np.float32)
                        raw_scores = np.full(len(frame), np.nan, dtype=np.float32)
                        if len(selected):
                            transformed = pd.DataFrame(
                                scaler.transform(selected[features]),
                                columns=transformed_columns, index=selected.index,
                            )
                            mapped, raw = score_transformed(model, transformed)
                            scores[selected.index.to_numpy()] = mapped
                            if raw is not None:
                                raw_scores[selected.index.to_numpy()] = raw
                        scores.tofile(score_file)
                        if is_bdtg:
                            raw_scores.tofile(raw_file)
                        entries += len(frame)
                        selected_entries += len(selected)
                if not spec["is_data"]:
                    shutil.copy2(spec["path"], temporary)
                call = str(macro) + "(" + ",".join([
                    json.dumps(str(temporary.resolve())), json.dumps(spec["tree"]),
                    json.dumps(str(scores_path)), "true" if spec["is_data"] else "false",
                    json.dumps(spec["path"]),
                    json.dumps(str(raw_path) if is_bdtg else ""),
                ]) + ")"
                subprocess.run(["root", "-l", "-b", "-q", call], check=True)
                temporary.replace(destination)
            records.append(dict(sample=key, input=spec, output=str(destination),
                                input_entries=entries, scored_entries=selected_entries,
                                score_selection=score_selection, unscored_value=-1,
                                source_branch_renames=branch_renames,
                                selection_applied_to_output=spec["is_data"],
                                output_entries=selected_entries if spec["is_data"] else entries))
            (output_dir / ("data_apply_summary.json" if args.only_data else "full_apply_summary.json")).write_text(json.dumps(
                dict(train_tag=tag, fiducial_selection=selection, score_branch="Score",
                     include_data=args.include_data, samples=records,
                     model_type=model_kind(tag),
                     raw_score_branch="Score_raw" if is_bdtg else None,
                     raw_unscored_value="NaN" if is_bdtg else None,
                     score_mapping="(raw+1)/2" if is_bdtg else "predict_proba",
                     valid_score="Score >= 0"),
                indent=2,
            ))
            print(f"Finished {key}: {records[-1]['output_entries']} entries", flush=True)
        finally:
            temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
