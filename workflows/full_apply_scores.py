"""Opt-in complete ROOT copies with Score; compact APPLY remains the default."""
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
    model = joblib.load(resolve_model_path(tag))
    scaler = joblib.load(resolve_scaler_path(tag))
    features = config["input_columns"]
    transformed_columns = config["trans_columns"]
    columns = list(dict.fromkeys(features + selection_columns(selection)))
    output_dir = Path(root_scored_dir(tag))
    output_dir.mkdir(parents=True, exist_ok=True)
    macro = Path(__file__).resolve().parents[1] / "utils/full_apply_root.C"
    records = []
    for key, spec in resolve_full_apply_config(sample, channel, year).items():
        destination = output_dir / Path(spec["path"]).name
        temporary = destination.with_suffix(".root.tmp")
        print(f"Full APPLY {key}: {spec['path']} -> {destination}", flush=True)
        try:
            with tempfile.TemporaryDirectory(prefix="full_apply_") as scratch:
                scores_path = Path(scratch) / "scores.bin"
                entries = selected_entries = 0
                with scores_path.open("wb") as score_file:
                    for arrays in uproot.iterate(
                        f"{spec['path']}:{spec['tree']}", expressions=columns,
                        step_size="100 MB", library="np",
                    ):
                        frame = pd.DataFrame(arrays)
                        selected = apply_selection(frame, selection, "full APPLY fiducial")
                        scores = np.full(len(frame), -1, dtype=np.float32)
                        if len(selected):
                            transformed = pd.DataFrame(
                                scaler.transform(selected[features]),
                                columns=transformed_columns, index=selected.index,
                            )
                            scores[selected.index.to_numpy()] = model.predict_proba(transformed)[:, 1]
                        scores.tofile(score_file)
                        entries += len(frame)
                        selected_entries += len(selected)
                if not spec["is_data"]:
                    shutil.copy2(spec["path"], temporary)
                call = str(macro) + "(" + ",".join([
                    json.dumps(str(temporary.resolve())), json.dumps(spec["tree"]),
                    json.dumps(str(scores_path)), "true" if spec["is_data"] else "false",
                    json.dumps(spec["path"]),
                ]) + ")"
                subprocess.run(["root", "-l", "-b", "-q", call], check=True)
                temporary.replace(destination)
            records.append(dict(sample=key, input=spec, output=str(destination),
                                input_entries=entries, fiducial_entries=selected_entries,
                                output_entries=selected_entries if spec["is_data"] else entries))
            (output_dir / "full_apply_summary.json").write_text(json.dumps(
                dict(train_tag=tag, fiducial_selection=selection, score_branch="Score", samples=records),
                indent=2,
            ))
            print(f"Finished {key}: {records[-1]['output_entries']} entries", flush=True)
        finally:
            temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
