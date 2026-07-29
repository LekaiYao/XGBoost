from pathlib import Path

import pandas as pd
import uproot


def score_frame(frame, model_bundles):
    output = frame.copy()
    for bundle in model_bundles:
        transformed = pd.DataFrame(
            bundle["scaler"].transform(frame[bundle["input_columns"]]),
            columns=bundle["trans_columns"],
            index=frame.index,
        )
        output[bundle["score_column"]] = bundle["model"].predict_proba(
            transformed[bundle["trans_columns"]]
        )[:, 1]
    return output


def write_scored_root(
    input_spec,
    output_path,
    output_tree,
    model_bundles,
    step_size="100 MB",
):
    output_path = Path(output_path)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary_path.unlink(missing_ok=True)
    entries = 0
    chunks = 0
    try:
        with uproot.recreate(temporary_path) as output_file:
            for frame in uproot.iterate(
                input_spec,
                step_size=step_size,
                library="pd",
            ):
                scored = score_frame(frame, model_bundles)
                arrays = {
                    column: scored[column].to_numpy()
                    for column in scored.columns
                }
                if chunks == 0:
                    output_file.mktree(output_tree, arrays)
                else:
                    output_file[output_tree].extend(arrays)
                entries += len(scored)
                chunks += 1
        if chunks == 0:
            raise ValueError(f"Input TTree contains no entries: {input_spec}")
        temporary_path.replace(output_path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
    return {"entries": int(entries), "chunks": int(chunks)}
