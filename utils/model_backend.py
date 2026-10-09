"""Load and score XGBoost or TMVA without changing the Python environment."""
from pathlib import Path
import json
import subprocess
import tempfile
import joblib
import numpy as np
from utils.tagging import model_kind


def run_tmva(mode, source, target, columns, options="", xml="", seed=42, fraction=.75, cwd=None):
    macro = Path(__file__).with_name("tmva_bdtg.C").resolve()
    args = [mode, str(Path(source).resolve()), str(Path(target).resolve()),
            ",".join(columns), options, str(Path(xml).resolve()) if xml else ""]
    call = str(macro) + "(" + ",".join([*(json.dumps(x) for x in args), str(seed), str(fraction)]) + ")"
    result = subprocess.run(["root", "-l", "-b", "-q", call], cwd=cwd,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if result.returncode:
        raise RuntimeError("TMVA failed:\n" + result.stdout[-12000:])
    return result.stdout


def map_bdtg_score(raw):
    raw = np.asarray(raw)
    if not np.isfinite(raw).all() or ((raw < -1) | (raw > 1)).any():
        raise ValueError("Invalid TMVA BDTG raw response")
    return (raw + 1.) / 2.


class BDTGModel:
    def __init__(self, path, columns):
        self.path = str(Path(path).resolve())
        if not Path(self.path).is_file():
            raise FileNotFoundError(self.path)
        self.columns = list(columns)

    def predict_raw(self, transformed):
        if list(transformed.columns) != self.columns:
            raise ValueError("BDTG feature order mismatch")
        values = np.asarray(transformed, dtype=np.float32)
        if not np.isfinite(values).all():
            raise ValueError("BDTG inputs must be finite")
        if not len(values):
            return np.empty(0, dtype=np.float64)
        with tempfile.TemporaryDirectory(prefix="bdtg_score_") as scratch:
            source = Path(scratch)/"input.bin"; output = Path(scratch)/"raw.bin"
            values.tofile(source)
            run_tmva("predict", source, output, self.columns, xml=self.path, cwd=scratch)
            raw = np.fromfile(output, dtype=np.float64)
        if len(raw) != len(values):
            raise ValueError("BDTG prediction length mismatch")
        map_bdtg_score(raw)
        return raw

    def predict_proba(self, transformed):
        score = map_bdtg_score(self.predict_raw(transformed))
        return np.column_stack([1-score, score])


def load_model(tag, config=None):
    from utils.paths import resolve_model_path, resolve_model_config_path
    config = config or json.loads(Path(resolve_model_config_path(tag)).read_text())
    kind = model_kind(tag)
    if config.get("model_type", kind) != kind:
        raise ValueError("Model tag/config type mismatch")
    if kind == "bdtg":
        return BDTGModel(resolve_model_path(tag), config["trans_columns"])
    return joblib.load(resolve_model_path(tag))


def score_transformed(model, transformed):
    if isinstance(model, BDTGModel):
        raw = model.predict_raw(transformed)
        return map_bdtg_score(raw), raw
    return model.predict_proba(transformed)[:, 1], None
