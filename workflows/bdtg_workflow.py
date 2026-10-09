"""Execute a dependency-controlled BDTG workflow step with frozen-code checks."""
import argparse
import json
import subprocess
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import uproot
import joblib
from utils import paths
from utils.training_identity import sha256
from utils.model_backend import load_model, score_transformed
from utils.selection import apply_selection, selection_columns


def validate(tag):
    selected=Path(paths.selected_dir(tag));config=json.loads(Path(paths.model_config_path(tag)).read_text())
    model=load_model(tag,config);scaler=joblib.load(paths.scaler_path(tag))
    if sha256(paths.model_path(tag))!=config["model_sha256"] or sha256(paths.scaler_path(tag))!=config["scaler_sha256"]:
        raise ValueError("Model or scaler checksum changed")
    metadata=json.loads((Path(paths.model_dir(tag))/"run_metadata.json").read_text())
    split=json.loads((Path(paths.training_dir(tag))/"split_identity.json").read_text())
    if sha256(Path(paths.training_dir(tag))/"split_identity.npz")!=split["npz_sha256"]:raise ValueError("Split checksum mismatch")
    def compare_scores(frame,score_name,raw_name):
        transformed=pd.DataFrame(scaler.transform(frame[config["input_columns"]]),columns=config["trans_columns"])
        mapped,raw=score_transformed(model,transformed)
        np.testing.assert_allclose(frame[score_name],mapped,rtol=0,atol=1e-7)
        np.testing.assert_allclose(frame[raw_name],raw,rtol=0,atol=1e-7)
    compact=[]
    for name in ["MC_with_score.root","DATA_with_score.root","MC_psi2s_with_score.root","REFERENCE_MC_with_score.root"]:
        path=selected/name
        with uproot.open(path) as f:
            trees=[f[k] for k in f.keys() if getattr(f[k],"classname","")=="TTree"]
            if len(trees)!=1:raise ValueError("Unexpected compact ROOT tree layout")
            tree=trees[0]
            for chunk in tree.iterate(expressions=["Prediction","Prediction_raw"],library="np",step_size="50 MB"):
                if not np.isfinite(chunk["Prediction_raw"]).all():raise ValueError("Invalid compact raw scores")
                np.testing.assert_allclose(chunk["Prediction"],(chunk["Prediction_raw"]+1)/2,atol=1e-7,rtol=0)
            frame=tree.arrays(config["input_columns"]+["Prediction","Prediction_raw"],entry_stop=256,library="pd")
            if frame.empty:raise ValueError("Empty compact APPLY")
            compare_scores(frame,"Prediction","Prediction_raw")
            compact.append(dict(path=str(path),entries=tree.num_entries))
    summary=json.loads((Path(paths.root_scored_dir(tag))/"full_apply_summary.json").read_text())
    expected={"x_prompt","x_nonprompt","psi2s_prompt","psi2s_nonprompt","data"}
    if {r["sample"] for r in summary["samples"]}!=expected:raise ValueError("Incomplete full APPLY sample set")
    full=[]
    for r in summary["samples"]:
        spec=r["input"];dest=Path(r["output"])
        with uproot.open(spec["path"]) as src,uproot.open(dest) as out:
            a=src[spec["tree"]];b=out[spec["tree"]]
            renamed=lambda name:"Score_source" if name=="Score" else name
            for name,typ in a.typenames().items():
                if b.typenames().get(renamed(name))!=typ:raise ValueError("Original branch type changed: "+name)
            if b.num_entries!=r["output_entries"]:raise ValueError("Output entry count mismatch")
            if not spec["is_data"]:
                if b.num_entries!=a.num_entries:raise ValueError("MC RECO count changed")
                for key,kind in src.classnames().items():
                    if kind=="TTree" and key.split(';')[0]!=spec["tree"]:
                        if out[key].num_entries!=src[key].num_entries or out[key].typenames()!=src[key].typenames():
                            raise ValueError("Non-RECO tree changed")
            scored=0
            for chunk in b.iterate(expressions=["Score","Score_raw"],library="np",step_size="50 MB"):
                good=chunk["Score"]>=0;scored+=int(good.sum())
                if not np.isfinite(chunk["Score"]).all():raise ValueError("Invalid mapped scores")
                if not np.isnan(chunk["Score_raw"][~good]).all():raise ValueError("Invalid unscored raw sentinel")
                if not (chunk["Score"][~good]==-1).all():raise ValueError("Invalid unscored sentinel")
                np.testing.assert_allclose(chunk["Score"][good],(chunk["Score_raw"][good]+1)/2,atol=1e-7,rtol=0)
                if spec["is_data"] and not good.all():raise ValueError("Unselected DATA was retained")
            if scored!=r["scored_entries"]:raise ValueError("Scored count mismatch")
            frame=b.arrays(config["input_columns"]+["Score","Score_raw"],entry_stop=min(20000,b.num_entries),library="pd")
            frame=frame[frame.Score>=0].head(256)
            if frame.empty:raise ValueError("No scored entries for spot check")
            compare_scores(frame,"Score","Score_raw")
            # Check source selection counts and original branch values on early selected entries.
            count=0;first=None
            for source_frame in a.iterate(step_size="50 MB",library="pd"):
                chosen=apply_selection(source_frame,r["score_selection"],"validation selection")
                count+=len(chosen)
                if first is None and len(chosen):first=chosen.head(32)
            if count!=scored:raise ValueError("Selection/count disagreement with source")
            for j,source_index in enumerate(first.index):
                index=j if spec["is_data"] else int(source_index)
                check=b.arrays(entry_start=index,entry_stop=index+1,library="np")
                for name in a.keys():
                    np.testing.assert_equal(check[renamed(name)][0],first.loc[source_index,name])
            full.append(dict(sample=r["sample"],entries=b.num_entries,scored=scored))
    result=dict(status="PASS",train_tag=tag,compact=compact,full=full,model_type="bdtg",
                actual_split_equals_xgb=metadata["notes"]["actual_split_equals_reference"],
                validation="all score mappings/counts; original branch types and 32 selected rows; GEN schema/counts; 256 rescored candidates per output")
    (selected/"bdtg_validation.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))



def validate_postprocessing(tag):
    from utils.bdtg_diagnostics import validate_training_diagnostics
    diagnostics=validate_training_diagnostics(tag)
    required={
        "thresholds": Path(paths.cut_scan_dir(tag))/"thresholds.json",
        "punzi": Path(paths.punzi_dir(tag))/"scan.json",
        "shap": Path(paths.shap_dir(tag))/"shap_metadata.json",
        "shap_importance": Path(paths.shap_importance_path(tag)),
    }
    for name,path in required.items():
        if not path.is_file(): raise FileNotFoundError(f"Missing {name}: {path}")
    punzi=json.loads(required["punzi"].read_text())
    shap=json.loads(required["shap"].read_text())
    if punzi["status"]!="PASS" or punzi["train_tag"]!=tag:raise ValueError("Punzi validation failed")
    if punzi["model_sha256"]!=sha256(paths.model_path(tag)):raise ValueError("Punzi model mismatch")
    if shap["train_tag"]!=tag:raise ValueError("SHAP tag mismatch")
    if shap["model_sha256"]!=sha256(paths.model_path(tag)):raise ValueError("SHAP model mismatch")
    from configs.year_pairings import resolve_year_pairing_for_tag
    pairing=resolve_year_pairing_for_tag(tag)
    result=dict(status="PASS",train_tag=tag,artifacts={k:str(v) for k,v in required.items()},
        fit_interface="skipped_no_registered_year_pairing" if pairing is None else "registered_pairing",
        model_sha256=sha256(paths.model_path(tag)),training_diagnostics="PASS",boosting_rounds=diagnostics["n_estimators"])
    (Path(paths.selected_dir(tag))/"standard_workflow_validation.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=["train","apply","extra","full","draw","shap","fit_interface","verify"])
    parser.add_argument("train_tag");parser.add_argument("code_manifest")
    args=parser.parse_args();manifest=json.loads(Path(args.code_manifest).read_text())
    if manifest["train_tag"]!=args.train_tag:raise ValueError("Manifest tag mismatch")
    for name,digest in manifest["code_sha256"].items():
        if sha256(name)!=digest:raise ValueError("Code changed since submission: "+name)
    for name,digest in manifest.get("protected_sha256", {}).items():
        if sha256(name)!=digest:raise ValueError("Protected XGBoost artifact changed: "+name)
    from utils.training_identity import check_revision
    for revision in manifest.get("input_revisions", []):check_revision(revision)
    if subprocess.check_output(["root-config","--version"],text=True).strip()!=manifest["root_version"]:
        raise ValueError("ROOT environment changed since validation")
    tag=args.train_tag
    commands={"train":["workflows.bdtg_train",tag],"apply":["workflows.batch_apply_scores",tag],
        "extra":["workflows.batch_apply_scores","--apply-extra-mc","psi2s",tag],
        "full":["workflows.full_apply_scores",tag],
        "draw":["workflows.batch_draw_scores",tag],"shap":["workflows.shap_importance",tag],
        "fit_interface":["workflows.integration.export_default_fit_interface",tag]}
    if args.stage=="verify":
        validate(tag)
        validate_postprocessing(tag)
    else:subprocess.run([sys.executable,"-m",*commands[args.stage]],check=True)

if __name__=="__main__":main()
