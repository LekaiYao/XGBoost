"""Fixed-parameter TMVA BDTG with an independent native seed-42 split."""
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
from sklearn.metrics import roc_curve, roc_auc_score
from configs import samples
from configs.direct_bdtg_settings import resolve_bdtg_params, tmva_options, SPLIT_SEED, TRAIN_FRACTION
from utils import paths
from utils.tagging import bdtg_version
from utils.varsets import get_varset_columns, infer_varset_from_tag
from utils.selection import apply_selection, selection_columns
from utils.training_weights import resolve_training_weights
from utils.training_identity import sha256, source_revision, check_revision, save_identity
from utils.model_backend import BDTGModel, run_tmva, load_model, map_bdtg_score


def roc_payload(labels, scores, weights):
    fpr, tpr, thresholds = roc_curve(labels, scores, sample_weight=weights)
    return dict(auc=float(roc_auc_score(labels, scores, sample_weight=weights)),
                fpr=fpr.tolist(), tpr=tpr.tolist(),
                threshold=[float(t) if np.isfinite(t) else None for t in thresholds])


def native_train(frame, labels, weights, model_dir, training_dir, params,
                 seed=SPLIT_SEED, fraction=TRAIN_FRACTION):
    """Train using TMVA's own random splitter; return its actual row membership."""
    model_dir=Path(model_dir); training_dir=Path(training_dir)
    model_dir.mkdir(parents=True,exist_ok=True);training_dir.mkdir(parents=True,exist_ok=True)
    if len(frame) >= 2**24:
        raise ValueError("TMVA row-index spectator exceeds exact float range")
    if not np.isfinite(frame.to_numpy()).all() or not np.isfinite(weights).all() or (weights<=0).any():
        raise ValueError("BDTG requires finite inputs and positive weights")
    with tempfile.TemporaryDirectory(prefix="bdtg_train_") as scratch:
        scratch=Path(scratch); prepared=scratch/"input.root"; result=scratch/"training.root"
        with uproot.recreate(prepared) as f:
            for name,mask in [("signal",labels),("background",~labels)]:
                data={c:frame[c].to_numpy(dtype=np.float32)[mask] for c in frame.columns}
                data["row_index"]=np.arange(len(frame),dtype=np.float32)[mask]
                data["training_weight"]=np.asarray(weights,dtype=np.float64)[mask]
                f.mktree(name,data)
        log=run_tmva("train",prepared,result,list(frame.columns),tmva_options(params),seed=seed,fraction=fraction,cwd=scratch)
        (training_dir/"tmva.log").write_text(log)
        with uproot.open(result) as f:
            split=f["split"].arrays(library="np")
        ids=split["row_index"].astype(np.int64)
        if not np.array_equal(np.sort(ids),np.arange(len(frame))):
            raise ValueError("TMVA dropped or duplicated selected candidates")
        np.testing.assert_allclose(split["training_weight"],weights[ids],rtol=2e-6,atol=1e-10)
        train=ids[split["is_train"]];test=ids[~split["is_train"]]
        xml=scratch/"dataset/weights/ML_BDTG.weights.xml"
        if not xml.is_file():raise FileNotFoundError(xml)
        model=BDTGModel(xml,list(frame.columns));before=model.predict_raw(frame.iloc[:min(256,len(frame))])
        target=model_dir/"bdtg.weights.xml"
        shutil.copy2(xml,target);shutil.copy2(result,training_dir/"tmva_training.root")
        after=BDTGModel(target,list(frame.columns)).predict_raw(frame.iloc[:min(256,len(frame))])
        np.testing.assert_array_equal(before,after)
    return train,test


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("train_tag")
    parser.add_argument("--reference-tag")
    args=parser.parse_args();tag=args.train_tag;version=bdtg_version(tag)
    reference=args.reference_tag or tag.rsplit("_bdtg_v",1)[0]+"_xgb_v1"
    sample=samples.infer_sample_from_tag(tag);channel=samples.infer_channel_from_tag(tag)
    year=samples.infer_dataset_year(tag,sample)
    cfg=samples.resolve_training_config(sample,channel,year,samples.infer_selection_profile(tag,sample))
    fid=samples.resolve_fiducial_config(sample,channel,samples.infer_fid_profile(tag,sample))
    if samples.infer_reweight_profile(tag)!="rw0" or cfg.get("training_signal_exclusion"):
        raise ValueError("Initial BDTG workflow requires rw0 and no training-only exclusion")
    features=get_varset_columns(sample,infer_varset_from_tag(tag,sample),channel)
    base_model=Path(paths.model_dir(reference));base_training=Path(paths.training_dir(reference))
    base_cfg=json.loads((base_model/"model_config.json").read_text())
    base_meta=json.loads((base_model/"run_metadata.json").read_text())
    identity=json.loads((base_training/"split_identity.json").read_text())
    if features!=base_cfg["input_columns"]:raise ValueError("Reference feature mismatch")
    for key in ["signal_selection","background_selection"]:
        if cfg[key]!=base_meta[key]:raise ValueError("Reference selection mismatch: "+key)
    if fid["expression"]!=cfg["apply_selection"]:raise ValueError("Apply fiducial mismatch")
    for key in ["signal","background"]:
        if samples.to_root_spec(cfg[key])!=base_meta[key+"_path"]:raise ValueError("Reference source mismatch")
    if sha256(base_training/"split_identity.npz")!=identity["npz_sha256"]:raise ValueError("Reference split changed")
    for rev in identity["sources"]:
        check_revision(rev)
        if sha256(rev["path"])!=rev["sha256"]:raise ValueError("Reference input checksum changed")
    weight_branch=cfg["signal_base_weight_branch"]
    if base_cfg["signal_weight_branches"]!=[weight_branch]:raise ValueError("Reference weight mismatch")
    frames=[];entry_ids=[]
    for key in ["signal","background"]:
        cut=cfg[key+"_selection"]
        columns=list(dict.fromkeys(features+selection_columns(cut)+([weight_branch] if key=="signal" else [])))
        frame=uproot.concatenate(samples.to_root_spec(cfg[key]),expressions=columns,library="pd")
        frame=apply_selection(frame,cut,key)
        entry_ids.append(frame.index.to_numpy(dtype=np.int64));frames.append(frame)
    sig,bkg=frames;ns=len(sig)
    weights=np.r_[resolve_training_weights(sig,weight_branch,"signal"),np.ones(len(bkg))]
    raw=pd.concat([sig[features],bkg[features]],ignore_index=True)
    labels=np.arange(len(raw))<ns
    saved=np.load(base_training/"split_identity.npz")
    np.testing.assert_array_equal(entry_ids[0],saved["signal_entry"])
    np.testing.assert_array_equal(entry_ids[1],saved["background_entry"])
    scaler=joblib.load(base_model/"scaler.pkl")
    transformed=pd.DataFrame(scaler.transform(raw),columns=base_cfg["trans_columns"])
    # Keep the approved reference multiplier fixed despite independent TMVA members.
    factor=float(base_cfg["model_params"]["scale_pos_weight"])
    fit_weights=weights*np.where(labels,factor,1.)
    model_dir=Path(paths.model_dir(tag));training_dir=Path(paths.training_dir(tag))
    if (model_dir/"model_config.json").exists() or (model_dir/"bdtg.weights.xml").exists():
        raise FileExistsError("Refusing to overwrite an existing BDTG model")
    params=resolve_bdtg_params(version)
    train,test=native_train(transformed,labels,fit_weights,model_dir,training_dir,params)
    shutil.copy2(base_model/"scaler.pkl",model_dir/"scaler.pkl")
    save_identity(tag,entry_ids[0],entry_ids[1],train,test,identity["sources"])
    score=map_bdtg_score(BDTGModel(model_dir/"bdtg.weights.xml",list(transformed.columns)).predict_raw(transformed))
    baseline_scores=load_model(reference,base_cfg).predict_proba(transformed)[:,1]
    rocs={};base_rocs={}
    for name,indices in [("train",train),("test",test),("combined",np.arange(len(raw)))]:
        rocs[name]=roc_payload(labels[indices],score[indices],weights[indices])
    for name,indices in [("train",saved["train_index"]),("test",saved["test_index"])]:
        base_rocs[name]=roc_payload(labels[indices],baseline_scores[indices],weights[indices])
        np.testing.assert_allclose(base_rocs[name]["auc"],base_meta["notes"][name+"_auc"],atol=1e-7,rtol=0)
    common=np.intersect1d(test,saved["test_index"])
    common_rocs={name:roc_payload(labels[common],values[common],weights[common]) for name,values in [("bdtg",score),("xgb",baseline_scores)]}
    comparison=dict(reference_tag=reference,split_policy="Independent TMVA seed-42 split; primary train/test metrics use each model's own members",reference=base_rocs,bdtg=rocs,common_test=dict(signal=int(labels[common].sum()),background=int((~labels[common]).sum()),roc=common_rocs),common_test_background_at_signal_efficiency={name:{str(eff):float(np.interp(eff,r["tpr"],r["fpr"])) for eff in [.5,.7,.9]} for name,r in common_rocs.items()})
    for filename,data in [("roc.json",dict(mode="bdtg",**rocs)),("comparison.json",comparison)]:
        (training_dir/filename).write_text(json.dumps(data,indent=2)+"\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots()
    for model,values in [("BDTG",rocs),("XGBoost",base_rocs)]:
        for split in ["train","test"]:
            r=values[split];ax.plot(r["tpr"],r["fpr"],label=f"{model} {split}: {r['auc']:.6f}",linestyle="--" if split=="train" else "-")
    ax.set(xlabel="Signal efficiency",ylabel="Background passing fraction",title="Independent splits, physical event weights",yscale="log",ylim=(1e-4,1));ax.legend();fig.tight_layout();fig.savefig(training_dir/"roc.pdf");plt.close(fig)
    from utils.bdtg_diagnostics import export_training_diagnostics
    diagnostics=export_training_diagnostics(tag,transformed,labels,weights,train,test,score,features)
    config=dict(model_type="bdtg",model_version=version,input_columns=features,trans_columns=base_cfg["trans_columns"],model_params=params,
        score_mapping="(raw+1)/2",score_range=[0,1],raw_score_range=[-1,1],unscored_value=-1,raw_unscored_value="NaN",validity="mapped score >= 0",
        reference_tag=reference,preprocessing="copied reference XGBoost scaler; no refit or TMVA transform",scaler_source_sha256=sha256(base_model/"scaler.pkl"),
        split_policy="TMVA native class-wise random 75/25; not shared with XGBoost",split_random_state=SPLIT_SEED,train_fraction=TRAIN_FRACTION,
        class_weighting=dict(method="explicit signal multiplier; NormMode=None; SkipNormalization=True",value=factor),
        signal_path=base_meta["signal_path"],signal_weight_branch=weight_branch,signal_weight_branches=[weight_branch],
        efficiency_reference_signal=base_meta["signal_path"],efficiency_reference_weight_branch=weight_branch,efficiency_reference_weight_branches=[weight_branch],
        reweight_profile="rw0",signal_input_override=False,training_signal_exclusion=None,
        model_sha256=sha256(model_dir/"bdtg.weights.xml"),scaler_sha256=sha256(model_dir/"scaler.pkl"))
    metadata=dict(train_tag=tag,model_type="bdtg",training_script=__file__,reference_tag=reference,
        diagnostics=diagnostics,
        root_version=subprocess.check_output(["root-config","--version"],text=True).strip(),
        fixed_model_params=params,input_columns=features,signal_path=base_meta["signal_path"],background_path=base_meta["background_path"],
        signal_selection=cfg["signal_selection"],background_selection=cfg["background_selection"],fiducial_selection=fid["expression"],
        train_fraction=TRAIN_FRACTION,test_fraction=1-TRAIN_FRACTION,split_random_state=SPLIT_SEED,
        split_policy=config["split_policy"],class_weighting=config["class_weighting"],
        notes=dict(train_auc=rocs["train"]["auc"],test_auc=rocs["test"]["auc"],combined_auc=rocs["combined"]["auc"],
            n_sig_train=int(labels[train].sum()),n_bkg_train=int((~labels[train]).sum()),n_sig_test=int(labels[test].sum()),n_bkg_test=int((~labels[test]).sum()),
            signal_train_weight_sum=float(weights[train][labels[train]].sum()),background_train_weight_sum=float(weights[train][~labels[train]].sum()),
            reference_model_sha256=sha256(base_model/"xgb_model.pkl"),actual_split_equals_reference=bool(np.array_equal(np.sort(train),np.sort(saved["train_index"])))) )
    (model_dir/"run_metadata.json").write_text(json.dumps(metadata,indent=2)+"\n")
    for rev in identity["sources"]:check_revision(rev)
    (model_dir/"model_config.json").write_text(json.dumps(config,indent=2)+"\n")
    print(json.dumps(metadata["notes"],indent=2),flush=True)

if __name__=="__main__":main()
