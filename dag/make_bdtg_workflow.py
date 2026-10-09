"""BDTG DAG with standard DRAW/Punzi, SHAP and fit-interface postprocessing."""
from pathlib import Path
from utils.tagging import bdtg_version
from configs.samples import infer_dataset_year,infer_sample_from_tag

def make_dag(out_dir,tag,manifest,postprocess_only=False):
    bdtg_version(tag);infer_dataset_year(tag,infer_sample_from_tag(tag))
    out_dir=Path(out_dir);out_dir.mkdir(parents=True,exist_ok=True)
    dest=out_dir/f"wf_bdtg_{'post_' if postprocess_only else ''}{tag}.dag"
    if dest.exists():raise FileExistsError(dest)
    lines=[];previous=None
    stages=[] if postprocess_only else ["train","apply","extra","full"]
    for stage in stages+["draw","shap","fit_interface","verify"]:
        node=stage.upper()
        lines += [f"JOB {node} submit_bdtg_job.sub",f'VARS {node} stage="{stage}" train_tag="{tag}" code_manifest="{Path(manifest).resolve()}" job_tag="{tag}_{stage}"']
        if previous:lines.append(f"PARENT {previous} CHILD {node}")
        previous=node
    dest.write_text("\n".join(lines)+"\n");return dest

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-tag", required=True)
    parser.add_argument("--code-manifest", required=True)
    parser.add_argument("--out-dir", default="dag/generated")
    parser.add_argument("--postprocess-only", action="store_true")
    args = parser.parse_args()
    print(make_dag(args.out_dir, args.train_tag, args.code_manifest, args.postprocess_only))
