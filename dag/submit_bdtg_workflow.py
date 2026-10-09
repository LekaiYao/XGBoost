"""Prepare a reproducible BDTG submission; submit only with explicit --submit.

Run from the checkout with .venv/bin/python -m dag.submit_bdtg_workflow TAG.
Requires the matching trained XGB reference, its split identities, and source
ROOT files. No historical submission manifest or automatic cleanup is used.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import uuid

from configs import samples
from configs.direct_bdtg_settings import resolve_bdtg_params
from dag.make_bdtg_workflow import make_dag
from utils import paths
from utils.tagging import bdtg_version
from utils.training_identity import check_revision, sha256, source_revision
from utils.varsets import get_varset_columns, infer_varset_from_tag

REPO = Path(__file__).resolve().parents[1]
DEFAULT_AFS = '/afs/cern.ch/user/l/leyao/private/pbpb_work/X_analysis/XGBoost'
STAGES = ['train', 'apply', 'extra', 'full', 'draw', 'shap', 'fit_interface', 'verify']


def capture(args, cwd=None):
    return subprocess.check_output(args, cwd=cwd, text=True).strip()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def resolve_inputs(tag, postprocess_only=False):
    if not re.fullmatch(r'X_(pb18|pb23)_v\d+_fid\d+_\d+v\d*(?:_rw0)?_bdtg_v\d+', tag):
        raise ValueError('Full BDTG workflow supports PbPb18/PbPb23 X, rw0 tags')
    params = resolve_bdtg_params(bdtg_version(tag))
    sample = samples.infer_sample_from_tag(tag)
    channel = samples.infer_channel_from_tag(tag)
    year = samples.infer_dataset_year(tag, sample)
    cfg = samples.resolve_training_config(sample, channel, year, samples.infer_selection_profile(tag, sample))
    fid = samples.resolve_fiducial_config(sample, channel, samples.infer_fid_profile(tag, sample))
    features = get_varset_columns(sample, infer_varset_from_tag(tag, sample), channel)
    if cfg.get('training_signal_exclusion'):
        raise ValueError('Training-only exclusion is unsupported by BDTG')
    if fid['expression'] != cfg['apply_selection']:
        raise ValueError('Fiducial/apply selection mismatch')
    reference = tag.rsplit('_bdtg_v', 1)[0] + '_xgb_v1'
    md = Path(paths.model_dir(reference))
    td = Path(paths.training_dir(reference))
    ref_cfg = json.loads((md / 'model_config.json').read_text())
    ref_meta = json.loads((md / 'run_metadata.json').read_text())
    identity = json.loads((td / 'split_identity.json').read_text())
    if features != ref_cfg['input_columns']:
        raise ValueError('Reference feature mismatch')
    if ref_cfg['signal_weight_branches'] != [cfg['signal_base_weight_branch']]:
        raise ValueError('Reference weight mismatch')
    for key in ['signal', 'background']:
        if samples.to_root_spec(cfg[key]) != ref_meta[key + '_path']:
            raise ValueError('Reference source mismatch: ' + key)
        if cfg[key + '_selection'] != ref_meta[key + '_selection']:
            raise ValueError('Reference selection mismatch: ' + key)
    if sha256(td / 'split_identity.npz') != identity['npz_sha256']:
        raise ValueError('Reference split checksum mismatch')
    revisions = {}
    for rev in identity['sources']:
        check_revision(rev)
        if sha256(rev['path']) != rev['sha256']:
            raise ValueError('Reference source checksum mismatch')
        revisions[(rev['path'], rev['tree'])] = rev
    full = samples.resolve_full_apply_config(sample, channel, year)
    for spec in full.values():
        key = (spec['path'], spec['tree'])
        if key not in revisions:
            revisions[key] = source_revision(samples.to_root_spec(spec))
    protected = [md / name for name in ['model_config.json', 'run_metadata.json', 'xgb_model.pkl', 'scaler.pkl']]
    protected += [td / 'split_identity.json', td / 'split_identity.npz']
    if postprocess_only:
        protected += [Path(paths.model_dir(tag)) / name for name in
                      ['model_config.json', 'run_metadata.json', 'bdtg.weights.xml', 'scaler.pkl']]
        protected += [Path(paths.training_dir(tag)) / name for name in ['split_identity.json', 'split_identity.npz']]
        for name in ['batch_apply_summary.json', 'full_apply_summary.json']:
            directory = Path(paths.selected_dir(tag)) if name == 'batch_apply_summary.json' else Path(paths.root_scored_dir(tag))
            if not (directory / name).is_file():
                raise FileNotFoundError(directory / name)
    else:
        for directory in [paths.model_dir(tag), paths.training_dir(tag), paths.selected_dir(tag), paths.shap_dir(tag)]:
            if Path(directory).exists() and any(Path(directory).iterdir()):
                raise FileExistsError('Existing output must be archived explicitly: ' + directory)
    return dict(train_tag=tag, reference_tag=reference, model_params=params,
                training=cfg, fiducial=fid, features=features, full_apply=full,
                input_revisions=list(revisions.values()),
                protected_sha256={str(p.resolve()): sha256(p) for p in protected})


def source_files(repo):
    names = capture(['git', 'ls-files', '--cached', '--others', '--exclude-standard'], cwd=repo).splitlines()
    roots = {'configs', 'utils', 'workflows', 'dag', 'wrappers', 'tests', 'scripts'}
    return sorted({name for name in names if Path(name).parts[0] in roots
                   and Path(name).suffix in {'.py', '.C', '.sh', '.sub', '.json'}
                   and (repo / name).is_file()})


def prepare(repo, resolved, postprocess_only=False):
    tag = resolved['train_tag']
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + uuid.uuid4().hex[:8]
    bundle = repo / 'output/submissions' / (run_id + '_' + tag)
    bundle.mkdir(parents=True, exist_ok=False)
    write_json(bundle / 'resolved_config.json', resolved)
    hashes = {}
    for name in source_files(repo):
        original = repo / name
        digest = sha256(original)
        target = bundle / 'source_snapshot' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(original, target)
        if sha256(target) != digest or sha256(original) != digest:
            raise ValueError('Source changed during snapshot: ' + name)
        hashes[str(original)] = digest
    environment = dict(python=sys.version, executable=sys.executable,
                       packages=capture([sys.executable, '-m', 'pip', 'freeze']),
                       root_version=capture(['root-config', '--version']),
                       git_head=capture(['git', 'rev-parse', 'HEAD'], cwd=repo),
                       git_status=capture(['git', 'status', '--porcelain'], cwd=repo))
    write_json(bundle / 'environment.json', environment)
    manifest = dict(train_tag=tag, reference_tag=resolved['reference_tag'],
                    root_version=environment['root_version'], code_sha256=hashes,
                    input_revisions=resolved['input_revisions'],
                    protected_sha256=resolved['protected_sha256'])
    manifest_path = bundle / 'submission_manifest.json'
    write_json(manifest_path, manifest)
    dag = make_dag(bundle, tag, manifest_path, postprocess_only)
    shutil.copy2(repo / 'dag/submit_templates/submit_bdtg_job.sub', bundle / 'submit_bdtg_job.sub')
    wrapper = bundle / 'run_bdtg_job.sh'
    wrapper.write_text('#!/bin/bash\nset -euo pipefail\nexport BDTG_REPO_DIR=' + shlex.quote(str(repo)) +
                       '\nexec ' + shlex.quote(str(repo / 'wrappers/run_bdtg_job.sh')) + ' "$@"\n')
    wrapper.chmod(0o755)
    receipt = dict(state='prepared_not_submitted', train_tag=tag, bundle=str(bundle),
                   dag=str(dag), manifest=str(manifest_path), stages=STAGES[4:] if postprocess_only else STAGES,
                   submit_command=[str(repo / '.venv/bin/python'), '-m', 'dag.submit_bdtg_workflow',
                                   '--prepared', str(bundle), '--submit'])
    receipt['bundle_sha256'] = {name: sha256(bundle / name) for name in
        [dag.name, 'submit_bdtg_job.sub', 'run_bdtg_job.sh', 'submission_manifest.json',
         'environment.json', 'resolved_config.json']}
    write_json(bundle / 'preparation.json', receipt)
    return bundle


def validate_bundle(bundle):
    receipt = json.loads((bundle / 'preparation.json').read_text())
    for name, digest in receipt['bundle_sha256'].items():
        if sha256(bundle / name) != digest:
            raise ValueError('Prepared bundle changed: ' + name)
    manifest = json.loads((bundle / 'submission_manifest.json').read_text())
    if receipt['bundle'] != str(bundle) or manifest['train_tag'] != receipt['train_tag']:
        raise ValueError('Preparation identity mismatch')
    for key in ['code_sha256', 'protected_sha256']:
        for name, digest in manifest[key].items():
            if sha256(name) != digest:
                raise ValueError('Changed since preparation: ' + name)
    for rev in manifest['input_revisions']:
        check_revision(rev)
        if sha256(rev['path']) != rev['sha256']:
            raise ValueError('Input checksum changed: ' + rev['path'])
    if capture(['root-config', '--version']) != manifest['root_version']:
        raise ValueError('ROOT version changed')
    if capture([sys.executable, '-m', 'pip', 'freeze']) != json.loads((bundle / 'environment.json').read_text())['packages']:
        raise ValueError('Python packages changed')
    if 'train' in receipt['stages']:
        tag = receipt['train_tag']
        for directory in [paths.model_dir(tag), paths.training_dir(tag), paths.selected_dir(tag), paths.shap_dir(tag)]:
            if Path(directory).exists() and any(Path(directory).iterdir()):
                raise FileExistsError('Existing output: ' + directory)
    return receipt


def submit(bundle, afs):
    if (bundle / 'submission_attempt.json').exists():
        raise FileExistsError('Submission already attempted; inspect receipt/log before preparing a new run')
    receipt = validate_bundle(bundle)
    tag = receipt['train_tag']
    active = capture(['condor_q', '-constraint', f'regexp("{tag}", Args)', '-af', 'ClusterId'])
    if active:
        raise RuntimeError('Active jobs for tag: ' + active)
    destination = afs / 'bdtg_submissions' / bundle.name
    destination.mkdir(parents=True, exist_ok=False)
    (destination / 'logs').mkdir()
    dag_name = Path(receipt['dag']).name
    for name in [dag_name, 'submit_bdtg_job.sub', 'run_bdtg_job.sh']:
        shutil.copy2(bundle / name, destination / name)
    dry = subprocess.run(['condor_submit', '-dry-run', str(destination / 'dry_run.ad'),
                          'submit_bdtg_job.sub', '-append', 'stage=train',
                          '-append', 'train_tag=' + tag, '-append', 'job_tag=dryrun',
                          '-append', 'code_manifest=' + receipt['manifest']],
                         cwd=destination, text=True, capture_output=True)
    (bundle / 'dry_run.log').write_text(dry.stdout + dry.stderr)
    dry.check_returncode()
    command = ['condor_submit_dag', '-MaxJobs', '1', dag_name]
    result = dict(receipt, state='submission_attempted', host=capture(['hostname', '-f']),
                  schedd=capture(['condor_config_val', 'SCHEDD_HOST']),
                  pool=capture(['condor_config_val', 'COLLECTOR_HOST']),
                  afs_directory=str(destination), command=command)
    write_json(bundle / 'submission_attempt.json', result)
    process = subprocess.run(command, cwd=destination, capture_output=True, text=True)
    (bundle / 'submit.log').write_text(process.stdout + process.stderr)
    match = re.search(r'submitted to cluster (\d+)', process.stdout)
    result.update(returncode=process.returncode, state='submitted' if process.returncode == 0 else 'submit_failed')
    if match:
        result['dag_cluster'] = int(match.group(1))
        result['query_command'] = shlex.join(['condor_q', '-name', result['schedd'], '-pool', result['pool'].split(',')[0].strip(), str(result['dag_cluster'])])
    write_json(bundle / 'submission.json', result)
    print(json.dumps(result, indent=2))
    process.check_returncode()
    if not match:
        raise RuntimeError('Submission returned success but ID was not parsed; inspect submit.log; do not resubmit blindly')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('train_tag', nargs='?')
    parser.add_argument('--postprocess-only', action='store_true')
    parser.add_argument('--prepared', type=Path, help='Submit an existing preparation without regenerating it')
    parser.add_argument('--submit', action='store_true', help='Actually submit; default only prepares EOS files')
    parser.add_argument('--afs-dir', type=Path, default=Path(DEFAULT_AFS))
    args = parser.parse_args()
    os.chdir(REPO)
    if args.prepared:
        if args.train_tag or args.postprocess_only or not args.submit:
            parser.error('--prepared requires --submit and cannot accompany tag or --postprocess-only')
        bundle = args.prepared.resolve()
    else:
        if not args.train_tag:
            parser.error('train_tag required')
        resolved = resolve_inputs(args.train_tag, args.postprocess_only)
        bundle = prepare(REPO, resolved, args.postprocess_only)
        print((bundle / 'preparation.json').read_text())
    if args.submit:
        afs = args.afs_dir.resolve()
        if not str(afs).startswith('/afs/'):
            parser.error('--afs-dir must resolve to AFS; EOS code / AFS submission separation is required')
        submit(bundle, afs)


if __name__ == '__main__':
    main()
