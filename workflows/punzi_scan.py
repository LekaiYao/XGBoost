"""Two approved Punzi scopes with original train/test membership; no retraining."""
import argparse
import json
from pathlib import Path
import textwrap
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import uproot
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from configs.samples import (split_root_spec, infer_sample_from_tag, infer_channel_from_tag,
    infer_dataset_year, resolve_draw_config)
from utils.paths import (model_dir, training_dir, selected_dir, cut_scan_dir, punzi_dir,
    resolve_model_path, resolve_scaler_path, resolve_model_config_path)
from utils.selection import apply_selection, selection_columns
from utils.training_identity import sha256, source_revision, check_revision
from utils.punzi import scan, optimal, mass_mask, sidebands_from_selection
from utils.model_backend import load_model
from utils.tagging import model_kind

RECIPE_PATH = Path(__file__).resolve().parents[1]/'configs/punzi_pb23_x.json'


def applicable(tag):
    sample = infer_sample_from_tag(tag)
    return (sample == 'pbpb' and infer_channel_from_tag(tag) == 'X'
            and infer_dataset_year(tag, sample) in {'2018', '2023'})


def read_selected(spec, columns, selections):
    """Retain physical source entry IDs while reading bounded chunks."""
    path, tree = split_root_spec(spec)
    frames, offset = [], 0
    with uproot.open(path) as root:
        for df in root[tree].iterate(list(dict.fromkeys(columns)), step_size='80 MB', library='pd'):
            df.index = np.arange(offset, offset+len(df), dtype=np.int64)
            offset += len(df)
            ids = np.unique(np.concatenate([apply_selection(df, s, 'Punzi input').index.to_numpy()
                                            for s in selections]))
            frames.append(df.loc[ids])
    if not frames: raise ValueError('Empty source')
    return pd.concat(frames)


def predict(df, config, scaler, model):
    features = df[config['input_columns']]
    if not np.isfinite(features.to_numpy()).all(): raise ValueError('Nonfinite features')
    transformed = pd.DataFrame(scaler.transform(features), columns=config['trans_columns'], index=df.index)
    return model.predict_proba(transformed)[:, 1]


def validated_split(kind,cfg,meta,signal_entries,background_entries,frozen):
    ns,nb=len(signal_entries),len(background_entries)
    if kind=='bdtg':
        if frozen is None:raise ValueError('BDTG native split is required')
        tr,te=frozen['train_index'],frozen['test_index']
    else:
        tr,te=train_test_split(np.arange(ns+nb),train_size=cfg['train_fraction'],random_state=cfg['split_random_state'])
    for ids in (tr,te):
        if ids.dtype.kind not in 'iu' or ids.ndim!=1:raise ValueError('Invalid split index dtype/shape')
    if not np.array_equal(np.sort(np.r_[tr,te]),np.arange(ns+nb)):
        raise ValueError('Split membership is not a disjoint complete partition')
    arrays=dict(signal_entry=signal_entries,background_entry=background_entries,train_index=tr,test_index=te)
    if frozen is not None:
        for key in arrays:
            if not np.array_equal(arrays[key],frozen[key]):raise ValueError('Original split mismatch: '+key)
    notes=meta['notes']
    for key,count in [('n_sig_train',(tr<ns).sum()),('n_bkg_train',(tr>=ns).sum()),
                      ('n_sig_test',(te<ns).sum()),('n_bkg_test',(te>=ns).sum())]:
        if key in notes and count!=notes[key]:raise ValueError('Training population/count mismatch: '+key)
    return arrays


def run(tag, draw_config=None):
    if not applicable(tag):
        print(f'Punzi not applicable: approved recipes are for PbPb18/PbPb23 X, tag={tag}')
        return None
    out = Path(punzi_dir(tag)); out.mkdir(parents=True, exist_ok=True)
    cfg_path = Path(resolve_model_config_path(tag)); cfg=json.loads(cfg_path.read_text())
    meta_path=Path(model_dir(tag))/'run_metadata.json'; meta=json.loads(meta_path.read_text())
    apply_path=Path(selected_dir(tag))/'batch_apply_summary.json'; app=json.loads(apply_path.read_text())
    year = infer_dataset_year(tag, infer_sample_from_tag(tag))
    recipe_path = RECIPE_PATH if year == '2023' else RECIPE_PATH.with_name('punzi_pb18_x.json')
    recipe=json.loads(recipe_path.read_text())
    kind=model_kind(tag)
    if kind=='bdtg':
        if cfg.get('model_type')!='bdtg' or Path(meta['training_script']).name!='bdtg_train.py':
            raise ValueError('BDTG provenance mismatch')
    elif meta['training_script'] != 'workflows/xgboost_train_direct.py' or meta['is_optuna']:
        raise ValueError('Punzi split support requires direct training with recorded original split')
    if cfg.get('training_signal_exclusion') or cfg.get('signal_weight_branches') != ['pThatreweight']:
        raise ValueError('Unsupported exclusion/weight definition; cannot invent membership')
    if kind=='bdtg':
        if cfg['split_policy']!='TMVA native class-wise random 75/25; not shared with XGBoost':
            raise ValueError('Unsupported BDTG split policy')
    elif cfg['split_policy'] != 'candidate_level_random_split_preliminary' or meta['val_fraction'] != 0:
        raise ValueError('Unsupported split policy')
    sigspec,bkgspec=meta['signal_path'],meta['background_path']
    if cfg['efficiency_reference_signal'] != sigspec or app['input_datasets']['data_input'] != bkgspec or app['input_datasets']['mc_input'] != sigspec:
        raise ValueError('Training, APPLY and efficiency populations need identical sources')
    sigsel,bkgsel=meta['signal_selection'],meta['background_selection']
    if app['input_selection']['signal_selection'] != sigsel or app['input_selection']['background_selection'] != bkgsel:
        raise ValueError('Training/APPLY selection metadata disagree')
    mcapply=app['apply_selection']['mc_expression']; dataapply=app['apply_selection']['expression']
    fidexpr=app['draw_selection']['fiducial_cut']['expression']
    # DRAW fiducial selection is part of the diagnostic population too.
    mcapply=f'({mcapply}) and ({fidexpr})'; dataapply=f'({dataapply}) and ({fidexpr})'
    identity_path=Path(training_dir(tag))/'split_identity.json'
    old_evidence=Path('output/code_validation/add1_2v1_v1_setup_20260928')
    if identity_path.exists():
        identity=json.loads(identity_path.read_text()); expected=identity['sources']
        npz_path=identity_path.with_suffix('.npz')
        if sha256(npz_path)!=identity['npz_sha256']: raise ValueError('Split identity hash mismatch')
        frozen=dict(np.load(npz_path)); membership='recorded_at_training'
    else:
        if kind=='bdtg':raise ValueError('BDTG requires saved native split identities')
        # Explicit migration only for the seven models with a pre-submission revision record.
        plan=json.loads((old_evidence/'plan.json').read_text())
        if tag not in plan['tags']: raise ValueError('No original source/split provenance available')
        expected=json.loads((old_evidence/'validation.json').read_text())['source_revisions']
        frozen=None;membership='reconstructed_original_split_verified_against_saved_counts_and_three_AUCs'
    for r in expected: check_revision(r)
    revisions=[source_revision(s) for s in (sigspec,bkgspec)]
    for got,want in zip(revisions,expected):
        if got['path']!=want['path'] or got['tree']!=want['tree']: raise ValueError('Source identity mismatch')
        if want.get('sha256') and got['sha256']!=want['sha256']: raise ValueError('Source hash changed')
    columns=cfg['input_columns']
    sig=read_selected(sigspec,columns+['Bmass','pThatreweight']+selection_columns(sigsel)+selection_columns(mcapply),[sigsel,mcapply])
    bkg=read_selected(bkgspec,columns+['Bmass']+selection_columns(bkgsel)+selection_columns(dataapply),[bkgsel,dataapply])
    sigtrain=apply_selection(sig,sigsel,'signal training'); bkgtrain=apply_selection(bkg,bkgsel,'background training')
    ns,nb=len(sigtrain),len(bkgtrain)
    q=float(cfg['train_fraction'])
    if q!=float(meta['train_fraction']): raise ValueError('Split fraction mismatch')
    arrays=validated_split(kind,cfg,meta,sigtrain.index.to_numpy(),bkgtrain.index.to_numpy(),frozen)
    tr,te=arrays['train_index'],arrays['test_index']
    notes=meta['notes']
    if kind=='bdtg':
        train_sig_ids=tr[tr<ns]
        if not np.isclose(sigtrain.pThatreweight.iloc[train_sig_ids].sum(),notes['signal_train_weight_sum'],rtol=1e-12):
            raise ValueError('BDTG training weight sum mismatch')
        # Native TMVA fixes the class counts before a random shuffle. This is
        # the pre-cut background inclusion probability, never a post-cut ratio.
        q=float((tr>=ns).sum()/nb)
    else:
        if ns!=notes['signal_weight_entries']:raise ValueError('Signal population/count mismatch')
        if not np.isclose(sigtrain.pThatreweight.sum(),notes['signal_weight_sum'],rtol=1e-12): raise ValueError('Weight sum mismatch')
    modelpath=resolve_model_path(tag); scalerpath=resolve_scaler_path(tag)
    if kind=='bdtg':
        if sha256(modelpath)!=cfg['model_sha256'] or sha256(scalerpath)!=cfg['scaler_sha256']:
            raise ValueError('BDTG model/scaler changed')
    model=load_model(tag,cfg); scaler=joblib.load(scalerpath)
    if kind=='xgb':model.set_params(n_jobs=4)
    sig['_score']=predict(sig,cfg,scaler,model);bkg['_score']=predict(bkg,cfg,scaler,model)
    weights=np.r_[sigtrain.pThatreweight.to_numpy(),np.ones(nb)]
    scores=np.r_[sig.loc[sigtrain.index,'_score'],bkg.loc[bkgtrain.index,'_score']]
    labels=np.r_[np.ones(ns),np.zeros(nb)]
    if not np.isfinite(weights).all() or np.any(weights<=0): raise ValueError('Invalid MC weights')
    roc=json.loads((Path(training_dir(tag))/'roc.json').read_text()); auc_check={}
    for name,ids in [('train',tr),('test',te),('combined',np.arange(ns+nb))]:
        auc_check[name]=float(roc_auc_score(labels[ids],scores[ids],sample_weight=weights[ids]))
        if not np.isclose(auc_check[name],roc[name]['auc'],rtol=0,atol=1e-10):
            raise ValueError('Recovered split fails saved AUC check: '+name)
    # Membership is attached by original source entries, never by a new random assignment.
    sig['_split']=-1;bkg['_split']=-1
    for value,ids in [(0,tr),(1,te)]:
        sig.loc[sigtrain.index.to_numpy()[ids[ids<ns]],'_split']=value
        bkg.loc[bkgtrain.index.to_numpy()[ids[ids>=ns]-ns],'_split']=value
    sa=apply_selection(sig,mcapply,'MC apply');ba=apply_selection(bkg,dataapply,'DATA apply')
    # Compare recalculated full application against already delivered score files.
    for df,spec,name in [(sa,sigspec,'MC'),(ba,bkgspec,'DATA')]:
        path=Path(selected_dir(tag))/(name+'_with_score.root'); tree=split_root_spec(spec)[1]
        with uproot.open(path) as f: stored=f[tree].arrays(['Bmass','Prediction'],library='np')
        if len(df)!=len(stored['Bmass']) or not np.array_equal(df.Bmass.to_numpy(),stored['Bmass']) or not np.allclose(df['_score'],stored['Prediction'],rtol=0,atol=1e-7):
            raise ValueError('Recomputed population/scores differ from APPLY: '+name)
    sa=sa.loc[mass_mask(sa.Bmass.to_numpy(),[tuple(recipe['selection']['mc_mass'])])]
    bands=sidebands_from_selection(bkgsel)
    btrain=ba.loc[mass_mask(ba.Bmass.to_numpy(),bands)]
    if (sa['_split']<0).any() or (btrain['_split']<0).any():
        raise ValueError('Apply population missing true training membership; no extrapolated members allowed')
    scopes={};curves={}
    for scope in ['wide','training']:
        sb=recipe['scopes']['wide']['sidebands'] if scope=='wide' else bands
        bd=ba if scope=='wide' else btrain
        groups=['total'] if scope=='wide' else ['total','train','test']
        curves[scope]={}
        for group in groups:
            value={'train':0,'test':1}.get(group)
            sm=sa if value is None else sa.loc[sa['_split']==value]
            dm=bd if value is None else bd.loc[bd['_split']==value]
            probability={'total':1.,'train':q,'test':1-q}[group]
            curves[scope][group]=scan(sm['_score'].to_numpy(),sm.pThatreweight.to_numpy(),dm.Bmass.to_numpy(),dm['_score'].to_numpy(),sb,probability)
        scopes[scope]=dict(sidebands=sb,width_ratio=.06/sum(hi-lo for lo,hi in sb),
                           curves=curves[scope],optimal=optimal(curves[scope]['total']))
    plot=draw_config or resolve_draw_config('pbpb','X',year)['plot']
    mass_range=plot['mass_range'];bins=np.arange(mass_range[0],mass_range[1]+plot['bin_width'],plot['bin_width'])
    fid=app['draw_selection']['fid_profile'];cutdir=Path(cut_scan_dir(tag));cutdir.mkdir(parents=True,exist_ok=True)
    title='\n'.join(textwrap.wrap(tag,width=48,break_long_words=True,break_on_hyphens=False))
    for scope,content in scopes.items():
        fig,ax=plt.subplots(figsize=(7,5))
        for name,rows in content['curves'].items(): ax.plot([r['score'] for r in rows],[r['FOM'] for r in rows],label=name)
        cut=content['optimal']['score'];ax.axvline(cut,color='gray',ls='--',label=f'total optimal > {cut:.2f}')
        ax.set(xlabel='Score threshold',ylabel='Punzi FOM',xlim=(.05,.95),title=f'{title}\n{scope} sidebands');ax.grid(alpha=.3);ax.legend();fig.tight_layout()
        fig.savefig(out/f'punzi_{scope}.pdf');fig.savefig(out/f'punzi_{scope}.png',dpi=150);plt.close(fig)
        dc=ba.loc[(ba['_score']>cut)&(ba.Bmass>mass_range[0])&(ba.Bmass<mass_range[1])]
        fig,ax=plt.subplots(figsize=(6,6));ax.hist(dc.Bmass,bins=bins,histtype='step',linewidth=2)
        for mass in plot.get('reference_masses',[]):ax.axvline(mass,linestyle='--',linewidth=1.2,color='gray',alpha=.8)
        ax.set(xlabel='Bmass (GeV)',ylabel='Entries',xlim=mass_range,title=f'{title}\n{scope} Punzi optimal: score > {cut:.2f}')
        ax.title.set_fontsize(10);ax.grid(alpha=.3);fig.tight_layout()
        suffix='_wide_optimal' if scope=='wide' else '_optimal'
        name=f'DATA_{fid}_score{cut:.2f}'.replace('.', 'p')+suffix+'.pdf'
        fig.savefig(cutdir/name,bbox_inches='tight');plt.close(fig)
        content['mass_plot']=str(cutdir/name);content['mass_plot_entries']=len(dc)
        content['mass_plot_histogram']=np.histogram(dc.Bmass,bins=bins)[0].tolist()
    for revision in revisions:check_revision(revision)
    np.savez_compressed(out/'split_identity.npz',**arrays)
    payload=dict(status='PASS',train_tag=tag,recipe_id=recipe['recipe_id'],recipe_sha256=sha256(recipe_path),
        sources=revisions,membership=membership,split_identity_sha256=sha256(out/'split_identity.npz'),
        split_seed=cfg['split_random_state'],q_train=q,q_test=1-q,auc_reproduced=auc_check,
        model_type=kind,split_policy=cfg['split_policy'],nominal_train_fraction=cfg['train_fraction'],
        inclusion_probability_source='native pre-cut class counts' if kind=='bdtg' else 'original sklearn split fraction',
        mc_apply_selection=mcapply,data_apply_selection=dataapply,background_training_selection=bkgsel,
        weight='source pThatreweight; no class balance',model_sha256=sha256(modelpath),scaler_sha256=sha256(scalerpath),
        config_sha256=sha256(cfg_path),metadata_sha256=sha256(meta_path),apply_summary_sha256=sha256(apply_path),
        source_identity='source ROOT tree entry index, separate signal/background namespaces',
        split_counts=dict(signal_train=int((tr<ns).sum()),background_train=int((tr>=ns).sum()),
                          signal_test=int((te<ns).sum()),background_test=int((te>=ns).sum())),
        plot_config=plot,scopes=scopes)
    (out/'recipe.json').write_text(recipe_path.read_text())
    (out/'scan.json').write_text(json.dumps(payload,indent=2)+'\n')
    print(json.dumps(dict(tag=tag,status='PASS',optimal={s:c['optimal'] for s,c in scopes.items()}),indent=2))
    return payload


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('train_tag')
    run(parser.parse_args().train_tag)
