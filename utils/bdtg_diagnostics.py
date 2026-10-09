"""Standard diagnostic exports for a fixed TMVA BDTG; no model selection."""
import json
from pathlib import Path
import tempfile
import numpy as np
import uproot
from utils import paths
from utils.model_backend import run_tmva
from utils.training_identity import sha256
from utils.training_weights import weighted_ks_curve


def weighted_logloss(labels, margin, weights):
    labels=np.asarray(labels,dtype=float);margin=np.asarray(margin,dtype=float)
    return float(np.average(np.logaddexp(0.,margin)-labels*margin,weights=weights))


def export_training_diagnostics(tag, transformed, labels, weights, train, test, score, features):
    out=Path(paths.training_dir(tag));model=Path(paths.model_path(tag))
    train=np.asarray(train);test=np.asarray(test);labels=np.asarray(labels,dtype=bool);weights=np.asarray(weights,dtype=float)
    np.testing.assert_array_equal(np.sort(np.r_[train,test]),np.arange(len(labels)))
    if not np.isfinite(weights).all() or (weights<=0).any():raise ValueError("Invalid physical diagnostic weights")
    is_train=np.zeros(len(labels),dtype=bool);is_train[train]=True
    raw_path=out/'diagnostic_history.root'
    with tempfile.TemporaryDirectory(prefix='bdtg_diagnostics_') as scratch:
        source=Path(scratch)/'input.root'
        data={c:transformed[c].to_numpy(dtype=np.float32) for c in transformed.columns}
        data.update(label=labels,is_train=is_train,physical_weight=weights)
        with uproot.recreate(source) as root:root.mktree('events',data)
        log=run_tmva('diagnostics',source,raw_path,list(transformed.columns),xml=model,cwd=scratch)
        (out/'diagnostics.log').write_text(log)
    with uproot.open(raw_path) as root:
        history=root['history'].arrays(library='np');pred=root['scores'].arrays(library='np')
    n=len(history['round']);np.testing.assert_array_equal(history['round'],np.arange(1,n+1))
    np.testing.assert_allclose((pred['raw']+1)/2,score,rtol=0,atol=2e-12)
    for name,ids in [('train',train),('test',test)]:
        np.testing.assert_allclose(history[name+'_logloss'][-1],weighted_logloss(labels[ids],pred['margin'][ids],weights[ids]),rtol=1e-11,atol=1e-13)
    with uproot.open(out/'tmva_training.root') as root:importance=root['native_importance'].arrays(library='np')
    np.testing.assert_array_equal(importance['feature_index'],np.arange(len(features)))
    values=importance['importance'];np.testing.assert_allclose(values.sum(),1.,rtol=0,atol=1e-12)
    if not np.isfinite(values).all() or (values<0).any():raise ValueError('Invalid TMVA importance')
    hist=dict(train_logloss=history['train_logloss'].tolist(),test_logloss=history['test_logloss'].tolist(),
        best_iteration=n,n_estimators=n,eval_metric='logloss',weight_definition='physical MC weights; background=1; no class-balance multiplier',
        score_definition='(TMVA raw+1)/2 = logistic(2*sum(leaf responses)); no probability calibration',
        model_selection='fixed approved tree count; no early stopping or optimization')
    (out/'training_history.json').write_text(json.dumps(hist,indent=2)+'\n')
    order=np.argsort(-values);rows=[dict(rank=i+1,feature=features[j],importance=float(values[j])) for i,j in enumerate(order)]
    (out/'feature_importance.json').write_text(json.dumps(rows,indent=2)+'\n')
    ks={name:weighted_ks_curve(labels[ids],score[ids],weights[ids]) for name,ids in [('train',train),('test',test)]}
    (out/'ks.json').write_text(json.dumps(ks,indent=2)+'\n')
    np.savez_compressed(out/'diagnostic_predictions.npz',score=score,margin=pred['margin'],labels=labels,physical_weights=weights)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    with plt.rc_context({'font.family':'serif','font.size':12,'legend.frameon':False}):
        fig,ax=plt.subplots(figsize=(7,5))
        for name in ['train','test']:ax.plot(history['round'],history[name+'_logloss'],label=name)
        ax.set(xlabel='Boosting round',ylabel='Physical-weight log-loss');ax.legend();ax.grid(alpha=.25);fig.tight_layout();fig.savefig(paths.condor_training_logloss_path(tag));plt.close(fig)
        fig,ax=plt.subplots(figsize=(7,5))
        for name,color in [('train','tab:blue'),('test','tab:orange')]:
            k=ks[name]
            for group,ls in [('sig','-'),('bkg','--')]:ax.plot(k['score_thresholds'],k[group+'_cdf'],color=color,linestyle=ls,label=f'{name} {group}')
        ax.text(.03,.97,'Train KS = %.3f\nTest KS = %.3f'%(ks['train']['ks_stat'],ks['test']['ks_stat']),ha='left',va='top',transform=ax.transAxes)
        ax.set(xlabel='Mapped BDTG score',ylabel='Cumulative fraction',xlim=(0,1),ylim=(0,1));ax.legend(loc='lower right');ax.grid(alpha=.25);fig.tight_layout();fig.savefig(paths.condor_training_ks_curve_path(tag));plt.close(fig)
        fig,ax=plt.subplots(figsize=(6,6))
        for group,name in [(True,'signal'),(False,'background')]:
            ids=test[labels[test]==group];ax.hist(score[ids],weights=weights[ids],bins=np.linspace(0,1,100),density=True,histtype='step',label=name)
        ax.set(xlabel='Mapped BDTG score (raw + 1) / 2',ylabel='Probability density',xlim=(0,1));ax.legend();fig.tight_layout();fig.savefig(paths.condor_training_score_path(tag));plt.close(fig)
        fig,ax=plt.subplots(figsize=(8,5));x=np.arange(len(order));ax.bar(x,values[order],alpha=.7,label='TMVA importance');ax.set_xticks(x,[features[j] for j in order]);ax.set_ylabel('TMVA native variable importance');ax2=ax.twinx();ax2.plot(x,np.cumsum(values[order])*100,'o-',color='black');ax2.set(ylabel='Cumulative (%)',ylim=(0,105));fig.tight_layout();fig.savefig(paths.condor_feature_importance_cumulative_path(tag));plt.close(fig)
    metadata=dict(status='PASS',train_tag=tag,model_sha256=sha256(model),n_estimators=n,
        weight_definition=hist['weight_definition'],importance_definition='TMVA MethodBDT::GetVariableImportance on live trained forest; not XGBoost gain',
        final_score_max_error=float(np.max(np.abs((pred['raw']+1)/2-score))),
        files={name:sha256(out/name) for name in ['training_history.json','feature_importance.json','ks.json','diagnostic_predictions.npz','diagnostic_history.root','logloss.pdf','ks_curve.pdf','xgb_score.pdf','feature_importance_cumulative.pdf']},
        score_filename_note='xgb_score.pdf retained as existing shared workflow filename; axes identify BDTG')
    (out/'training_diagnostics.json').write_text(json.dumps(metadata,indent=2)+'\n')
    return metadata


def validate_training_diagnostics(tag):
    out=Path(paths.training_dir(tag));meta=json.loads((out/'training_diagnostics.json').read_text())
    cfg=json.loads(Path(paths.model_config_path(tag)).read_text())
    if meta['status']!='PASS' or meta['model_sha256']!=sha256(paths.model_path(tag)):raise ValueError('Training diagnostics/model mismatch')
    if meta['n_estimators']!=cfg['model_params']['NTrees']:raise ValueError('Diagnostic tree count mismatch')
    for name,digest in meta['files'].items():
        if sha256(out/name)!=digest:raise ValueError('Diagnostic file changed: '+name)
    pred=np.load(out/'diagnostic_predictions.npz');split=np.load(out/'split_identity.npz')
    hist=json.loads((out/'training_history.json').read_text());ks=json.loads((out/'ks.json').read_text())
    for name in ['train','test']:
        ids=split[name+'_index'];loss=np.asarray(hist[name+'_logloss'])
        if len(loss)!=meta['n_estimators'] or not np.isfinite(loss).all() or (loss<0).any():raise ValueError('Invalid history')
        np.testing.assert_allclose(loss[-1],weighted_logloss(pred['labels'][ids],pred['margin'][ids],pred['physical_weights'][ids]),rtol=1e-11)
        recalculated=weighted_ks_curve(pred['labels'][ids],pred['score'][ids],pred['physical_weights'][ids])
        for key in recalculated:np.testing.assert_allclose(ks[name][key],recalculated[key],rtol=0,atol=1e-12)
    return meta
