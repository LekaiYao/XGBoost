"""Compare identical-sample add-one models using test and pooled weighted AUC."""
import argparse
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from configs.feature_selection import resolve_feature_selection_experiment
from utils.paths import model_config_path, model_dir, training_dir, experiment_output_dir


def dual_order(rows, baseline):
    orders = {metric: sorted(rows, key=lambda r: (-r[metric], r['train_tag'] != baseline, r['train_tag']))
              for metric in ('test_auc', 'combined_auc')}
    same = [r['train_tag'] for r in orders['test_auc']] == [r['train_tag'] for r in orders['combined_auc']]
    return orders, same


def main(experiment):
    cfg = resolve_feature_selection_experiment(experiment)
    assert cfg['operation'] == 'add1'
    items = [{'train_tag': cfg['baseline'], 'added_variable': ''}] + cfg['candidates']
    rows, rocs = [], {}
    base_config = base_meta = None
    for item in items:
        tag = item['train_tag']
        c = json.loads(Path(model_config_path(tag)).read_text())
        m = json.loads((Path(model_dir(tag))/'run_metadata.json').read_text())
        roc = json.loads((Path(training_dir(tag))/'roc.json').read_text());rocs[tag] = roc
        if base_config is None:
            base_config, base_meta = c, m
        else:
            assert c['input_columns'] == base_config['input_columns'] + [item['added_variable']]
        for key in ('model_params','split_random_state','train_fraction','signal_weight_branches'):
            assert c[key] == base_config[key], (tag,key)
        for key in ('signal_path','background_path','signal_selection','background_selection'):
            assert m[key] == base_meta[key], (tag,key)
        for key in ('n_sig_train','n_bkg_train','signal_weight_sum','signal_weight_entries'):
            assert m['notes'][key] == base_meta['notes'][key], (tag,key)
        rows.append(dict(train_tag=tag, added_variable=item['added_variable'],
                         train_auc=roc['train']['auc'], test_auc=roc['test']['auc'],
                         combined_auc=roc['combined']['auc'],
                         train_test_gap=roc['train']['auc']-roc['test']['auc']))
    orders, same = dual_order(rows,cfg['baseline'])
    for metric, order in orders.items():
        for i,r in enumerate(order,1):r[metric+'_rank']=i
    out=Path(experiment_output_dir('training',experiment));out.mkdir(parents=True,exist_ok=True)
    combined_primary = cfg.get('ranking_metric') == 'combined_weighted_auc'
    primary_order = orders['combined_auc'] if combined_primary else orders['test_auc']
    decision = 'COMBINED_AUC_PRIMARY_TEST_REFERENCE' if combined_primary else ('RANKINGS_AGREE' if same else 'HUMAN_REVIEW_REQUIRED')
    winner = primary_order[0]['train_tag'] if combined_primary or same else None
    payload=dict(experiment=experiment,baseline=cfg['baseline'],ranking_scope='common signal/background training selections recorded below',
                 signal_selection=base_meta['signal_selection'],background_selection=base_meta['background_selection'],
                 decision_status=decision,machine_recommendation=winner,rankings_agree=same,
                 test_order=[r['train_tag'] for r in orders['test_auc']],combined_order=[r['train_tag'] for r in orders['combined_auc']],
                 rows=primary_order,next_round_submitted=False)
    (out/'ranking.json').write_text(json.dumps(payload,indent=2)+'\n')
    with (out/'ranking.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(primary_order)
    for split in ('train','test','combined'):
        fig,ax=plt.subplots(figsize=(10,7))
        for row in primary_order:
            r=rocs[row['train_tag']][split]
            label=(row['added_variable'] or 'Baseline')+f" ({r['auc']:.5f})"
            ax.plot(r['tpr'],r['background_rejection'],label=label,linewidth=1)
        ax.set(xlabel='Signal efficiency',ylabel='Background rejection',xlim=(0,1),ylim=(0,1),title=f'{split}: weighted ROC, common training selection')
        ax.grid(alpha=.3);ax.legend(fontsize=7,loc='lower left',ncol=2);fig.tight_layout()
        fig.savefig(out/f'roc_{split}.pdf');fig.savefig(out/f'roc_{split}.png',dpi=150);plt.close(fig)
    from utils.feature_selection_report import update_feature_selection_report
    update_feature_selection_report(experiment)
    print(json.dumps(payload,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('experiment')
    main(parser.parse_args().experiment)
