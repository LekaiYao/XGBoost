"""Rebuild the human-readable add/drop ledger from saved ranking results."""
import fcntl
import json
import os
from pathlib import Path
import shutil
import tempfile
from datetime import datetime, timezone


def render_experiment(payload, config, metadata, source, target):
    rows = payload.get('rows', payload.get('ranking'))
    baseline = payload['baseline']
    assert rows and sum(r['train_tag'] == baseline for r in rows) == 1
    assert len({r['train_tag'] for r in rows}) == len(rows)
    order = lambda metric: sorted(rows, key=lambda r: (-r[metric], r['train_tag'] != baseline, r['train_tag']))
    test, combined = order('test_auc'), order('combined_auc')
    combined_primary = config.get('ranking_metric') == 'combined_weighted_auc'
    primary, reference = (combined, test) if combined_primary else (test, combined)
    ranks = {r['train_tag']: i for i, r in enumerate(reference, 1)}
    ordering_note = '按总和 AUC（pooled train+test）降序排列；test AUC 仅供参考' if combined_primary else '按 test AUC 排列'
    base = next(r for r in rows if r['train_tag'] == baseline)
    operation = config['operation']
    same = [r['train_tag'] for r in test] == [r['train_tag'] for r in combined]
    link = lambda p: os.path.relpath(p, target.parent)
    lines = [f"## {payload['experiment']}", '', f"起点：`{baseline}`。",
        '', '起点输入：`[' + ', '.join(metadata['input_columns']) + ']`。', '',
        f"操作：{operation}；{len(rows)-1} 个候选，表中包含起点；{ordering_note}。",
        '', f"信号训练筛选：`{metadata['signal_selection']}`。",
        '', f"背景训练筛选：`{metadata['background_selection']}`。",
        '', f"训练参数：`{json.dumps(metadata['fixed_model_params'], ensure_ascii=False, sort_keys=True)}`。",
        '', f"当前排序规则：`{config.get('ranking_metric', payload.get('metric', '未记录'))}`。",
        '', ("排名依据为总和 AUC；test 排序差异仅供参考，不影响排名。" if combined_primary else f"排序任务结束时状态：`{payload.get('decision_status', '未记录')}`；该状态不代表后续人工决定。"),
        '', f"test 与 train+test 完整顺序：{'一致' if same else '不一致'}。",
        '', f"test 第一名：`{test[0]['train_tag']}`；train+test 第一名：`{combined[0]['train_tag']}`。",
        '', f"[原始 JSON]({link(source)}) · [完整 CSV]({link(source.with_suffix('.csv'))})", '',
        ('| 总和AUC排名 | Test参考排名 |' if combined_primary else '| Test排名 | 合并排名 |') + ' 完整tag | 新增／删除变量 | Train AUC | Test AUC（参考） | Train+test AUC | Train−test | ΔTest对起点 | Δ合并对起点 |',
        '|---:|---:|---|---|---:|---:|---:|---:|---:|---:|']
    for i, r in enumerate(primary, 1):
        variable = '起点' if r['train_tag'] == baseline else ('+ ' + r['added_variable'] if operation == 'add1' else '− ' + r['removed_variable'])
        lines.append(f"| {i} | {ranks[r['train_tag']]} | `{r['train_tag']}` | {variable} | {r['train_auc']:.8f} | {r['test_auc']:.8f} | {r['combined_auc']:.8f} | {r['train_auc']-r['test_auc']:.8f} | {r['test_auc']-base['test_auc']:+.8f} | {r['combined_auc']-base['combined_auc']:+.8f} |")
    plots = [source.parent/f'roc_{split}.pdf' for split in ('train','test','combined')]
    links = [f'[{p.stem}]({link(p)})' for p in plots if p.exists()]
    if links: lines += ['', 'ROC：' + ' · '.join(links)]
    return '\n'.join(lines)


def update_feature_selection_report(experiment=None, report_dir="output/feature_selection_reports"):
    """Write derived reports to an explicit artifact directory; never modify session docs by default."""
    if experiment is not None:
        from configs.feature_selection import resolve_feature_selection_experiment
        from utils.paths import experiment_output_dir, model_dir
        cfg = resolve_feature_selection_experiment(experiment)
        source = Path(experiment_output_dir('training', experiment))/'ranking.json'
        payload = json.loads(source.read_text())
        assert payload['experiment'] == experiment and payload['baseline'] == cfg['baseline']
        metadata = json.loads((Path(model_dir(cfg['baseline']))/'run_metadata.json').read_text())
        target = Path(report_dir)/experiment/'results.md'
        target.parent.mkdir(parents=True, exist_ok=True)
        text = '# X PbPb 变量选择结果\n\n仅记录本实验；排序口径见下表，train/test/总 AUC 同表保留。\n\n' + render_experiment(payload, cfg, metadata, source, target) + '\n'
        with (target.parent/'.results.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if target.exists() and target.read_text() == text:
                return target
            if target.exists():
                backup = Path('output/code_validation/feature_selection_report_backups')/experiment
                backup.mkdir(parents=True, exist_ok=True)
                shutil.copy2(target, backup/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'.md'))
            fd, name = tempfile.mkstemp(prefix='.results.', dir=target.parent)
            try:
                with os.fdopen(fd, 'w') as stream:
                    stream.write(text)
                os.chmod(name, target.stat().st_mode & 0o777 if target.exists() else 0o644)
                os.replace(name, target)
            finally:
                if os.path.exists(name):
                    os.unlink(name)
        return target
    from configs.feature_selection import FEATURE_SELECTION_EXPERIMENTS, resolve_feature_selection_experiment
    from utils.paths import experiment_output_dir, model_dir
    target = Path(report_dir)/'results.md'
    target.parent.mkdir(parents=True, exist_ok=True)
    with (target.parent/'.results.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        sections = []
        # User requested this ledger start at the 2026-09-20 2v1 add-one line.
        # Future add/drop experiments are included automatically; these two
        # earlier experiments remain available only in their original outputs.
        historical = {
            "fs_7v10_drop1_X_pb23_v27_fid18_7v10_rw0_xgb_v1",
            "fs_9v9_drop1_X_pb23_v27_fid18_9v9_rw0_xgb_v1",
        }
        sources = [(name, Path(experiment_output_dir('training', name))/'ranking.json')
                   for name, config in FEATURE_SELECTION_EXPERIMENTS.items()
                   if name not in historical and not config.get('retired')
                   and not config.get('report_archived')]
        sources = [(name, source) for name, source in sources if source.is_file()]
        sources.sort(key=lambda item: (item[1].stat().st_mtime_ns, item[0]), reverse=True)
        for name, source in sources:
            if name in historical:
                continue
            cfg = resolve_feature_selection_experiment(name)
            source = Path(experiment_output_dir('training', name))/'ranking.json'
            if not source.is_file(): continue
            payload = json.loads(source.read_text())
            assert payload['experiment'] == name and payload['baseline'] == cfg['baseline']
            metadata = json.loads((Path(model_dir(cfg['baseline']))/'run_metadata.json').read_text())
            sections.append(render_experiment(payload, cfg, metadata, source, target))
        text = '\n'.join(['# X PbPb 变量增减结果汇总', '',
            '本文件由 add/drop 排序程序完成后自动重建；以各轮 ranking.json 为数值来源，按其更新时间倒序排列（最新轮在上）。请勿直接编辑自动生成内容。', '',
            '完整 tag 区分不同 selection、fiducial region、变量及超参数。不同轮的样本定义可能不同，不能直接把跨轮 AUC 差值解释为新增变量收益。', '',
            '本汇总只收录活动实验；退役实验通过其归档产物与冻结配置查阅。排序依据以各轮配置为准；Train+test AUC 来自合并事例的分数和权重，不是两份 AUC 的算术平均。', '',
            '各轮训练与应用筛选以该轮metadata及apply summary为准，不预设新配置。', '',
            '本汇总不自动批准或提交下一轮；后续操作需要单独确认。', '',
            f'已收录 {len(sections)} 轮。', '', *sections, ''])
        if target.exists() and target.read_text() == text: return target
        if target.exists():
            backup = Path('output/code_validation/feature_selection_report_backups')
            backup.mkdir(parents=True, exist_ok=True)
            shutil.copy2(target, backup/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'.md'))
        fd, name = tempfile.mkstemp(prefix='.results.', dir=target.parent)
        try:
            with os.fdopen(fd, 'w') as stream: stream.write(text)
            if target.exists(): shutil.copymode(target, name)
            else: os.chmod(name, 0o644)
            os.replace(name, target)
        finally:
            if os.path.exists(name): os.unlink(name)
    return target


if __name__ == '__main__':
    print(update_feature_selection_report())
