# XGBoost Analysis Workflow

CMS Run 3 pp/PbPb 的 XGBoost 训练与 apply 工作流。正式主线执行：

```text
TRAIN → APPLY → DRAW → {SHAP, FIT_INTERFACE}
```

XGBoost 最终向下游提供带 `Prediction` 的 DATA/MC ROOT，以及 `analysis_manifest.json`。

## 1. 正式提交：Direct XGBoost + SHAP + fit interface

SHAP 和 fit-interface exporter 均为默认环节；以下命令等价于显式传入
`with_shap=1`、`with_fit_interface=1`：

```bash
bash dag/submit_single_workflow.sh <train_tag>
```

- 如需临时关闭，可显式传入 `with_shap=0`；
- 第六个参数 `with_fit_interface` 可显式设为 `0`，关闭拟合接口节点；
- `use_precut` 默认同样为 `0`，读取标准 ROOT 输入。

带 reweighting 的训练通过 tag 中的 profile 从 `configs/samples.py` 自动解析 signal
ROOT、TTree 和 weight branch；提交命令不接受显式路径。例如：

```bash
bash dag/submit_single_workflow.sh \
  X_pp24_v4_fid3_8v2_rwpsi2sr5v1_xgb_v1
```

旧 tag 和显式 `rw0` 都表示无权重。weighted signal 的 SHAP 使用与训练一致的 signal
selection 和配置中的 weight branch。

### pp reweight DAG

配置在 `workflows/reweighting/run_configured_job.py` 的 pp reweight 使用：

```bash
bash dag/submit_reweight_workflow.sh <reweight_tag> [with_splot_validation] [with_mc_domain_validation]
```

两个验证开关默认均为 `1`。DAG 默认执行 train、已配置年份的 PbPb apply、
sPlot/RW 验证和逐年份 ppRef/PbPb MC 验证；任一验证可用对应参数设为 `0` 跳过。
提交前会报告 sPlot 文件是否存在，验证节点运行时还会再次检查；若缺失，节点写入
`status=skipped_missing_splot` 的 manifest 后成功退出。验证输出位于
`output/reweighting/<reweight_tag>/validation/`。

### Bu pp24 示例

```bash
bash dag/submit_single_workflow.sh \
  Bu_pp24_v1_fid1_10v1_xgb_v1 1 0
```

执行完成后应产生：

```text
output/models/Bu_pp24_v1_fid1_10v1_xgb_v1/
output/training/Bu_pp24_v1_fid1_10v1_xgb_v1/
output/selected/Bu_pp24_v1_fid1_10v1_xgb_v1/
output/shap/Bu_pp24_v1_fid1_10v1_xgb_v1/
```

关键产物：

```text
models/<tag>/xgb_model.pkl
models/<tag>/scaler.pkl
models/<tag>/model_config.json
models/<tag>/run_metadata.json

training/<tag>/roc.pdf
training/<tag>/logloss.pdf
training/<tag>/ks_curve.pdf
training/<tag>/xgb_score.pdf

selected/<tag>/DATA_with_score.root
selected/<tag>/MC_with_score.root
selected/<tag>/cut_scan/*.pdf

shap/<tag>/shap_summary.pdf
shap/<tag>/shap_bar.pdf
shap/<tag>/shap_importance_fraction.json
```

### Bd PbPb 示例

Bd PbPb DATA 很大，必须启用 ROOT-native precut：

```bash
bash dag/submit_single_workflow.sh \
  Bd_pb24_v1_fid1_17v1_xgb_v1 1 1
```

- `with_shap=1`；
- `use_precut=1`。

precut 会生成 `train_background.root`、`apply_data.root` 及对应 metadata/fingerprint；配置未改变时复用，改变后自动重建。

## 2. 生成 Analysis_CODES 接口

TRAIN/APPLY 完成后执行：

```bash
.venv/bin/python -m workflows.integration.export_analysis_manifest <train_tag>
```

Bu 示例：

```bash
.venv/bin/python -m workflows.integration.export_analysis_manifest \
  Bu_pp24_v1_fid1_10v1_xgb_v1
```

输出：

```text
output/selected/Bu_pp24_v1_fid1_10v1_xgb_v1/analysis_manifest.json
```

exporter 会验证：

- DATA/MC ROOT 存在且可读；
- TTree 与配置一致；
- 两棵树都包含 `Prediction`。

manifest 包含 ROOT/TTree、channel/system、fid/selection、sidebands 和建议质量区间；不包含 Punzi 参数、最终 cut 或 fit model。XGBoost 不直接修改 Analysis_CODES。

DRAW 后的默认 `FIT_INTERFACE` 节点调用：

```bash
.venv/bin/python -m workflows.integration.export_default_fit_interface <train_tag>
```

dispatcher 只为 `configs/year_pairings.py` 中登记的 PbPb23/PbPb24 组合生成接口。
若当前 tag 未配置 pairing，或另一年份尚未完成，节点记录原因后成功跳过；若当前 tag
应有的 model/scored ROOT/weighted-efficiency thresholds 缺失，或两年配置不兼容，则失败。
两个年份的 DAG 都会执行该节点，因此无论哪一年最后完成，接口均会在 PbPb23 anchor 下生成。

X 当前默认接口为：

```text
output/selected/<pb23_anchor_tag>/fit_scan_manifest.pb23_pb24_simultaneous_mc_shape_nominal_v2.json
```

schema 位于
`workflows/integration/schemas/pbpb_x_simultaneous_year_mc_shape_nominal_fit_scan_v2.schema.json`。
manifest 交付两年 scored DATA/weighted MC、10%--40%（间隔 5%）的逐年 matched-efficiency
thresholds、完整 selection，以及“各年份先拟合 weighted MC 双 Gaussian shape，再 simultaneous
拟合 DATA”的 nominal contract。接口生成不提交 Analysis_CODES fit、不执行 toys，也不选择 WP。

历史单年 DATA-only `fit_scan_manifest.data_only_nominal_v*.json` 与早期
`fit_scan_manifest.json` 只作 provenance 保留，不是当前 paired-year nominal 接口。

## 3. 同名 tag 重新提交

重新提交前清理 AFS 中的旧 DAG lock/rescue/log：

```bash
bash scripts/cleanup/clear_dag_locks.sh <train_tag>
```

Bu 示例：

```bash
bash scripts/cleanup/clear_dag_locks.sh \
  Bu_pp24_v1_fid1_10v1_xgb_v1

bash dag/submit_single_workflow.sh \
  Bu_pp24_v1_fid1_10v1_xgb_v1 1 0
```

该清锁命令不删除 EOS `output/`。

## 4. 输出清理

### 删除旧 DATA/MC score ROOT

先 dry-run：

```bash
python3 scripts/cleanup/cleanup_selected_events.py --hours 120
```

确认后执行：

```bash
python3 scripts/cleanup/cleanup_selected_events.py --hours 120 --run
```

只删除：

```text
DATA_with_score.root
MC_with_score.root
```

模型、training PDF/JSON、draw PDF、SHAP 和 extra MC 保留。

### 整体归档

必须先 dry-run：

```bash
python3 scripts/cleanup/archive_output_outdated.py --dry-run
```

注意：正式运行默认先永久删除 `DATA_with_score.root`，再把目录移动到 `output/backup_outdate/`。如需保留 DATA ROOT：

```bash
python3 scripts/cleanup/archive_output_outdated.py \
  --keep-selected-data-root
```

归档仍在同一文件系统，移动本身不释放空间。

## 5. train tag 格式与当前可选项

Direct tag 格式：

```text
{channel}_{dataset}_v{selection}_fid{fid}_{varset}_xgb_v{model}
```

例如：

```text
Bu_pp24_v1_fid1_10v1_xgb_v1
```

含义：

- `Bu`：channel；
- `pp24`：dataset；
- `v1`：selection profile；
- `fid1`：fiducial profile；
- `10v1`：varset；
- `xgb_v1`：direct XGBoost model version。

### pp24

| Channel | selection | fid | varset |
|---|---|---|---|
| X | `v1,v2,v3` | `fid1,fid2` | `4v1,5v1,7v1,8v1,11v1,12v1,13v1,14v1,14v2,16v1,17v1,17v2,18v1` |
| Bu | `v1` | `fid1` | `4v1,5v1,10v1,12v1` |
| Bd | `v1` | `fid1` | `9v1,14v1,17v1` |
| Bs | `v1,v2,v3` | `fid1,fid2` | `7v1,14v1,17v1` |

### pb24

| Channel | selection | fid | varset |
|---|---|---|---|
| X | `v1,v2,v3,v4,v5` | `fid1,fid2,fid3,fid4,fid5` | `5v2,6v4,7v1,8v1,9v1,11v1,12v1,13v1,14v1,14v2,16v1,17v1,17v2,18v1` |
| Bu | `v1` | `fid1` | `4v1,5v1,10v1,12v1` |
| Bd | `v1` | `fid1` | `9v1,14v1,17v1` |
| Bs | `v1,v2,v3,v4` | `fid1,fid2` | `6v1,7v1,8v1,14v1,17v1` |

### pb23

| Channel | selection | fid | varset |
|---|---|---|---|
| X | `v1,v2` | `fid1,fid2` | 与 X PbPb varset 相同 |
| Bu | `v1` | `fid1` | `4v1,5v1,10v1,12v1` |
| Bd | `v1` | `fid1` | `9v1,14v1,17v1` |
| Bs | `v1` | `fid1` | `6v1,7v1,8v1,14v1,17v1` |

selection、fid 和 varset 的实际定义分别以：

```text
configs/samples.py
utils/varsets.py
```

为准。表中选项可以通过格式校验，但正式训练前仍应确认所选 selection 与 fid 的物理定义相匹配。

## 6. Optuna 主线（可选）

显式单 tag：

```bash
bash dag/submit_dagman_workflow.sh \
  X_pb24_v2_fid1_8v1_1o200_v1 auto
```

旧批量展开：

```bash
bash dag/submit_dagman_workflow.sh \
  X_pb24_v2_fid1_8v1_1o200 1 10 auto
```

Optuna 默认执行 `TRAIN → APPLY → DRAW → {SHAP, FIT_INTERFACE}`；可分别用
`with_shap=0`、`with_fit_interface=0` 关闭对应节点。objective index `1` 表示 validation AUC。

## 7. 主要配置入口

| 内容 | 文件 |
|---|---|
| ROOT、TTree、selection、fid | `configs/samples.py` |
| varset | `utils/varsets.py` |
| Direct XGBoost 参数 | `configs/direct_xgb_settings.py` |
| Optuna 空间与 early stop | `configs/optuna_spaces.py` |
| 输出路径 | `utils/paths.py` |

## 8. 汇总 selected tags

从 `output/selected/` 的直接子目录重新生成配置摘要：

```bash
python3 scripts/summarize_selected_tags.py
```

默认写入 `docs/selected_tag_summary.md`；也可用 `--selected-dir` 和 `--output` 指定输入、输出。
脚本优先采用各 tag 已落盘的 `model_config.json`、`run_metadata.json` 和
`batch_apply_summary.json`，并用当前 resolver 补全及检查差异。任一 tag 无法解析时不会写出不完整摘要。

## 9. 其他可选功能

X channel extra MC：

```bash
.venv/bin/python -m workflows.batch_apply_scores \
  --apply-extra-mc all X_pp24_v3_fid2_4v1_xgb_v1
```

从 SHAP 95% 累计重要度生成新 varset：

```bash
.venv/bin/python scripts/update_varset_from_shap95.py <train_tag>
```

旧 `workflows/optimization/run_optimization_from_tag.py` 只作为 compatibility helper 保留，不再扩展。

## 10. 测试

```bash
.venv/bin/python -m unittest -v \
  tests.test_export_analysis_manifest \
  tests.test_export_x_fit_scan_manifest
```

```bash
.venv/bin/python -m py_compile \
  workflows/condor_optuna_XGBoost.py \
  workflows/batch_apply_scores.py \
  workflows/batch_draw_scores.py \
  workflows/shap_importance.py \
  workflows/integration/export_analysis_manifest.py \
  workflows/integration/export_x_fit_scan_manifest.py \
  dag/make_dagman_workflow.py dag/make_single_workflow.py \
  utils/varsets.py utils/paths.py configs/samples.py
```
