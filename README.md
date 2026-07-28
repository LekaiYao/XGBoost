# XGBoost Analysis Workflow

CMS Run 3 pp/PbPb 的 XGBoost 训练与 apply 工作流。正式主线执行：

```text
TRAIN → APPLY → DRAW → SHAP
```

XGBoost 最终向下游提供带 `Prediction` 的 DATA/MC ROOT，以及 `analysis_manifest.json`。

## 1. 正式提交：Direct XGBoost + SHAP

推荐始终显式写出两个开关：

```bash
bash dag/submit_single_workflow.sh <train_tag> 1 0
```

- 第一个 `1`：`with_shap=1`，在 DRAW 后运行 SHAP；
- 第二个 `0`：`use_precut=0`，读取标准 ROOT 输入。

带 reweighting 的训练通过 tag 中的 profile 从 `configs/samples.py` 自动解析 signal
ROOT、TTree 和 weight branch；提交命令不接受显式路径。例如：

```bash
bash dag/submit_single_workflow.sh \
  X_pp24_v4_fid3_8v2_rwpsi2sr5v1_xgb_v1 0 0
```

旧 tag 和显式 `rw0` 都表示无权重。weighted signal 的 SHAP 输入策略尚未固定，因此当前
先使用 `with_shap=0`。

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

Optuna 执行 `TRAIN → APPLY → DRAW`，当前不自动追加 SHAP。objective index `1` 表示 validation AUC。

## 7. 主要配置入口

| 内容 | 文件 |
|---|---|
| ROOT、TTree、selection、fid | `configs/samples.py` |
| varset | `utils/varsets.py` |
| Direct XGBoost 参数 | `configs/direct_xgb_settings.py` |
| Optuna 空间与 early stop | `configs/optuna_spaces.py` |
| 输出路径 | `utils/paths.py` |

## 8. 其他可选功能

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

## 9. 测试

```bash
.venv/bin/python -m unittest -v tests.test_export_analysis_manifest
```

```bash
.venv/bin/python -m py_compile \
  workflows/condor_optuna_XGBoost.py \
  workflows/batch_apply_scores.py \
  workflows/batch_draw_scores.py \
  workflows/shap_importance.py \
  workflows/integration/export_analysis_manifest.py \
  dag/make_dagman_workflow.py dag/make_single_workflow.py \
  utils/varsets.py utils/paths.py configs/samples.py
```
