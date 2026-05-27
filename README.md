# XGBoost Analysis Workflow

## 项目目标
统一 PbPb 与 pp 分析训练/应用/画图工作流，避免在脚本内硬编码路径与 cut。

## 目录结构
- `configs/`：样本、路径、TTree、fid/train cuts、搜索空间
- `workflows/`：训练/应用/画图/SHAP 与调度脚本
- `wrappers/`：Condor 节点 wrapper（只做环境与参数转发）
- `dag/`：DAG 生成与提交
- `utils/`：varsets、路径、metadata
- `output/`：统一输出目录

## 主线工作流
### 主线 DAGMan（批量）
```bash
bash dag/submit_dagman_workflow.sh <group_tag> <version_start> <version_end> [fid_profile]
```
执行链：`make_dagman_workflow.py -> train_dispatch.py -> (condor_optuna_XGBoost.py / staged_optuna_pipeline.py) -> group_apply_draw.py -> batch_apply_scores.py + batch_draw_scores.py`
（Condor 层通过 `wrappers/run_train_dispatch.sh`、`wrappers/run_group_apply_draw.sh` 调用）

参数说明：
- `group_tag`：同一批任务共享前缀，必须为 `{channel}_...`，例如 `X_pb24v2_8v_4o200`、`Bu_pp24v2_6v4_o200`
- `version_start`：起始版本号，整数，例如 `1`
- `version_end`：结束版本号，整数且 `>= version_start`，例如 `10`
- `fid_profile`：apply/draw 使用的 fid 配置名（定义在 `configs/samples.py`），例如 `fid` 或 `fid3`
- 自动解析：`group_tag` 内必须包含 Optuna trial 信息（例如 `_o200` 或 `_4o200`），并由前缀自动推断 `dataset_year` 和 `selection_profile`

当前命名到筛选映射（实际行为）：
- `pb24v1_*`：`selection_profile=pb24v1`，`fid_profile=fid`
- `pb24v2_*`：`selection_profile=pb24v2`，`fid_profile=fid3`
- `pb23v6_*`：`fid_profile=fid3`
- 其它命名：回退到各 channel 的默认 profile

示例（核心提交命令）：
```bash
# PbPb：提交 v1-v10，共10个训练 + 1个最终apply/draw
bash dag/submit_dagman_workflow.sh X_pb24v2_8v_4o200 1 10 fid3

# PbPb：提交 v1-v100
bash dag/submit_dagman_workflow.sh Bu_pb24v2_8v_4o200 1 100 fid3

# pp：提交 v1-v20
bash dag/submit_dagman_workflow.sh Bd_pp24v2_6v4_o200 1 20 fid
```

### 不用 Optuna 的 XGBoost 主线（单模型 DAG）
命名约定：`train_tag` 包含 `_xgb_` 时，TRAIN 节点自动走 `workflows/xgboost_train_direct.py`。
```bash
bash dag/submit_single_workflow.sh X_pb24v2_8v_xgb_v1 0
bash dag/submit_single_workflow.sh Bs_pp24v2_6v4_xgb_v1 0
```
执行链：`TRAIN(Direct XGBoost) -> APPLY -> DRAW`。
（Condor 层通过 `wrappers/run_train_job.sh`、`wrappers/run_apply_job.sh`、`wrappers/run_draw_job.sh` 调用）

## 如何输入 Optuna 超参数空间与训练配置
- Optuna 空间：
  - 文件：[configs/search_spaces.py](/eos/home-l/leyao/pbpb_work/X_analysis/XGBoost/configs/search_spaces.py)
  - 字段：`OPTUNA_SPACES["pbpb"]`、`OPTUNA_SPACES["pp"]`
  - 主线脚本按 sample 自动读取，不再从命令行传空间名。
- 直接 XGBoost（非 Optuna）参数：
  - 同文件的 `DIRECT_XGB_PARAMS["pbpb"/"pp"]`
- 样本路径、TTree、训练筛选、fid cut：
  - 文件：[configs/samples.py](/eos/home-l/leyao/pbpb_work/X_analysis/XGBoost/configs/samples.py)
- `samples.py` 核心结构：
  - `SAMPLES[sample]["channels"][channel]["datasets"][year]["train/apply/draw"]`
  - `selection_profiles`：训练筛选
  - `fiducial_profiles`：apply/draw fid cut
- draw 输入约定：
  - draw 读取的是 apply 产物 `output/selected/<train_tag>/DATA_with_score.root`
  - 读取树名来自 `datasets[year].draw.data.tree`（按 channel 独立）
- `selection_profiles` 控制 train cut
- `fiducial_profiles` 控制 apply/draw cut（由 `fid_profile` 选择）
  - 以上均按 `sample + channel` 独立配置
- 变量组合：
  - 文件：[utils/varsets.py](/eos/home-l/leyao/pbpb_work/X_analysis/XGBoost/utils/varsets.py)

## 如何修改关键配置
- 修改 varset：编辑 [utils/varsets.py](/eos/home-l/leyao/pbpb_work/X_analysis/XGBoost/utils/varsets.py)
  - 结构：`VARSETS[sample][channel][varset]`
  - 示例：`VARSETS["pbpb"]["X"]["4v2"]`、`VARSETS["pbpb"]["Bu"]["4v2"]`、`VARSETS["pp"]["X"]["4v2"]`
- 修改输入 ROOT/TTree：编辑 [configs/samples.py](/eos/home-l/leyao/pbpb_work/X_analysis/XGBoost/configs/samples.py) 的 `datasets`（`train/apply/draw`）
- 修改训练筛选条件：编辑 `configs/samples.py` 的 `selection_profiles`
- 修改 apply/draw fiducial region：编辑 `configs/samples.py` 的 `fiducial_profiles`
- 修改 Optuna 搜索空间：编辑 [configs/search_spaces.py](/eos/home-l/leyao/pbpb_work/X_analysis/XGBoost/configs/search_spaces.py) 的 `OPTUNA_SPACES`
- 修改无 Optuna 训练参数：编辑 `configs/search_spaces.py` 的 `DIRECT_XGB_PARAMS`

## 下一步改造（命名显式映射）
- 当前仍存在“前缀推断 profile”的隐式行为。
- 下一步计划：将命名规则扩展为可显式指定训练筛选与画图 fid 的字段，并在解析阶段强校验，确保“名称即配置”。

### 单模型 DAG
```bash
bash dag/submit_single_workflow.sh <train_tag> [with_shap]
```
执行链：`TRAIN -> APPLY -> DRAW -> (optional SHAP)`，`.sub` 直接调用 `workflows/*.py`。

## SHAP 运行命令
### 1) 作为 single DAG 的一部分运行 SHAP
将 `with_shap` 设为 `1`，DAG 会在 `DRAW` 后追加 `SHAP` 节点：
```bash
bash dag/submit_single_workflow.sh <train_tag> 1
```
示例：
```bash
bash dag/submit_single_workflow.sh X_pp24v2_6v4_xgb_v1 1
```

### 2) 对已完成训练单独运行 SHAP
不重跑 train/apply/draw，直接针对已有模型做 SHAP：
```bash
.venv/bin/python -m workflows.shap_importance <train_tag> [max_events]
```
示例：
```bash
.venv/bin/python -m workflows.shap_importance X_pp24v2_6v4_xgb_v1
.venv/bin/python -m workflows.shap_importance Bu_pb24v1_5v2_xgb_v1 30000
```

### SHAP 输入筛选说明（当前实现）
- SHAP 输入样本与训练保持一致：
  - signal/background 使用 `configs/samples.py` 对应 `train` 输入
  - background 先做 sideband 质量窗，再应用同一套 `train_cut`
  - signal 也应用同一套 `train_cut`
- 即 SHAP 现在严格复用训练筛选逻辑，不再使用宽松/不一致筛选。

## Score 图与 Cut Scan（当前实现）
- direct XGBoost 训练（`workflows/xgboost_train_direct.py`）现在会输出真实的 `xgb_score.pdf`（分数分布图），不再把 JSON 写入 PDF 文件。
- 对应 ROC/AUC 指标输出到 `output/training/<train_tag>/test_roc.json`。
- draw 的 score cut 扫描点（`workflows/batch_draw_scores.py`）固定为：
  - `0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.90, 0.95`

## 命名规范
- 普通训练：`<sample>_<varset>_v<version>`
- Optuna 训练：`<sample>_<varset>_o<optunaTrials>_v<version>`

## 关键约束
- ROOT/TTree/fid/train cuts 只在 `configs/samples.py` 配置。
- `workflows/*` 只读配置，不再硬编码路径和 cut。
- 历史脚本仅在 `workflow_archive/` 存档，不参与主流程。
- 旧输入标签（无 channel 前缀）已禁用。

## 输出与排障
- draw 输出目录（pp/pbpb 一致）：`output/selected/<train_tag>/cut_scan/`
- draw 输入文件：`output/selected/<train_tag>/DATA_with_score.root`（来自 apply）
- `batch_apply_summary.json` 字段：
  - `input_datasets`（输入数据集）
  - `input_selection`（输入时筛选条件）
  - `draw_selection`（画图时筛选条件）
  - `training_varset`（训练 varset）

手动检查失败任务时优先看：
- `/afs/cern.ch/user/l/leyao/private/pbpb_work/X_analysis/XGBoost/logs/job_<train_tag>_{train,apply,draw}.err`
- `/afs/cern.ch/user/l/leyao/private/pbpb_work/X_analysis/XGBoost/logs/job_<train_tag>_{train,apply,draw}.log`
- `/afs/cern.ch/user/l/leyao/private/pbpb_work/X_analysis/XGBoost/logs/job_<train_tag>_{train,apply,draw}.out`

## 清理与归档
- 清理脚本统一放在 `scripts/cleanup/`。
- 旧入口 `cleanup_selected_events.py` 仅做兼容转发。

1) 清理 `output/selected` 下旧目录并删除大 ROOT：
```bash
python3 scripts/cleanup/cleanup_selected_events.py --days 5 --root-threshold-mb 500
```

2) 全量归档当前输出到 `output/backup_outdate/<timestamp>/`：
- 归档前会删除每个 tag 的 `output/selected/<tag>/DATA_with_score.root`（可用 `--keep-selected-data-root` 关闭）。
- 归档过程中会自动调用 `scripts/cleanup/clear_dag_locks.sh` 清理 single DAG 重提冲突文件（含 `.dag.condor.sub/.dag.lib.out/.dag.lib.err/.dag.dagman.log` 等）。
```bash
python3 scripts/cleanup/archive_output_outdated.py
python3 scripts/cleanup/archive_output_outdated.py --dry-run
```

3) single DAG 重提前仅清理锁文件：
```bash
bash scripts/cleanup/clear_dag_locks.sh <train_tag>
```
