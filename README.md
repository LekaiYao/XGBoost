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
bash dag/submit_dagman_workflow.sh <group_tag_with_{n}o{N}_v{k}> [fid_profile]
bash dag/submit_dagman_workflow.sh <group_tag_with_{n}o{N}> <version_start> <version_end> [fid_profile]
```
执行链：`make_dagman_workflow.py -> train_dispatch.py -> condor_optuna_XGBoost.py -> group_apply_draw.py -> batch_apply_scores.py + batch_draw_scores.py`
（Condor 层通过 `wrappers/run_train_dispatch.sh`、`wrappers/run_group_apply_draw.sh` 调用）

参数说明：
- 显式模式（不展开版本）：
  - `group_tag`：`{channel}_{dataset}_v{n}_fid{n}_{varset}_{n}o{N}_v{k}`
  - 命令：`bash dag/submit_dagman_workflow.sh <group_tag> [fid_profile]`
  - 行为：不追加 train_tag 后缀，直接按输入标签运行一组 `TRAIN->APPLY->DRAW`
- 旧批量模式（展开版本）：
  - `group_tag`：`{channel}_{dataset}_v{n}_fid{n}_{varset}_{n}o{N}`
  - 命令：`bash dag/submit_dagman_workflow.sh <group_tag> <version_start> <version_end> [fid_profile]`
  - 行为：按 `version_start..version_end` 展开训练节点，并生成 `..._v<run>` 的 train_tag
- `fid_profile`：可选覆盖值；默认从 `group_tag` 解析的 `dataset+fid{n}` 自动确定

示例（核心提交命令）：
```bash
# 显式模式（不展开）
bash dag/submit_dagman_workflow.sh X_pb24_v2_fid1_8v1_1o200_v1 auto

# 旧批量模式（展开 v1-v10）
bash dag/submit_dagman_workflow.sh X_pb24_v2_fid1_8v1_1o200 1 10 auto
```

### 批量 DAGMan（Optuna）配置与记录位置
1) Optuna 训练配置在哪改：
- 训练流程入口与行为：`workflows/condor_optuna_XGBoost.py`（单空间一次性调参）
- DAGMan 训练调度：`workflows/train_dispatch.py`
- 提交链：`dag/make_dagman_workflow.py`、`dag/submit_dagman_workflow.sh`
- 训练输入数据与筛选：`configs/samples.py`
- 训练变量：`utils/varsets.py`

2) Optuna 超参数空间在哪改：
- `configs/optuna_spaces.py`
  - `OPTUNA_SPACES[dataset][channel][version]`
  - `OPTUNA_TRAINING_OPTIONS[dataset][channel][version]`（如 `early_stopping_rounds`）
- `configs/direct_xgb_settings.py`
  - `DIRECT_XGB_PARAMS[dataset][channel]`

3) 每轮 trial 超参数和目标值（如 AUC）在哪看：
- 当前主线单空间 Optuna默认不落盘“全部 trial 明细”。
- 已落盘内容：
  - `output/models/<train_tag>/optuna_top20_ranges.json`：
    - 包含 top20 trial 的 `trial_number`、`objective_value`（validation AUC）和 `params`
    - 包含初始搜索空间与 top20 收缩区间对比
  - `output/training/<train_tag>/run_metadata.json`：
    - 记录 best params、best objective、搜索空间、n_trials 等元信息

### 不用 Optuna 的 XGBoost 主线（单模型 DAG）
命名约定：`train_tag` 必须严格满足 `{channel}_{dataset}_v{n}_fid{n}_{varset}_xgb_v{n}`，TRAIN 节点自动走 `workflows/xgboost_train_direct.py`。
```bash
bash dag/submit_single_workflow.sh X_pp24_v1_fid1_18v1_xgb_v1 1
bash dag/submit_single_workflow.sh Bu_pp24_v1_fid1_12v1_xgb_v1 1
bash dag/submit_single_workflow.sh Bs_pp24_v1_fid1_17v1_xgb_v1 1
bash dag/submit_single_workflow.sh Bd_pp24_v1_fid1_17v1_xgb_v1 1
bash dag/submit_single_workflow.sh Bs_pp24_v1_fid1_6v1_xgb_v1 0
```
执行链：`TRAIN(Direct XGBoost) -> APPLY -> DRAW`。
（Condor 层通过 `wrappers/run_train_job.sh`、`wrappers/run_apply_job.sh`、`wrappers/run_draw_job.sh` 调用）

## Single DAG 工作内容（配置来源）
1) 训练输入、训练前筛选、SHAP筛选
- 训练输入数据（signal/background 的 ROOT 路径与 TTree）定义在：
  - [configs/samples.py](configs/samples.py) 的 `SAMPLES[sample]["channels"][channel]["datasets"][year]["train"]`
- 训练前筛选条件定义在：
  - 同文件的 `selection_profiles`（`signal_selection` / `background_selection` 表达式）
  - 训练脚本读取：`workflows/xgboost_train_direct.py`（single DAG 直训）/ `workflows/condor_optuna_XGBoost.py`（optuna）
- SHAP筛选条件：
  - SHAP 复用训练同一套输入与同一组表达式（`signal_selection` / `background_selection`）
  - 脚本：`workflows/shap_importance.py`

2) apply输入、apply筛选、draw筛选
- apply 输入数据（MC/data 的 ROOT 路径与 TTree）定义在：
  - [configs/samples.py](configs/samples.py) 的 `datasets[year]["apply"]`
- apply 阶段在 summary 中记录训练筛选表达式（`signal_selection` / `background_selection`），来源：
  - `resolve_training_config(...)`（`configs/samples.py`）
  - 执行脚本：`workflows/batch_apply_scores.py`
- draw 阶段筛选（fiducial region）定义在：
  - `fiducial_profiles`（表达式格式，`configs/samples.py`）
  - 执行脚本：`workflows/batch_draw_scores.py`

3) 训练变量
- 训练变量组合定义在：
  - [utils/varsets.py](utils/varsets.py) 的 `VARSETS[sample][channel][varset]`
- single DAG 会从 `train_tag` 解析 `sample/channel/selection_profile/fid_profile/varset`，据此读取配置。

## 如何输入 Optuna 超参数空间与训练配置
- Optuna 空间：
  - 文件：[configs/optuna_spaces.py](configs/optuna_spaces.py)
  - 字段：`OPTUNA_SPACES[dataset][channel][version]`（例如 `OPTUNA_SPACES["pb23"]["X"]["v1"]`）
  - 训练附加配置：`OPTUNA_TRAINING_OPTIONS[dataset][channel][version]`
- 直接 XGBoost（非 Optuna）参数：
  - 文件：[configs/direct_xgb_settings.py](configs/direct_xgb_settings.py)
  - 字段：`DIRECT_XGB_PARAMS[dataset][channel]`
- 样本路径、TTree、训练筛选、fid cut：
  - 文件：[configs/samples.py](configs/samples.py)
- `samples.py` 核心结构：
  - `SAMPLES[sample]["channels"][channel]["datasets"][year]["train/apply/draw"]`
  - `selection_profiles`：训练筛选表达式（`signal_selection` / `background_selection`）
  - `fiducial_profiles`：apply/draw fid 表达式
- draw 输入约定：
  - draw 读取的是 apply 产物 `output/selected/<train_tag>/DATA_with_score.root`
  - 读取树名来自 `datasets[year].draw.data.tree`（按 channel 独立）
- `selection_profiles` 控制训练筛选表达式
- `fiducial_profiles` 控制 apply/draw 筛选表达式（由 `fid_profile` 选择）
  - 以上均按 `sample + channel` 独立配置
- 变量组合：
  - 文件：[utils/varsets.py](utils/varsets.py)

## 如何修改关键配置
- 修改 varset：编辑 [utils/varsets.py](utils/varsets.py)
  - 结构：`VARSETS[sample][channel][varset]`
  - 示例：`VARSETS["pbpb"]["X"]["4v2"]`、`VARSETS["pbpb"]["Bu"]["4v2"]`、`VARSETS["pp"]["X"]["4v2"]`
- 修改输入 ROOT/TTree：编辑 [configs/samples.py](configs/samples.py) 的 `datasets`（`train/apply/draw`）
- 修改训练筛选条件：编辑 `configs/samples.py` 的 `selection_profiles`
- 修改 apply/draw fiducial region：编辑 `configs/samples.py` 的 `fiducial_profiles`
- 修改 Optuna 搜索空间：编辑 [configs/optuna_spaces.py](configs/optuna_spaces.py) 的 `OPTUNA_SPACES`
- 修改 Optuna early stop：编辑 `configs/optuna_spaces.py` 的 `OPTUNA_TRAINING_OPTIONS`
- 修改无 Optuna 训练参数：编辑 `configs/direct_xgb_settings.py` 的 `DIRECT_XGB_PARAMS`

## 命名强校验
- single DAG 仅接受：`{channel}_{dataset}_v{n}_fid{n}_{varset}_xgb_v{n}`
- 任一字段缺失或格式不符会直接报错，不做默认值补全。
- `selection_profile` 和 `fid_profile` 按 dataset 独立定义并解析（`pp24` / `pb23` / `pb24` 不混用）。

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
bash dag/submit_single_workflow.sh X_pp24_v2_fid2_6v4_xgb_v1 1
```

### 2) 对已完成训练单独运行 SHAP
不重跑 train/apply/draw，直接针对已有模型做 SHAP：
```bash
.venv/bin/python -m workflows.shap_importance <train_tag> [max_events]
```
示例：
```bash
.venv/bin/python -m workflows.shap_importance X_pp24_v1_fid1_18v1_xgb_v1
.venv/bin/python -m workflows.shap_importance Bu_pb24_v1_fid1_5v2_xgb_v1 30000
```

### SHAP 输入筛选说明
- SHAP 输入样本与训练保持一致：
  - signal/background 使用 `configs/samples.py` 对应 `train` 输入
  - signal 应用 `signal_selection` 表达式
  - background 应用 `background_selection` 表达式
- 即 SHAP 现在严格复用训练筛选逻辑，不再使用宽松/不一致筛选。

## Score 图与 Cut Scan
- direct XGBoost 训练（`workflows/xgboost_train_direct.py`）现在会输出真实的 `xgb_score.pdf`（分数分布图），不再把 JSON 写入 PDF 文件。
- 对应 ROC/AUC 指标输出到 `output/training/<train_tag>/test_roc.json`。
- draw 的 score cut 扫描点（`workflows/batch_draw_scores.py`）固定为：
  - `0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.90, 0.95`

## 命名规范
- single DAG（direct XGB）：`{channel}_{dataset}_v{n}_fid{n}_{varset}_xgb_v{n}`
- DAGMan Optuna group（显式模式）：`{channel}_{dataset}_v{n}_fid{n}_{varset}_{n}o{N}_v{k}`
- DAGMan Optuna group（旧批量模式）：`{channel}_{dataset}_v{n}_fid{n}_{varset}_{n}o{N}`

## 如何运行 Optuna DAG 训练
显式模式命令：
```bash
bash dag/submit_dagman_workflow.sh <group_tag_with_{n}o{N}_v{k}> [fid_profile]
```
旧批量模式命令：
```bash
bash dag/submit_dagman_workflow.sh <group_tag_with_{n}o{N}> <version_start> <version_end> [fid_profile]
```
示例：
```bash
bash dag/submit_dagman_workflow.sh X_pb23_v1_fid1_18v1_1o200_v1 auto
bash dag/submit_dagman_workflow.sh X_pb23_v1_fid1_18v1_1o200 1 10 auto
```
- `n`：Optuna objective index（当前仅支持 `1`，对应 validation AUC）
- `N`：Optuna trial 次数
- `k`：Optuna 空间版本（仅显式模式需要；映射到 `configs/optuna_spaces.py` 中的 `v{k}`）

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

2) 归档当前输出到 `output/backup_outdate/<timestamp>/`：
- 归档前会删除每个 tag 的 `output/selected/<tag>/DATA_with_score.root`（可用 `--keep-selected-data-root` 关闭）。
- 归档过程中会自动调用 `scripts/cleanup/clear_dag_locks.sh` 清理 single DAG 重提冲突文件（含 `.dag.condor.sub/.dag.lib.out/.dag.lib.err/.dag.dagman.log` 等）。
- 支持按 workflow 类型过滤：`--workflow-type all|single|optuna`（默认 `all`）
  - `single`：`{channel}_{dataset}_v{n}_fid{n}_{varset}_xgb_v{n}`
  - `optuna`：`{channel}_{dataset}_v{n}_fid{n}_{varset}_{n}o{N}_v{k}`
```bash
python3 scripts/cleanup/archive_output_outdated.py
python3 scripts/cleanup/archive_output_outdated.py --dry-run
python3 scripts/cleanup/archive_output_outdated.py --dry-run --workflow-type single
python3 scripts/cleanup/archive_output_outdated.py --workflow-type optuna
```

3) single DAG 重提前仅清理锁文件：
```bash
bash scripts/cleanup/clear_dag_locks.sh <train_tag>
```

## Punzi Optimization（B）
- 新增脚本：`workflows/optimization/run_optimization_from_tag.py`
- 功能：
  - 根据 `train_tag` 生成/覆盖 `../Analysis_CODES/selectionER/optimalCUT.conf` 中同名 profile
  - profile 名直接使用训练标签
  - `preCut` 使用该标签对应 `fid_profile`，并自动转换为 ROOT 表达式（`and/or/not -> &&/||/!`）
  - `dataPath/mcPath` 固定写为 `../../XGBoost/output/selected/<train_tag>/{DATA,MC}_with_score.root`
  - `sidebandLow/high` 从 `background_selection` 中的 `Bmass` 窗口提取（单边窗会自动生成零宽 low sideband）
  - `refScoreCut` 默认 `0.6`
  - `fsRegion`：
    - `Bu/Bd`: `(Bmass > 5.2 && Bmass < 5.36)`
    - `Bs`: `(Bmass > 5.3 && Bmass < 5.46)`
  - `signalWidth` 取 `fsRegion` 宽度；`sidebandWidth` 取 sideband 总宽度
  - 默认 `punziA=2.0`、`punziB=5.0`

示例：
```bash
# 仅更新/覆盖 conf
.venv/bin/python workflows/optimization/run_optimization_from_tag.py Bs_pp24_v1_fid1_17v1_xgb_v1

# 更新 conf 并运行 punzi + fom
.venv/bin/python workflows/optimization/run_optimization_from_tag.py Bs_pp24_v1_fid1_17v1_xgb_v1 --run

# 仅运行 punzi
.venv/bin/python workflows/optimization/run_optimization_from_tag.py Bs_pp24_v1_fid1_17v1_xgb_v1 --run_punzi

# 仅运行 fom
.venv/bin/python workflows/optimization/run_optimization_from_tag.py Bs_pp24_v1_fid1_17v1_xgb_v1 --run_fom
```
