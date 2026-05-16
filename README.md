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

### 单模型 DAG
```bash
bash dag/submit_single_workflow.sh <train_tag> [with_shap]
```
执行链：`TRAIN -> APPLY -> DRAW -> (optional SHAP)`，`.sub` 直接调用 `workflows/*.py`。

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
