# XGBoost Analysis Workflow

## 项目目标
本项目用于 `PbPb/ppRef` 分析中的 XGBoost 训练、打分、出图与 SHAP。
当前架构统一为可迁移工作流：`pp` 与 `pbpb` 共用脚本框架、独立配置与变量集。

## 目录结构
- `workflows/`：核心训练/应用/出图/SHAP 脚本
- `pipelines/`：Condor 节点执行器
- `dag/`：DAG 生成与提交脚本、Condor 模板
- `utils/`：varset / paths / metadata 公共模块
- `configs/`：样本与超参数空间配置
- `output/`：统一输出根目录

## 工作流
### 1) 主线 PbPb 批量 DAG
- 提交：`bash dag/submit_dagman_workflow.sh <group_tag> <version_start> <version_end> <version_token> [optuna_n_trials] [dataset_year] [selection_profile] [fid_profile]`
- 执行链：
  - `dag/make_dagman_workflow.py`
  - `pipelines/run_staged.sh`
  - `workflows/condor_optuna_XGBoost.py` 或 `workflows/staged_optuna_pipeline.py`
  - `pipelines/run_batch_compare.sh`
  - `workflows/batch_apply_scores.py` + `workflows/batch_draw_scores.py`

### 2) 单模型 DAG（pp/pbpb 通用）
- 提交：`bash dag/submit_single_workflow.sh <train_tag> [with_shap]`
- 执行链：
  - `dag/make_single_workflow.py`
  - `pipelines/run_train_job.sh`
  - `pipelines/run_apply_job.sh`
  - `pipelines/run_draw_job.sh`
  - `pipelines/run_shap_job.sh`（仅当 `with_shap=1`）
  - 其中 pp 当前调用 `workflow_archive/legacy_non_dag/*`，pbpb 调用 `workflows/*`

## 命名规范
- 普通训练：`<sample>_<varset>_v<version>`
- Optuna 训练：`<sample>_<varset>_o<optunaTrials>_v<version>`

## varset 规则
- `utils/varsets.py` 采用 sample-aware 定义：
  - `VARSETS["pbpb"]`
  - `VARSETS["pp"]`
- 同名 varset 在不同 sample 下可对应不同变量列表。

## 输出目录（新）
- `output/models/...`
- `output/training/...`
- `output/shap/...`
- `output/selected/...`

## 常用命令
语法检查：
```bash
.venv/bin/python -m py_compile workflows/condor_optuna_XGBoost.py workflows/staged_optuna_pipeline.py workflows/batch_apply_scores.py workflows/batch_draw_scores.py workflows/shap_importance.py dag/make_dagman_workflow.py dag/make_single_workflow.py utils/varsets.py utils/paths.py
```

主线示例：
```bash
bash dag/submit_dagman_workflow.sh pb24v2_8v_4o200 1 10 v 20 2024 pb24v2 fid3
```

单模型示例：
```bash
bash dag/submit_single_workflow.sh pp24v2_6v4_xgb_v1 0
bash dag/submit_single_workflow.sh pp24v2_6v4_xgb_v1 1
```

## 清理脚本
```bash
python3 cleanup_selected_events.py --days 5 --root-threshold-mb 500
```
