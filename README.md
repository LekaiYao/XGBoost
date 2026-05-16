# XGBoost Analysis Workflow

## 项目目标
统一 PbPb 与 pp 分析训练/应用/画图工作流，避免在脚本内硬编码路径与 cut。

## 目录结构
- `configs/`：样本、路径、TTree、fid/train cuts、搜索空间
- `workflows/`：训练/应用/画图/SHAP 与调度脚本
- `dag/`：DAG 生成与提交
- `utils/`：varsets、路径、metadata
- `output/`：统一输出目录

## 主线工作流
### 主线 DAGMan（批量）
```bash
bash dag/submit_dagman_workflow.sh <group_tag> <version_start> <version_end> <version_token> [optuna_n_trials] [dataset_year] [selection_profile] [fid_profile]
```
执行链：`make_dagman_workflow.py -> train_dispatch.py -> (condor_optuna_XGBoost.py / staged_optuna_pipeline.py) -> group_apply_draw.py -> batch_apply_scores.py + batch_draw_scores.py`

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
