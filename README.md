# XGBoost Analysis Workflow

## 项目目标
本项目用于 `PbPb/ppRef` 质量谱分析中的 XGBoost 训练、打分、cut-scan 画图和 SHAP 解释。  
当前主线目标是：在稳定训练的前提下提升本底压制能力并便于批量比较不同超参数空间。

## 环境与路径
- EOS 代码目录：`/eos/home-l/leyao/pbpb_work/X_analysis/XGBoost`
- AFS 提交目录：`/afs/cern.ch/user/l/leyao/private/pbpb_work/X_analysis/XGBoost`
- Condor 必须从 AFS 提交，实际运行脚本在 EOS

Python 依赖见 `requirements.txt`（`uproot`, `xgboost`, `optuna`, `shap` 等）。

## 当前工作流
### 1) 主线 DAG（训练 + 批量 apply/draw）
- 入口：`submit_dagman_workflow.sh`
- DAG 生成：`make_dagman_workflow.py`
- 训练节点：`run_staged.sh`
- 批处理节点：`run_batch_compare.sh`
- 批处理脚本：`batch_apply_scores.py` + `batch_draw_scores.py`

### 2) 单模型 DAG（train -> apply -> draw -> shap）
- 入口：`submit_single_legacy_dag.sh`
- DAG 生成：`make_single_legacy_dag.py`
- 节点执行器：`run_single_legacy_step.sh`
- 单模型脚本：`workflow_archive/legacy_non_dag/XGBoost.py`, `apply.py`, `draw.py`
- SHAP 脚本：`shap_importance.py`

## 输入与输出
### 主线训练输入（当前）
- signal：`/eos/user/h/hmarques/RUN3_Data_MC_sharing/X3872/PbPb23/flat_ntmix_PbPb23_MC.root:ntmix`
- background：`/eos/user/h/hmarques/RUN3_Data_MC_sharing/X3872/PbPb24/flat_ntmix_PbPb24_DATA.root:ntmix`

对应代码：
- `condor_optuna_XGBoost.py` 的 `BKG_PATH`
- `staged_optuna_pipeline.py` 的 `BKG_PATH`

### 主线 apply 输入（当前默认）
- `batch_apply_scores.py` 的 `DATA_INPUT_DEFAULT` 也已切换为
  `/eos/user/h/hmarques/RUN3_Data_MC_sharing/X3872/PbPb24/flat_ntmix_PbPb24_DATA.root:ntmix`

### 输出目录
- 模型与训练图：`xgb_output/models/...`, `xgb_output/training/...`
- SHAP：`xgb_output/shap/...`
- 打分与 cut-scan：`selected_events/...`

## 当前 `<varset>`（`utils/varsets.py`）
当前注册：
- `4v`
- `4v2`
- `5v`
- `6v`
- `7v`
- `7v2`
- `8v2`
- `8v3`
- `9v`

说明：
- 以下含 `Bnorm_svpvDistance_2D` 的组合已删除：`10v`, `8v`, `8v4`, `9v2`

## 常用命令
语法检查：
```bash
.venv/bin/python -m py_compile condor_optuna_XGBoost.py staged_optuna_pipeline.py batch_apply_scores.py batch_draw_scores.py workflow_archive/legacy_non_dag/XGBoost.py
```

主线 DAG 提交（例）：
```bash
bash submit_dagman_workflow.sh pb24v1_4v_4o200 1 10 v 200
```

单模型 DAG 提交（例）：
```bash
bash submit_single_legacy_dag.sh pp24_7v2_xgb_v1
```

## 近期重要更新
- 主线 background DATA 全部切到 PbPb24
- 主线批量 apply 默认 DATA 输入切到 PbPb24
- 修复 `submit_dagman_workflow.sh` 路径捕获问题（`make_dagman_workflow.py` 多行输出导致 `cp` 失败），现在取最后一行 DAG 路径
- 单模型 `xgb_score` 图更新：
  - 50 个 bin
  - test 为半透明填充
  - train 为误差棒点（含 x 方向 bin 宽误差），marker size 进一步减小
