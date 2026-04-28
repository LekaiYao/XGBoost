# XGBoost Analysis Workflow

## 项目目标
本项目用于 `PbPb/ppRef` 质量谱分析中的 XGBoost 训练、打分、cut-scan 画图和 SHAP 解释。
当前目标是：稳定主线 PbPb 批量流程，并支持 pp 单模型流程快速迭代。

## 环境与路径
- EOS 代码目录：`/eos/home-l/leyao/pbpb_work/X_analysis/XGBoost`
- AFS 提交目录：`/afs/cern.ch/user/l/leyao/private/pbpb_work/X_analysis/XGBoost`
- Condor 从 AFS 提交，运行脚本在 EOS。

## 当前工作流
### 1) 主线 PbPb DAG（训练 + 批量 apply/draw）
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
### 主线 PbPb 输入（当前）
- 支持按年份选择 2023/2024 输入（训练与 batch apply/draw 共用该逻辑）。
- 主要脚本：
  - `condor_optuna_XGBoost.py`
  - `staged_optuna_pipeline.py`
  - `batch_apply_scores.py`

### pp24v2（单模型线）训练输入与筛选（当前）
- 训练脚本：`workflow_archive/legacy_non_dag/XGBoost.py`（`optuna_XGBoost.py` 同步）
- 训练输入：
  - signal(MC)：`/eos/user/h/hmarques/RUN3_Data_MC_sharing/X3872/ppRef24/flat_ntmix_ppRef_MC_X3872.root:ntmix_X3872`
  - background(DATA)：`/eos/user/h/hmarques/RUN3_Data_MC_sharing/X3872/ppRef24/flat_ntmix_ppRef_DATA.root:ntmix`
- 训练筛选：
  - signal：`Bchi2Prob > 0.02 && Btrk1dR < 0.5`
  - background：`Bchi2Prob > 0.02 && Btrk1dR < 0.5 && ((Bmass > 3.95 && Bmass < 4.00) || (Bmass > 3.75 && Bmass < 3.80))`

### pp24v2（单模型线）apply/draw（当前）
- apply 脚本：`workflow_archive/legacy_non_dag/apply.py`
- draw 脚本：`workflow_archive/legacy_non_dag/draw.py`
- apply 输入：
  - DATA：`.../flat_ntmix_ppRef_DATA.root:ntmix`
  - MC：
    - `flat_ntmix_ppRef_MC_PSI2S_nonPrompt.root:ntmix_PSI2S`
    - `flat_ntmix_ppRef_MC_PSI2S.root:ntmix_PSI2S`
    - `flat_ntmix_ppRef_MC_X3872_nonPrompt.root:ntmix_X3872`
    - `flat_ntmix_ppRef_MC_X3872.root:ntmix_X3872`
- apply 输出（`selected_events/<train_tag>/`）：
  - `DATA_wScore.root`
  - `MC_PSI2S_nonPrompt_wScore.root`
  - `MC_PSI2S_wScore.root`
  - `MC_X3872_nonPrompt_wScore.root`
  - `MC_X3872_wScore.root`
- draw 在 `pp*` tag 下默认读取 `DATA_wScore.root`。

### 输出目录
- 模型与训练图：`xgb_output/models/...`, `xgb_output/training/...`
- SHAP：`xgb_output/shap/...`
- 打分与 cut-scan：`selected_events/...`

## 当前 `<varset>`（`utils/varsets.py`）
- `4v`, `4v2`, `5v`, `5v2`, `5v3`
- `6v`, `6v2`, `6v3`, `6v4`
- `7v2`, `7v3`, `7v4`
- `8v`, `8v2`, `8v3`, `8v4`
- `9v`

## 常用命令
语法检查：
```bash
.venv/bin/python -m py_compile condor_optuna_XGBoost.py staged_optuna_pipeline.py batch_apply_scores.py batch_draw_scores.py workflow_archive/legacy_non_dag/XGBoost.py workflow_archive/legacy_non_dag/apply.py workflow_archive/legacy_non_dag/draw.py
```

主线 DAG 提交（例）：
```bash
bash submit_dagman_workflow.sh pb24v2_8v_4o200 1 10 v 200
```

单模型 DAG 提交（例）：
```bash
bash submit_single_legacy_dag.sh pp24v2_6v4_xgb_v1
```

## 近期更新
- 主线 PbPb 流程统一为按年份（2023/2024）成组切换输入，并移除历史混配 fallback。
- 修复主线批处理 DAG 参数链，确保 apply/draw 的 dataset year 与 fid profile 透传正确。
- 单模型 `xgb_score` 图：train 点带统计误差和 x 方向 bin 宽误差。
- 新增 `6v4` 变量组合。
- pp24v2 apply/draw 输入输出命名已更新（`DATA_wScore.root` + 四个 MC wScore ROOT）。

## 清理脚本
清理历史 selected events（可配置按天数移动目录并删除大 ROOT）：
```bash
python3 cleanup_selected_events.py --days 5 --root-threshold-mb 500
```
