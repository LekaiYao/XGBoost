# AGENTS Guide

## 核心任务
- 维护并扩展 XGBoost 训练/应用/画图工作流
- 支持 `Condor + DAGMan` 与 `单模型 DAG` 两种模式
- 保证输出目录结构稳定，便于批量比较和人工检查
- 保持 EOS 代码与 AFS 提交的一致性

## 工作要求
- 遵从最小化修改原则
- 不修改无关的代码
- 修改之前，说明要修改哪些文件
- 对于非平凡的修改，做修改前给出简单的计划说明
- 当存在不确定的问题时，检查已有的代码，而不是直接做修改
- 每一次功能更新后，及时提醒我更新到github

## 目录职责
- `utils/`
  - `varsets.py`：变量组合注册表与 tag 解析
  - `paths.py`：统一路径规则（local/condor/legacy 兼容）
  - `run_metadata.py`：训练元数据落盘
- `condor_optuna_XGBoost.py`
  - 主线 Condor Optuna 训练（`v1-v100`）
- `staged_optuna_pipeline.py`
  - 非主线的替代训练方案：分阶段串行调参（`2v*` 和部分 `v*` stage 组）
- `make_dagman_workflow.py` + `submit_dagman_workflow.sh`
  - 生成/提交 `训练并行 + FINAL 批处理` DAG
- `batch_apply_scores.py`
  - 同组多模型一次性 apply 到同一个 DATA/MC ROOT
- `batch_draw_scores.py`
  - 从 grouped DATA ROOT 批量出图并写 sigma 摘要
- `run_batch_compare.sh` + `submit_batch_compare.sub`
  - 批处理 apply/draw 的 Condor 入口
- `workflow_archive/legacy_non_dag/`
  - 单模型脚本 `XGBoost.py`, `apply.py`, `draw.py`
- `make_single_legacy_dag.py` + `run_single_legacy_step.sh` + `submit_single_legacy_dag.sh`
  - 单模型 DAG 入口（`TRAIN->APPLY->DRAW->SHAP`）

## 关键入口文件
- 主线训练入口：`submit_dagman_workflow.sh`
- 主线训练执行器：`run_staged.sh`
- 批处理入口：`run_batch_compare.sh`
- 单模型入口：`submit_single_legacy_dag.sh`

## 修改时必须遵守
- 先改 `utils/varsets.py` 再新增新变量组合；禁止在多个脚本重复硬编码变量列表
- 路径规则改动必须经 `utils/paths.py`，避免脚本间目录不一致
- 训练输出命名依赖 `train_tag`，不要改已有 tag 解析规则（`_<varset>_`）
- 批量流程要求“部分失败可跳过”：`batch_apply_scores.py` 的 skip 机制不可删除
- EOS 代码 + AFS 提交分离必须保持
- `submit_dagman_workflow.sh` 依赖 `make_dagman_workflow.py` 最后一行输出 DAG 路径，勿回退这个行为

## 测试/验证命令
语法检查：
```bash
.venv/bin/python -m py_compile condor_optuna_XGBoost.py staged_optuna_pipeline.py batch_apply_scores.py batch_draw_scores.py workflow_archive/legacy_non_dag/XGBoost.py
```

单模型 DAG 验证：
```bash
bash submit_single_legacy_dag.sh pp24_5v_xgb_v1
condor_q -dag <cluster_id>
```

主线 DAG 验证（小批）：
```bash
bash submit_dagman_workflow.sh pb23v6_5v_4o200 1 10 v 200
```

## 不要轻易改动
- `utils/paths.py` 中 `train_batch_tag()` 与 `condor_*_dir()`：影响所有输出归档
- `run_staged.sh` 中 `stage_group` 分流逻辑：`v*` 走 `condor_optuna_XGBoost.py`，其余走 staged pipeline
- `run_single_legacy_step.sh` 中 `PYTHONPATH` 导出：修复过 DAG 环境导入失败
- `batch_draw_scores.py` 的 `score_cuts` 与 fid profile 分支（`pb23v6_` 特判）
- 主线 DATA 输入目前固定为 PbPb24（训练与 batch apply 均已切换），不要误改回 PbPb23_DATA0
- 不随意在项目目录下保留一次性的txt,json等文本文件，如需要使用，用完后要删除

## 已知历史坑点
- 在 EOS 目录直接工具执行可能遇到沙箱写 `.codex` 失败；通常需要提权命令
- `condor_submit_dag` 复提同名 DAG 会因文件已存在失败；需用 `-f`
- 单模型脚本从 `workflow_archive` 直接执行时，若缺 `PYTHONPATH` 会报 `No module named utils`
- 主线与单模型样本路径不同（PbPb vs ppRef），不可混用解释
- `run_metadata.py` 默认 `train/val/test` 字段仍是 `0.8/0.1/0.1`，但单模型线实际是 `0.8/0.2`
- `make_dagman_workflow.py` 增加了 `print` 后，`submit_dagman_workflow.sh` 若不取最后一行会提交失败（`cp cannot stat`）
- EOS 偶发 I/O 抖动会导致 draw 读 ROOT 报 `received 0 bytes from FSSpecSource`，通常重跑 draw 可恢复

## 当前 varset 名单
以 `utils/varsets.py` 为准
