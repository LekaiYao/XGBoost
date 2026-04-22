# Session Bootstrap Summary

## 1. 项目目标（代码可见）
- 维护 XGBoost 分析流水线：训练、打分、出图、SHAP。
- 同时支持两条执行链：
  - 主线（PbPb）：`Condor + DAGMan` 批量训练 + FINAL 批处理。
  - 单模型（ppRef）：`TRAIN -> APPLY -> DRAW -> SHAP`。
- 统一输出路径与命名，支持历史兼容与批量比较（由 `utils/paths.py` 实现）。

## 2. 关键目录与职责
- `utils/varsets.py`
  - 维护 `VARSET_COLUMNS` 与 `infer_varset_from_tag()`。
- `utils/paths.py`
  - 统一 local/condor/legacy 路径与 fallback 解析（`resolve_*`）。
- `utils/run_metadata.py`
  - 统一写 `run_metadata.json`。
- 主线入口与执行
  - `submit_dagman_workflow.sh` -> `make_dagman_workflow.py` -> `run_staged.sh` -> (`condor_optuna_XGBoost.py` 或 `staged_optuna_pipeline.py`) -> `run_batch_compare.sh` -> `batch_apply_scores.py` + `batch_draw_scores.py`
- 单模型入口与执行
  - `submit_single_legacy_dag.sh` -> `make_single_legacy_dag.py` -> `run_single_legacy_step.sh` -> `workflow_archive/legacy_non_dag/{XGBoost.py,apply.py,draw.py}` + `shap_importance.py`

## 3. 主入口与执行流程（实际脚本）
- 主线 DAG：
  1. `submit_dagman_workflow.sh` 生成 DAG 并复制到 AFS 后 `condor_submit_dag`。
  2. 训练节点执行 `run_staged.sh`：
     - `stage_group` 匹配 `^v[0-9]+$`：调用 `condor_optuna_XGBoost.py <train_tag> <stage_group>`
     - 否则：调用 `staged_optuna_pipeline.py <train_tag> --stage-group ... [--resume]`
  3. FINAL 节点执行 `run_batch_compare.sh`，运行 grouped apply/draw。
- 单模型 DAG：
  1. `make_single_legacy_dag.py` 生成 `TRAIN->APPLY->DRAW->SHAP`。
  2. `run_single_legacy_step.sh` 根据 `step` 分发到 train/apply/draw/shap 脚本。
  3. 脚本内显式设置 `PYTHONPATH=${repo_dir}:${PYTHONPATH:-}`。

## 4. 核心模块与依赖关系
- `condor_optuna_XGBoost.py`
  - 依赖 `utils.varsets` 推断 varset，`utils.paths` 写 condor 输出，`utils.run_metadata` 落 metadata。
  - 输入样本：`PbPb23 MC` + `PbPb24 DATA`（脚本内常量）。
- `staged_optuna_pipeline.py`
  - 同样依赖 `utils.varsets/utils.paths/utils.run_metadata`。
  - `STAGE_CONFIGS` 定义 `2v*` 和部分 `v*` 的 staged 搜索；支持 `--resume` 状态文件。
  - 输入样本：`PbPb23 MC` + `PbPb24 DATA`（脚本内常量）。
- `batch_apply_scores.py`
  - 通过 `resolve_model_path/resolve_scaler_path/resolve_model_config_path` 加载模型。
  - 要求同组模型 `input_columns/trans_columns` 一致。
  - 支持坏模型跳过并记录 `skipped_models`。
- `batch_draw_scores.py`
  - 读取 grouped DATA 根文件中的 `xgb_score_<train_tag>` 分支。
  - 产出每个 train_tag 的 cut 扫描图和 `X3872` sigma 摘要 markdown。
- 单模型脚本
  - `workflow_archive/legacy_non_dag/XGBoost.py`：ppRef 训练并输出模型/图/ROC/metadata。
  - `apply.py`：输出 `xgb_score` 到 `selected_events/<train_tag>/`。
  - `draw.py`：读取 `xgb_score` 按 cut 出图。
  - `shap_importance.py`：加载模型后计算 SHAP 并输出 json/pdf。

## 5. 当前 varset（来自 `utils/varsets.py`）
- `4v`, `4v2`, `5v`, `6v`, `7v`, `7v2`, `8v2`, `8v3`, `9v`

## 6. 已完成内容（代码已体现）
- 单模型 DAG 已包含 `SHAP` 节点（`make_single_legacy_dag.py`）。
- `run_single_legacy_step.sh` 已设置 `PYTHONPATH`。
- 主线提交脚本已通过 `tail -n 1` 取 DAG 路径（`submit_dagman_workflow.sh`）。
- 主线训练与 grouped apply 默认 background DATA 为 PbPb24（`condor_optuna_XGBoost.py`、`staged_optuna_pipeline.py`、`batch_apply_scores.py`）。
- 主线 grouped draw 已写 `X3872_sigma_summary_*.md`（`batch_draw_scores.py`）。

## 7. 未完成内容（文档与代码一致）
- `run_metadata` split 字段固定写 `train/val/test = 0.8/0.1/0.1`，与单模型 `train_test_split(test_size=0.2)` 不一致。
- `submit_single_legacy_dag.sh` 目前未提供 `-f/--force` 选项。
- `batch_draw_scores.py` 的 `score_cuts` 仍为脚本硬编码。
- 缺少最小自动化回归测试（仓库中未见对应测试集）。

## 8. 下一步最合理的 3 个任务
1. 让 `save_run_metadata()` 接收可配置 split，并在主线/单模型分别传入真实拆分。
2. 给 `submit_single_legacy_dag.sh` 增加 `-f/--force`，透传到 `condor_submit_dag -f`。
3. 将 `batch_draw_scores.py` 的 `score_cuts`（及相关 profile 参数）外置到配置文件。

## 9. 最容易改坏的地方
- `utils/paths.py` 的 `train_batch_tag()` 与 `condor_*_dir()`：影响全链路输出定位与兼容。
- `run_staged.sh` 的分流条件：决定走 `condor_optuna_XGBoost.py` 还是 `staged_optuna_pipeline.py`。
- `utils/varsets.py` 的 tag 推断规则：影响训练输入列一致性。
- `submit_dagman_workflow.sh` 对 `make_dagman_workflow.py` 输出取最后一行的约定。
- `batch_apply_scores.py` 的 skip 机制与列一致性校验。
- `run_single_legacy_step.sh` 的 `PYTHONPATH` 导出。
