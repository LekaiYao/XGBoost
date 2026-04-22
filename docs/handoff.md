# Handoff (for new Codex session)

## 0. 当前状态（2026-04-20）

### 当前目标
- 监控并验收本轮主线 `pb24v2` 全量重跑结果：`8v/8v4` 的 `v1-100` 与 `3v1-100`（共 40 个 DAG 批次，`fid3` 画图）。

### 已完成内容
- 已清理上一轮失败状态与产物：
  - EOS 输出（`xgb_output/models`、`xgb_output/training`、`selected_events`）中 `pb24v2_8v*` / `pb24v2_8v4*` 失败批次已删除。
  - AFS 提交侧旧 DAG/rescue/log（`dags/wf_pb24v2_8v*_4o200_*`、`logs/job_pb24v2_8v*_4o200_*`）已清空。
- 已完成 40 组全量重新提交（每组 10 个版本）：
  - `pb24v2_8v_4o200_v1-v100`
  - `pb24v2_8v_4o200_3v1-3v100`
  - `pb24v2_8v4_4o200_v1-v100`
  - `pb24v2_8v4_4o200_3v1-3v100`
  - 参数：`dataset_year=2024`、`selection_profile=pb24v2`、`fid_profile=fid3`、`optuna_n_trials=200`
  - 提交结果：`ok=40, fail=0`；`condor_q -dag` 可见 40 个 DAG。
- 提交后快速排查中，暂未出现上一轮两类系统性错误：
  - `Unknown stage group`
  - `FileNotFoundError: .../auto`

### 未完成内容
- 40 组 DAG 仍在运行，尚未全部完成。
- 尚未完成本轮完整验收：
  - 训练节点成功率统计
  - FINAL（apply+draw）完成率统计
  - `selected_events` 下 `fid3` 出图与 sigma summary 完整性核查
- 尚未形成“失败批次补跑清单”。

### 下一步
1. 按批次监控 40 个 DAG（优先看 `3v*` 训练节点和 `POST_BATCH`）。
2. 完成后做三类验收：
   - 训练输出：`xgb_output/models` 与 `xgb_output/training`
   - apply 输出：`selected_events/<group>/...with_score.root`
   - draw 输出：`DATA_fid3_cut*.pdf` 与 `X3872_sigma_summary_fid3.md`
3. 对失败节点补跑并更新失败原因归档。
4. 汇总本轮最终完成率并更新 `docs/worklog.md`。

### 相关文件
- 提交与 DAG：
  - `submit_dagman_workflow.sh`
  - `make_dagman_workflow.py`
- 训练执行链：
  - `run_staged.sh`
  - `submit_staged_single.sub`
  - `condor_optuna_XGBoost.py`
- 批处理执行链：
  - `run_batch_compare.sh`
  - `submit_batch_compare_single.sub`
  - `batch_apply_scores.py`
  - `batch_draw_scores.py`
- 运行侧目录（AFS）：
  - `dags/wf_pb24v2_8v*_4o200_*.dag*`
  - `logs/job_pb24v2_8v*_4o200_*`

## 1. 项目整体架构
本仓库是 XGBoost 分析流水线，分两条执行链：

- 主线（PbPb，大批量）
  - `DAGMan`: 并行训练节点 + FINAL 批处理节点
  - 训练脚本：`condor_optuna_XGBoost.py` 或 `staged_optuna_pipeline.py`
  - 批处理脚本：`batch_apply_scores.py` + `batch_draw_scores.py`
- 单模型线（ppRef，快速迭代）
  - `DAGMan`: `TRAIN -> APPLY -> DRAW -> SHAP`
  - 脚本：`workflow_archive/legacy_non_dag/XGBoost.py` / `apply.py` / `draw.py` + `shap_importance.py`

EOS/AFS 分离：
- 代码运行在 EOS 工作区
- Condor 提交在 AFS 镜像目录

近期修复：
- `submit_dagman_workflow.sh` 已改为只取 `make_dagman_workflow.py` 输出最后一行作为 DAG 路径（避免 `cp cannot stat`）。

## 2. 数据流 / 调用链

### 主线 DAG
1. `submit_dagman_workflow.sh`
2. `make_dagman_workflow.py` 生成 DAG
3. 每个训练节点调用 `run_staged.sh`
4. `run_staged.sh` 分流：
   - `stage_group` 是 `vN`：`condor_optuna_XGBoost.py <train_tag> <vN>`
   - 否则：`staged_optuna_pipeline.py <train_tag> --stage-group <...>`
5. FINAL 节点调用 `submit_batch_compare_single.sub` -> `run_batch_compare.sh`
6. `run_batch_compare.sh` 执行：
   - `batch_apply_scores.py`
   - `batch_draw_scores.py`

### 单模型 DAG
1. `submit_single_legacy_dag.sh`
2. `make_single_legacy_dag.py` 生成 DAG
3. 节点统一通过 `run_single_legacy_step.sh`
4. `step=train|apply|draw|shap` 分别调用：
   - `workflow_archive/legacy_non_dag/XGBoost.py`
   - `workflow_archive/legacy_non_dag/apply.py`
   - `workflow_archive/legacy_non_dag/draw.py`
   - `shap_importance.py`

## 3. 关键脚本关系（输入/输出/依赖）

### `condor_optuna_XGBoost.py`
- 输入：
  - CLI: `<train_tag> <search_space_tag>`
  - Env: `OPTUNA_N_TRIALS`
  - ROOT: `PbPb23 MC` + `PbPb24 DATA`
  - varset 来自 `utils/varsets.py`
- 输出：
  - `xgb_output/models/<batch>/<train_tag>/...`
  - `xgb_output/training/<batch>/<train_tag>/...`
  - `run_metadata.json`
- 依赖：
  - `utils.paths`, `utils.run_metadata`, `utils.varsets`

### `staged_optuna_pipeline.py`
- 输入：
  - CLI: `<train_tag> [--stage-group ...] [--resume]`
  - Env: `OPTUNA_N_TRIALS`
  - ROOT: `PbPb23 MC` + `PbPb24 DATA`
- 功能：
  - Step1~Step5 分阶段扫描
  - `scale_pos_weight` 可按 `ratio` 基线缩放
  - 目标：`max validation AUC (staged Step1-6)`（见 metadata）
- 输出：
  - 同主线 condor 目录结构

### `batch_apply_scores.py`
- 输入：
  - train tags 列表
  - 可选 `--output-tag`、`--data-input`、`--output-prefix`
- 输出：
  - `selected_events/<output_tag>/<prefix>DATA_with_score.root`
  - `selected_events/<output_tag>/<prefix>MC_with_score.root`
  - `selected_events/<output_tag>/<prefix>batch_apply_summary.json`
- 关键特性：
  - 自动跳过坏模型（记录 `skipped_models`）
  - 要求同组模型输入列一致
  - 默认 DATA 输入已切换为 PbPb24

### `batch_draw_scores.py`
- 输入：
  - grouped DATA ROOT + train tags
  - 可选 `--output-tag`、`--output-prefix`
- 输出：
  - `selected_events/<output_tag>/<prefix><train_tag>/DATA_<fid>_cut*.pdf`
  - `selected_events/<output_tag>/<prefix>X3872_sigma_summary_<fid>.md`
- 关键特性：
  - 自动识别 pb23v6 fid profile
  - 计算 `3.872` bin 的近似 sigma

### `workflow_archive/legacy_non_dag/XGBoost.py`
- 输入：
  - `<train_tag>`
  - ppRef MC/DATA
  - varset 从 tag 自动推断
- 输出：
  - `xgb_output/models/<train_tag>/...`
  - `xgb_output/training/<train_tag>/xgb_score.pdf, ROC.pdf, test_roc.json, ...`
- 当前作图设置：
  - `xgb_score` 50 bin
  - test 填充、train 点+误差棒（含 xerr），点大小已缩小
  - ROC 轴：Signal efficiency vs Background rejection

### `shap_importance.py`
- 输入：
  - `<train_tag> [max_events]`
  - 模型+scaler+config
  - ppRef MC/DATA
- 输出：
  - `xgb_output/shap/<train_tag>/shap_importance*.json`
  - `shap_summary.pdf`, `shap_bar.pdf`, `shap_cumulative.pdf`

## 4. 当前 `<varset>`（来自 `utils/varsets.py`）
- `4v`, `4v2`, `5v`, `6v`, `7v`, `7v2`, `8v2`, `8v3`, `9v`

已删除（含 `Bnorm_svpvDistance_2D`）：
- `10v`, `8v`, `8v4`, `9v2`

## 5. 已完成内容（近期）
- 单模型 DAG 已包含 SHAP 节点
- 修复了单模型 DAG 的 `utils` 导入失败（`PYTHONPATH`）
- 单模型训练图样式更新（ROC 与 xgb_score）
- 主线批处理支持 `draw_only`、`data_input_override`、`output_prefix`
- 主线训练与 batch apply 的 DATA 输入统一切换到：
  `/eos/user/h/hmarques/RUN3_Data_MC_sharing/X3872/PbPb24/flat_ntmix_PbPb24_DATA.root`
- 已提交主线训练（每10版本一组）：
  - `pb24v1_4v_4o200_v1-v100`
  - `pb24v1_5v_4o200_v1-v100`

## 6. 未完成内容
- 元数据 split 字段与单模型实际拆分不一致（metadata 仍写 0.8/0.1/0.1）
- 缺少自动化测试（目前主要靠 Condor 真跑验证）
- `submit_single_legacy_dag.sh` 尚未内置强制重提模式（需手动 `condor_submit_dag -f`）

## 7. 当前最重要待办（建议优先级）
1. 统一 metadata 的 split 与真实训练拆分
2. 给单模型 submit 脚本加 `--force`（转发到 `condor_submit_dag -f`）
3. 把 `batch_draw_scores.py` 的 `score_cuts` 外置为配置文件
4. 增加最小回归测试：
   - varset 推断
   - 路径生成
   - batch apply skip 行为

## 8. 风险点与疑难点
- EOS/AFS 混用时容易出现“代码已改但 AFS wrapper 旧版本”问题
- 同名 DAG 二次提交会失败，必须 `-f`
- `utils/paths.py` 变更会影响历史结果定位与兼容
- 主线与单模型样本来源不同，结果不可直接横向比较
- `staged_optuna_pipeline.py` 内置大量 stage 配置，改动前要确认版本映射
- EOS 瞬时 I/O 抖动会导致 draw 读 ROOT 失败（`FSSpecSource received 0 bytes`），通常只需重跑 draw

## 9. 快速继续开发清单（新会话可直接执行）
1. 先读：`AGENTS.md` -> `utils/varsets.py` -> `utils/paths.py`
2. 再读：`run_staged.sh`、`make_dagman_workflow.py`、`run_batch_compare.sh`
3. 变更后先跑：
   - `python -m py_compile ...`
   - 单模型 DAG smoke test
4. 再跑主线小批（例如 `v1-v10`）
5. 检查 `logs/job_*.err` 与 `selected_events/` 输出完整性
