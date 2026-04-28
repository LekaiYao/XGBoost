# Handoff

## 0. 当前状态（2026-04-24）

### 当前目标
- 验收 `pb24v2` 的 apply/draw 重跑结果（40 组，`fid3`），并补齐缺失模型对应批次输出。

### 已完成内容
- 数据输入已统一为“按年份成组”：
  - 训练：
    - `2023`: `PbPb23_MC` + `PbPb23_DATA0`
    - `2024`: `PbPb24_MC` + `PbPb24_DATA_SMALL`
  - apply/draw：
    - `2023`: `PbPb23_MC` + `PbPb23_DATA`
    - `2024`: `PbPb24_MC` + `PbPb24_DATA`
  - 已移除 legacy 混配 fallback（`23MC + 24DATA`）。
- 修复了 POST_BATCH 参数链路问题：
  - 解决空参数塌缩导致 `auto` 被当成 DATA 路径的问题。
  - 解决 DAG 指令（`fid3`/`dataset_year=2024`）与运行值不一致的问题。
  - 当前 `run_batch_compare.sh` 实际队列参数已为：`... 2024 __EMPTY__ __EMPTY__ fid3`。
- 已完成一次“停止旧任务 + 仅重做 apply/draw”：
  - 已中止仍在运行的旧 DAG 与子作业。
  - 已删除上一轮 40 组 `selected_events` 输出。
  - 已基于已有训练模型重提 40 组 batch apply+draw（不重训），提交结果 `ok=40, fail=0`。
- 目录清理脚本已落地并执行：
  - 新增 `cleanup_selected_events.py`
  - 本次执行结果：移动旧目录 74 个、删除 >500MB ROOT 88 个、释放约 `161.77 GB`。
- 代码已同步到 GitHub：
  - `main` 最新提交：`8929047`

### 未完成内容
- `fid3` 重跑批次的完整性验收未完成（需按组检查 apply summary 与 draw 输出）。
- 当前 40 组对应训练模型并非 400/400 完整，现有约 396 个模型目录；缺失模型会触发 batch apply skip 机制。
- 尚未输出最终“缺失模型批次补跑清单 + 完成率摘要”。

### 2024数据集切换问题（简要）
1. 历史输入存在混配风险：曾出现 `23MC + 24DATA` 的 fallback 逻辑，不符合“按年份成组一致”要求。
2. POST_BATCH 参数错位：空参数在 submit/脚本位置参数链中塌缩，`auto` 被误当作 DATA 输入路径。
3. 指令与运行不一致：DAG 中设置了 `fid3`/`dataset_year=2024`，运行日志却出现 `FID profile: auto` 与 `Dataset year:` 为空。
4. AFS/EOS 同步风险：提交侧与代码侧文件若不同步，会导致“代码已改、运行仍旧配置”的问题反复出现。

### 下一步
1. 逐组核查 40 组 `batchcmp_fid3redo` 输出完整性（`MC/DATA_with_score.root`、`batch_apply_summary.json`、`X3872_sigma_summary_fid3.md`）。
2. 整理并确认 4 个缺失模型 tag 的影响批次，决定是否补训后再补做 apply/draw。
3. 形成最终验收表（成功组/跳过组/失败组）并更新 `docs/worklog.md` 与 `docs/handoff.md`。

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
  - `submit_batch_compare.sub`
  - `batch_apply_scores.py`
  - `batch_draw_scores.py`
- 运维脚本：
  - `cleanup_selected_events.py`
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
  - CLI可选: `--dataset-year {2023,2024}`、`--selection-profile {legacy,pb24v2}`
  - Env: `OPTUNA_N_TRIALS`
  - ROOT(训练):
    - `2023`: `PbPb23_MC` + `PbPb23_DATA0`
    - `2024`: `PbPb24_MC` + `PbPb24_DATA_SMALL`
  - varset 来自 `utils/varsets.py`
- 输出：
  - `xgb_output/models/<batch>/<train_tag>/...`
  - `xgb_output/training/<batch>/<train_tag>/...`
  - `run_metadata.json`
- 依赖：
  - `utils.paths`, `utils.run_metadata`, `utils.varsets`

### `staged_optuna_pipeline.py`
- 输入：
  - CLI: `<train_tag> [--stage-group ...] [--resume] [--dataset-year {2023,2024}]`
  - Env: `OPTUNA_N_TRIALS`
  - ROOT(训练):
    - `2023`: `PbPb23_MC` + `PbPb23_DATA0`
    - `2024`: `PbPb24_MC` + `PbPb24_DATA_SMALL`
- 功能：
  - Step1~Step5 分阶段扫描
  - `scale_pos_weight` 可按 `ratio` 基线缩放
  - 目标：`max validation AUC (staged Step1-6)`（见 metadata）
- 输出：
  - 同主线 condor 目录结构

### `batch_apply_scores.py`
- 输入：
  - train tags 列表
  - 可选 `--output-tag`、`--data-input`、`--output-prefix`、`--dataset-year`
- ROOT(apply):
  - `2023`: `PbPb23_MC` + `PbPb23_DATA`
  - `2024`: `PbPb24_MC` + `PbPb24_DATA`
- 输出：
  - `selected_events/<output_tag>/<prefix>DATA_with_score.root`
  - `selected_events/<output_tag>/<prefix>MC_with_score.root`
  - `selected_events/<output_tag>/<prefix>batch_apply_summary.json`
- 关键特性：
  - 自动跳过坏模型（记录 `skipped_models`）
  - 要求同组模型输入列一致
  - 年份可由 tag 推断（`pb23*`/`pb24*`）或显式传参覆盖

### `batch_draw_scores.py`
- 输入：
  - grouped DATA ROOT + train tags
  - 可选 `--output-tag`、`--output-prefix`
- 输出：
  - `selected_events/<output_tag>/<prefix><train_tag>/DATA_<fid>_cut*.pdf`
  - `selected_events/<output_tag>/<prefix>X3872_sigma_summary_<fid>.md`
- 关键特性：
  - 自动识别 pb23v6 fid profile
  - 支持显式 `--fid-profile auto|fid|fid3`
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
- `4v`, `4v2`, `5v`, `5v2`, `5v3`, `6v`, `6v2`, `6v3`, `7v2`, `7v3`, `7v4`, `8v`, `8v2`, `8v3`, `8v4`, `9v`

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
