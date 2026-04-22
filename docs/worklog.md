# Worklog

## 2026-04-20

### 今天做了什么
- 主线训练脚本改造：
  - 增加训练数据源选择（`2023`/`2024`）。
  - 增加筛选 profile（`legacy`/`pb24v2`），并让 `pb24v2` 固定匹配 2024 数据与新动力学筛选。
  - 训练后新增 `optuna_top20_ranges.json` 产物。
- 主线 DAG 参数链路扩展：
  - 从 `submit_dagman_workflow.sh` 到 `run_staged.sh` 透传 `dataset_year`/`selection_profile`。
- 画图脚本与流程改造：
  - `batch_draw_scores.py` 增加 `--fid-profile auto|fid|fid3`。
  - `run_batch_compare.sh`、`submit_batch_compare*.sub`、`make_dagman_workflow.py`、`submit_dagman_workflow.sh` 打通 `fid_profile` 透传。
- 执行任务管理：
  - kill 旧 `pb24v2_8v/8v4_4o200_(v|3v)1-100` 任务。
  - 删除旧任务对应输出与日志（EOS+AFS）。
  - 用 `fid3` 重新提交 4 组共 40 个 DAG 批次。

### 得到了什么结果
- 代码层面：
  - 训练数据源、筛选 profile、top20 范围 JSON、fid profile 可选均已落地。
- 运行层面：
  - 旧任务已清空（`condor_q` 对应约束归零）。
  - 新一轮 `fid3` 任务已在队列中，`condor_q -dag` 可见 40 个批次。
  - 抽查 DAG 文件确认 FINAL 节点包含 `fid_profile=\"fid3\"`。

### 哪些方案失败
- 初次重提时 `submit_batch_compare_single.sub` 用了字面 `\"\"` 占位参数，导致 Condor dry-run 报 `illegal unescaped double-quote`。
  - 修复：改为 submit 文件变量 `data_input_override`/`output_prefix`（空字符串由变量提供），避免字面引号。
- 单次重提过程中遇到 EOS `dags/` 写文件偶发 `PermissionError`。
  - 修复：删除冲突 DAG 文件后重提，并按“缺失批次补提”完成全量提交。

### 新发现/风险
- EOS 目录在高并发/抖动时可能出现瞬时写失败，影响 DAG 文件生成；需要重试策略。
- 清理与重提是高风险操作，必须先按 tag 精确过滤再执行 `condor_rm` 与删除命令。
- 当前批次仍在运行中，最终质量风险在于：
  - 某些训练节点失败导致 batch apply/draw 不完整；
  - 需要后续按日志做补跑与结果完整性核查。

## 2026-04-20（补充）

### 今天做了什么
- 按“完全重跑”方案执行了本轮重提前清理：
  - 删除 AFS 上旧 `pb24v2_8v* / pb24v2_8v4*` 对应 DAG/rescue/log。
  - 核对并使用当前提交流程脚本进行全量重提。
- 全量提交 40 个 DAG 批次（每 10 版本一组）：
  - `pb24v2_8v_4o200_v1-v100`
  - `pb24v2_8v_4o200_3v1-3v100`
  - `pb24v2_8v4_4o200_v1-v100`
  - `pb24v2_8v4_4o200_3v1-3v100`
  - 统一参数：`dataset_year=2024`、`selection_profile=pb24v2`、`fid_profile=fid3`、`optuna_n_trials=200`

### 得到了什么结果
- 提交结果：`ok=40, fail=0`。
- 即时状态检查：`condor_q -dag` 可见 40 个相关 DAG 在队列中。
- 提交后日志关键字扫描未见上一轮两类系统性报错：
  - `Unknown stage group`
  - `FileNotFoundError: .../auto`

### 哪些方案失败
- 本次全量重提流程中无新增失败方案；此前失败主要来自旧版 AFS wrapper 与参数接口不一致，已在本轮前清理并重提。

### 新发现/风险
- `condor_q -dag` 按关键字检索有时会受截断/显示影响，计数需配合 DAG 文件唯一计数交叉验证。
- 40 组全量在跑，仍存在运行期失败风险（节点超时、I/O 抖动、单节点异常），需要在完成后统一补跑与验收。
