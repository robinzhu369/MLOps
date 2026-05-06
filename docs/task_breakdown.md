# 风控建模 Agent MVP — 开发任务拆分

> 基于 `docs/risk_modeling_agent.md` v0.4 技术方案拆分  
> 拆分原则：按模块纵向切分，每个任务可独立开发和测试，任务间依赖关系明确标注

---

## 阶段一：项目基础设施

### Task 1.1 项目初始化与工程骨架

**目标：** 搭建项目目录结构、依赖管理、基础配置

**交付物：**
- `requirements.txt` 包含所有 MVP 依赖（langgraph, autogluon, streamlit, scikit-learn, pandas, numpy, pytest 等）
- `config.yaml` 全局配置（LLM 接口地址、模型参数、文件路径等）
- 按技术方案第 18 节创建完整目录结构（agents/, tools/, core/, pipelines/, tests/, artifacts/, data/）
- `app.py` Streamlit 入口文件（空壳）
- `.gitignore`（忽略 artifacts/, data/, __pycache__, .env 等）

**验收标准：**
- `pip install -r requirements.txt` 成功
- `pytest` 可运行（无测试也不报错）
- `streamlit run app.py` 可启动空白页面

---

### Task 1.2 Core 模块实现

**目标：** 实现核心基础模块，供所有 Agent 和 Tool 使用

**交付物：**
- `core/state.py` — `RiskModelingProjectState` 和 `UploadedFileState` TypedDict（按技术方案第 17 节）
- `core/permissions.py` — `AGENT_TOOL_PERMISSIONS` 白名单 + `check_tool_permission()` 函数（按第 16 节）
- `core/schemas.py` — 公共数据结构定义（字段语义、质量报告、清洗计划等 Pydantic model）
- `core/constants.py` — 常量定义（数据类型枚举、Pipeline 类型枚举、风险关键词列表等）
- `core/exceptions.py` — 自定义异常类

**验收标准：**
- 所有模块可正常 import
- 单元测试覆盖 `check_tool_permission()` 正向和反向场景

---

### Task 1.3 Trace 与日志工具

**目标：** 实现全流程追踪和日志记录机制

**交付物：**
- `tools/trace_tools.py` — Agent trace 写入（`agent_trace.json`）
- 人工确认记录写入（`human_confirmations.json`）
- 清洗日志写入（`cleaning_log.json`）
- 统一的 append/read 接口

**验收标准：**
- 可写入和读取 trace 记录
- JSON 格式符合技术方案第 15 节示例
- 单元测试通过

---

## 阶段二：数据上传与元数据提取

### Task 2.1 文件上传与元数据提取工具

**目标：** 实现文件上传保存和元数据提取

**交付物：**
- `tools/file_tools.py` — `save_uploaded_file()`、`inspect_file_format()`
- `tools/metadata_tools.py` — `extract_file_metadata()`（行数、列数、列名、列类型、缺失率、唯一值比例、脱敏样例值）
- 支持 CSV 和 Excel 格式

**验收标准：**
- 上传 CSV/Excel 文件后正确保存到 `artifacts/{project_name}/raw_files/`
- 元数据提取结果包含：n_rows, n_cols, columns, column_profiles, sample_values_masked
- 单元测试通过

---

### Task 2.2 DataIntakeAgent

**目标：** 实现数据接入 Agent，协调文件上传和元数据提取流程

**交付物：**
- `agents/data_intake_agent.py` — 调用 file_tools 和 metadata_tools，输出文件元数据到 State
- 遵循 ReAct 结构，写入 agent_trace

**依赖：** Task 1.2, Task 1.3, Task 2.1

**验收标准：**
- 给定上传文件，能正确提取元数据并更新 State
- agent_trace.json 有对应记录

---

## 阶段三：字段语义解析

### Task 3.1 数据字典解析工具与 Agent

**目标：** 实现数据字典上传解析功能

**交付物：**
- `tools/dictionary_tools.py` — `parse_data_dictionary()`、`validate_dictionary_columns()`、`map_dictionary_to_dataset()`
- `agents/data_dictionary_parser_agent.py` — 读取数据字典，解析字段业务含义，识别 label/ID/时间/敏感字段，生成字段语义表

**验收标准：**
- 给定符合第 6.2 节格式的数据字典 CSV，能正确解析出所有字段角色
- 输出格式符合第 7.1 节 JSON 示例
- 单元测试通过

---

### Task 3.2 LLM 字段语义解析工具与 Agent

**目标：** 无数据字典时，使用 LLM 推断字段语义

**交付物：**
- `tools/field_semantic_tools.py` — `parse_field_semantics_by_llm()`、`merge_dictionary_and_llm_semantics()`
- `agents/field_semantic_parser_agent.py` — 基于字段名、类型、画像、脱敏样例调用 LLM 解析语义
- Prompt 模板设计（识别 label、ID、时间、金额、方向、敏感、疑似泄露字段）

**依赖：** Task 2.1（需要元数据作为 LLM 输入）

**验收标准：**
- 给定典型风控数据集元数据，LLM 能正确识别关键字段角色
- 输出包含 confidence 和 need_human_review 字段
- 输出格式符合第 7.2 节 JSON 示例

---

## 阶段四：数据类型识别与 Pipeline 路由

### Task 4.1 数据类型分类工具与 Agent

**目标：** 实现规则 + LLM 融合的数据类型识别

**交付物：**
- `tools/data_type_tools.py` — `classify_data_type_by_rules()`、`classify_data_type_by_llm()`、`merge_data_type_classification()`
- `agents/data_type_classifier_agent.py` — 融合规则和 LLM 结果，输出数据类型和推荐 Pipeline
- 规则识别器：基于字段特征（是否有 transaction_time、account_id、label 等）打分

**依赖：** Task 3.1 或 Task 3.2（需要字段语义结果）

**验收标准：**
- 能正确区分 structured_modeling_table、transaction_flow_table、main_table
- 输出格式符合第 8.2 节 JSON 示例
- 低置信度时 need_human_review = true
- 单元测试覆盖三种主要数据类型

---

### Task 4.2 Pipeline 路由 Agent

**目标：** 根据数据类型分类结果路由到对应 Pipeline

**交付物：**
- `agents/pipeline_router_agent.py` — 根据 final_detected_type 决定进入哪个 Pipeline
- `pipelines/__init__.py` — Pipeline 注册和调度入口

**依赖：** Task 4.1

**验收标准：**
- structured_modeling_table → structured_modeling_pipeline
- transaction_flow_table → transaction_feature_pipeline
- main_table + transaction_flow_table → main_plus_transaction_pipeline
- unknown → manual_configuration_pipeline

---

## 阶段五：数据质量分析

### Task 5.1 通用数据质量分析工具与 Agent

**目标：** 实现建模前数据质量全面检查

**交付物：**
- `tools/data_quality_tools.py` — 实现以下工具函数：
  - `analyze_duplicates()` — 完全重复行、主键重复
  - `analyze_missing_values()` — 缺失率、高缺失字段
  - `analyze_outliers()` — IQR 方法检测异常值
  - `analyze_type_mismatch()` — 字段类型错误
  - `analyze_label_quality()` — label 缺失率、分布不均衡
  - `analyze_key_quality()` — 主键唯一性
  - `generate_data_quality_report()` — 汇总报告 + overall_quality_score
- `agents/data_quality_agent.py` — 调用上述工具，生成质量报告

**验收标准：**
- 输出格式符合第 9.1 节 JSON 示例
- 能检测出重复、缺失、异常、类型错误等问题
- 单元测试覆盖各类质量问题场景

---

### Task 5.2 交易流水质量分析工具与 Agent

**目标：** 针对交易流水表的专项质量检查

**交付物：**
- `tools/transaction_quality_tools.py` — `analyze_transaction_quality()`
- `agents/transaction_quality_agent.py` — 检查 transaction_id 重复、account_id 缺失、时间范围、负金额比例、与主表关联覆盖率

**依赖：** Task 5.1

**验收标准：**
- 输出格式符合第 9.2 节 JSON 示例
- 能正确计算关联覆盖率
- 单元测试通过

---

## 阶段六：数据清洗

### Task 6.1 清洗方案规划 Agent

**目标：** 根据质量报告生成可选清洗方案

**交付物：**
- `agents/data_cleaning_planner_agent.py` — 根据 data_quality_report 生成 cleaning_plan
- 输出包含各维度策略建议和 protected_columns

**依赖：** Task 5.1

**验收标准：**
- 输出格式符合第 10.1 节 JSON 示例
- label、ID 字段自动加入 protected_columns
- 单元测试通过

---

### Task 6.2 清洗执行工具

**目标：** 实现确定性数据清洗操作

**交付物：**
- `tools/data_cleaning_tools.py` — `execute_cleaning_plan()`
- 支持白名单动作：删除重复行、按主键去重、缺失值填充、增加缺失指示变量、异常值缩尾/截断、字段类型转换、删除高缺失字段、删除常数列
- 原始数据不覆盖，输出 cleaned_{file_name}.csv
- 写入 cleaning_log.json

**依赖：** Task 6.1, Task 1.3

**验收标准：**
- 输出格式符合第 10.3 节 JSON 示例
- 原始文件未被修改
- cleaning_log.json 记录所有执行动作
- before_shape 和 after_shape 正确
- 单元测试覆盖各清洗动作

---

## 阶段七：交易流水特征工程

### Task 7.1 账户日维度特征生成

**目标：** 按 account_id + transaction_date 粒度生成日维度特征

**交付物：**
- `tools/transaction_feature_tools.py` 中实现 `build_account_daily_features()`
- 生成第 11.2 节定义的 15 个日维度特征
- 输出 `transaction_daily_features.csv`

**依赖：** Task 6.2（需要清洗后的流水数据）

**验收标准：**
- 输出粒度为 account_id + transaction_date
- 所有 15 个特征正确计算
- 单元测试验证计算逻辑

---

### Task 7.2 观察点窗口特征生成

**目标：** 基于观察点（apply_date）生成历史时间窗口特征

**交付物：**
- `tools/transaction_feature_tools.py` 中实现 `build_account_window_features()`、`validate_transaction_feature_cutoff()`
- 支持 7d/14d/30d/60d/90d/180d 窗口
- 实现活跃度、金额规模、收入支出、波动性、行为偏好、大额交易、对手方分散度共 7 组特征
- 严格保证 transaction_time < apply_date（时间穿越校验）
- 输出 `transaction_window_features.csv`

**依赖：** Task 7.1

**验收标准：**
- 所有窗口特征仅使用观察点之前的数据
- `validate_transaction_feature_cutoff()` 能检测时间穿越
- 特征数量和命名符合第 11.3 节定义
- 单元测试覆盖时间截断正确性

---

### Task 7.3 TransactionFeatureAgent

**目标：** 编排交易流水特征生成流程

**交付物：**
- `agents/transaction_feature_agent.py` — 调用 schema 推断、日维度特征、窗口特征、特征画像工具
- `tools/transaction_feature_tools.py` 中实现 `infer_transaction_schema()`、`profile_generated_features()`

**依赖：** Task 7.1, Task 7.2

**验收标准：**
- 端到端：给定清洗后流水表 + 主表，能生成完整特征表
- agent_trace 有完整记录

---

## 阶段八：风险字段检查

### Task 8.1 RiskGuardAgent 与工具

**目标：** 建模前检查并排除风险字段

**交付物：**
- `agents/risk_guard_agent.py`
- 工具函数（可放在现有 tools 中）：
  - `detect_leakage_columns()` — 基于关键词列表（第 12.1 节）检测疑似标签泄露字段
  - `detect_id_columns()` — 检测 ID 类字段
  - `detect_high_missing_columns()` — 高缺失字段
  - `detect_time_leakage_candidates()` — 时间穿越候选字段

**验收标准：**
- 能检测出包含 "dpd", "overdue", "default", "逾期" 等关键词的字段
- 输出 drop_recommendations 和 warnings
- 输出格式符合第 12.1 节 JSON 示例

---

## 阶段九：建模与评估

### Task 9.1 AutoML 建模工具与 Agent

**目标：** 集成 AutoGluon 进行二分类建模

**交付物：**
- `tools/modeling_tools.py` — `train_autogluon_binary()`
- `tools/split_tools.py` — 训练集/测试集划分
- `agents/modeling_agent.py` — 协调数据准备、特征合并、模型训练
- 输出 model/、leaderboard.csv、train.csv、test.csv

**依赖：** Task 8.1（需要排除风险字段后的数据）

**验收标准：**
- AutoGluon 训练成功，生成 leaderboard
- 模型保存到 artifacts/{project_name}/model/
- 支持 time_limit 配置

---

### Task 9.2 模型评估工具与 Agent

**目标：** 计算模型评估指标

**交付物：**
- `tools/metric_tools.py` — `evaluate_binary_model()`
- `agents/evaluation_agent.py`
- 计算：AUC、KS、Accuracy、Precision、Recall、F1、混淆矩阵
- KS 计算按第 13.2 节实现
- 输出 metrics.json

**依赖：** Task 9.1

**验收标准：**
- 所有指标计算正确
- KS 函数单元测试通过
- 输出 metrics.json 格式正确

---

### Task 9.3 阈值策略分析工具与 Agent

**目标：** 生成不同阈值下的业务指标表

**交付物：**
- `tools/strategy_tools.py` — `build_threshold_table()`
- `agents/strategy_agent.py`
- 输出：阈值、通过率、拒绝率、通过客群坏账率、坏样本捕获率
- 输出 threshold_table.csv

**依赖：** Task 9.1

**验收标准：**
- 阈值表格式符合第 13.3 节
- 各指标计算逻辑正确
- 单元测试通过

---

### Task 9.4 特征重要性与解释 Agent

**目标：** 输出特征重要性排序

**交付物：**
- `tools/explain_tools.py` — `compute_feature_importance()`
- `agents/explain_agent.py`
- 输出 feature_importance.csv

**依赖：** Task 9.1

**验收标准：**
- 特征重要性排序正确
- 输出包含 feature_name 和 importance_score

---

### Task 9.5 报告生成 Agent

**目标：** 生成 Markdown 格式的模型评估报告

**交付物：**
- `tools/report_tools.py` — `generate_markdown_report()`
- `agents/report_agent.py`
- 报告内容覆盖第 23.3 节验收要求的所有问题
- 输出 model_report.md

**依赖：** Task 9.2, Task 9.3, Task 9.4

**验收标准：**
- 报告包含数据概况、质量分析摘要、清洗记录、模型指标、阈值策略、特征重要性、风险提示、优化建议
- Markdown 格式可正常渲染

---

## 阶段十：LangGraph Pipeline 编排

### Task 10.1 结构化建模 Pipeline

**目标：** 编排场景 A 的完整流程

**交付物：**
- `pipelines/structured_modeling_pipeline.py` — LangGraph 图定义
- 节点顺序：DataIntake → FieldSemantic → DataType → DataQuality → CleaningPlan → CleaningExecute → RiskGuard → Modeling → Evaluation → Strategy → Explain → Report
- 所有 HumanReviewGate 节点正确插入

**依赖：** 阶段二至九所有 Agent 和 Tool

**验收标准：**
- 给定场景 A 数据（带 label 的结构化宽表），能端到端跑通
- 所有人工确认节点正确暂停等待

---

### Task 10.2 交易流水特征 Pipeline

**目标：** 编排场景 B 的完整流程

**交付物：**
- `pipelines/transaction_feature_pipeline.py` — LangGraph 图定义
- 节点顺序：DataIntake → FieldSemantic → DataType → TransactionQuality → CleaningPlan → CleaningExecute → TransactionFeature → Report
- 无建模步骤，只输出特征表和画像报告

**依赖：** 阶段二至七

**验收标准：**
- 给定场景 B 数据（无 label 的交易流水表），能端到端跑通
- 输出 transaction_daily_features.csv 和 transaction_window_features.csv

---

### Task 10.3 主表 + 流水表联合建模 Pipeline

**目标：** 编排场景 C 的完整流程

**交付物：**
- `pipelines/main_plus_transaction_pipeline.py` — LangGraph 图定义
- 包含表关系识别、流水特征生成、特征合并回主表、建模全流程
- 时间穿越校验节点

**依赖：** Task 10.1, Task 10.2

**验收标准：**
- 给定主表 + 流水表，能正确关联、生成窗口特征、合并、建模
- 时间穿越校验通过

---

## 阶段十一：Streamlit 前端

### Task 11.1 页面 1-3：上传与识别

**目标：** 实现项目创建、文件上传、字段语义展示、数据类型识别页面

**交付物：**
- 页面 1：项目名称输入、文件上传（多文件）、数据字典上传、LLM 配置、开始按钮
- 页面 2：字段语义解析结果表格展示、确认/修改/重新解析操作
- 页面 3：数据类型识别结果展示、确认/修改操作

**验收标准：**
- 文件上传成功并触发后端 Agent 流程
- 字段语义表格可编辑
- 用户确认后流程继续

---

### Task 11.2 页面 4-6：质量分析与清洗

**目标：** 实现数据质量展示、清洗方案确认、Pipeline 路由确认页面

**交付物：**
- 页面 4：质量分析结果分维度展示
- 页面 5：各维度清洗策略选择器、执行按钮
- 页面 6：Pipeline 路由建议展示、确认按钮

**验收标准：**
- 质量问题清晰展示
- 用户可选择不同清洗策略
- 确认后触发清洗执行

---

### Task 11.3 页面 7-8：执行过程与结果

**目标：** 实现 Agent 执行进度展示和最终结果展示

**交付物：**
- 页面 7：Agent 执行状态实时展示（已完成/进行中/等待中）
- 页面 8：模型排行榜、核心指标、阈值策略表、特征重要性图表、报告下载

**验收标准：**
- 执行过程有进度反馈
- 结果页面数据完整、可视化清晰
- 报告可下载

---

## 阶段十二：集成测试与端到端验证

### Task 12.1 测试数据准备

**目标：** 准备用于端到端测试的模拟数据

**交付物：**
- 场景 A 测试数据：loan_application.csv（含 label）
- 场景 B 测试数据：transaction_flow.csv（无 label）
- 场景 C 测试数据：loan_application.csv + transaction_flow.csv
- 测试用数据字典 CSV

---

### Task 12.2 端到端集成测试

**目标：** 验证三个场景的完整流程

**交付物：**
- 集成测试脚本覆盖场景 A/B/C
- 验证所有 artifacts 输出文件正确生成
- 验证 agent_trace.json 完整性
- 验证 human_confirmations.json 记录正确

**验收标准：**
- 三个场景均可端到端跑通
- 无时间穿越问题
- 所有输出文件格式正确

---

## 任务依赖关系总览

```text
阶段一 (1.1, 1.2, 1.3) — 无依赖，最先开发
    ↓
阶段二 (2.1, 2.2) — 依赖阶段一
    ↓
阶段三 (3.1, 3.2) — 依赖阶段二
    ↓
阶段四 (4.1, 4.2) — 依赖阶段三
    ↓
阶段五 (5.1, 5.2) — 依赖阶段二（可与阶段三四并行开发工具层）
    ↓
阶段六 (6.1, 6.2) — 依赖阶段五
    ↓
阶段七 (7.1, 7.2, 7.3) — 依赖阶段六
    ↓
阶段八 (8.1) — 依赖阶段三
    ↓
阶段九 (9.1→9.2→9.3→9.4→9.5) — 依赖阶段六和阶段八
    ↓
阶段十 (10.1, 10.2, 10.3) — 依赖阶段二至九所有模块
    ↓
阶段十一 (11.1, 11.2, 11.3) — 依赖阶段十（Pipeline 可用后开发 UI）
    ↓
阶段十二 (12.1, 12.2) — 依赖阶段十一
```

---

## 建议开发顺序（可并行）

| 优先级 | 任务 | 说明 |
|:---:|------|------|
| P0 | 1.1, 1.2, 1.3 | 基础设施，阻塞所有后续任务 |
| P1 | 2.1, 2.2 | 数据入口，核心链路起点 |
| P1 | 5.1, 6.2 | 数据质量和清洗工具可独立开发 |
| P2 | 3.1, 3.2, 4.1, 4.2 | 字段解析和类型识别 |
| P2 | 7.1, 7.2 | 特征工程工具可独立开发 |
| P3 | 5.2, 6.1, 7.3, 8.1 | Agent 编排层 |
| P3 | 9.1, 9.2, 9.3, 9.4, 9.5 | 建模评估链路 |
| P4 | 10.1, 10.2, 10.3 | Pipeline 集成 |
| P5 | 11.1, 11.2, 11.3 | 前端页面 |
| P6 | 12.1, 12.2 | 端到端验证 |
