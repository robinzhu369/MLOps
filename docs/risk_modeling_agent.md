# 风控建模 Agent MVP 技术方案

> 版本：v0.4  
> 适用开发工具：Codex / Claude Code / Cursor / Windsurf  
> 产品方向：数据上传即识别、数据字典优先、无字典时大模型字段解析、人在回路清洗确认、ReAct 受控执行、多 Pipeline 路由风控建模与交易流水特征挖掘。

---

## 1. 产品目标

本项目目标是开发一个 **风控建模 Agent MVP**，实现从数据上传到模型评估报告的最小可落地闭环。

系统需要支持：

1. 用户上传一个或多个数据文件。
2. 用户可选上传数据字典。
3. 若用户跳过数据字典，则使用大模型基于字段名、字段类型、字段画像和脱敏样例进行字段语义解析。
4. 上传阶段使用大模型 + 规则识别数据类型：普通结构化建模表、交易流水表、主表、辅助表或不确定数据。
5. 根据数据类型自动进入不同 Pipeline。
6. 建模前进行数据质量分析，包括重复、缺失、异常、类型错误、主键质量、标签质量、时间字段质量和多表关联质量。
7. 用户根据数据质量分析结果选择数据清洗策略。
8. 系统执行受控清洗，不覆盖原始数据，保留清洗日志。
9. 对交易流水表，按账户和交易日维度挖掘新特征，并按观察点生成时间窗口特征。
10. 对普通结构化宽表或主表 + 流水表组合，完成 AutoML 建模、评估、阈值策略分析、解释和报告生成。
11. 全流程体现人在回路控制、ReAct 推理行动控制、工具白名单控制、日志留痕和可复现。

---

## 2. MVP 最小落地场景

### 2.1 场景 A：普通结构化建模表

用户上传：

```text
loan_application.csv
```

该文件一行代表一个客户、一笔贷款或一次申请，包含 label 字段，例如：

```text
customer_id, loan_id, age, income, loan_amount, apply_date, credit_score, bad_flag
```

系统流程：

```text
上传数据
→ 可选上传数据字典
→ 字段语义解析
→ 数据类型识别
→ 数据质量分析
→ 用户确认清洗方案
→ 执行清洗
→ 风险字段检查
→ AutoML 建模
→ 模型评估
→ 阈值策略分析
→ 特征重要性
→ 报告生成
```

---

### 2.2 场景 B：交易流水表

用户上传：

```text
transaction_flow.csv
```

该文件一行代表一笔交易，通常没有 label 字段，例如：

```text
transaction_id, account_id, customer_id, transaction_time, transaction_amount, debit_credit_flag, channel, merchant_category, balance_after_txn
```

系统流程：

```text
上传流水表
→ 可选上传数据字典
→ 字段语义解析
→ 识别为交易流水表
→ 流水数据质量分析
→ 用户确认清洗方案
→ 执行清洗
→ 账户日维度特征生成
→ 账户窗口特征生成
→ 导出特征和报告
```

注意：如果用户只上传交易流水表且没有 label，则系统不直接训练监督学习模型，只生成流水画像和特征表。

---

### 2.3 场景 C：主表 + 交易流水表联合建模

用户上传：

```text
loan_application.csv
transaction_flow.csv
```

系统识别：

```text
loan_application.csv = 主建模表
transaction_flow.csv = 交易流水明细表
```

系统流程：

```text
多文件上传
→ 可选上传数据字典
→ 字段语义解析
→ 数据类型识别
→ 表角色确认
→ 数据质量分析
→ 用户确认清洗方案
→ 执行清洗
→ 表关系识别
→ 流水窗口特征生成
→ 合并回主表
→ 风险字段检查
→ AutoML 建模
→ 模型评估
→ 策略分析
→ 报告生成
```

核心约束：

```text
transaction_time < apply_date
```

所有交易流水窗口特征必须严格基于观察点之前的数据生成，防止时间穿越。

---

## 3. 总体架构

```text
Frontend: Streamlit
    ↓
Agent Orchestration: LangGraph
    ↓
Agents:
    DataIntakeAgent
    DataDictionaryParserAgent
    FieldSemanticParserAgent
    DataTypeClassifierAgent
    DataQualityAgent
    TransactionQualityAgent
    DataCleaningPlannerAgent
    PipelineRouterAgent
    TransactionFeatureAgent
    RiskGuardAgent
    ModelingAgent
    EvaluationAgent
    StrategyAgent
    ExplainAgent
    ReportAgent
    ↓
Tools:
    file_tools
    metadata_tools
    dictionary_tools
    field_semantic_tools
    data_type_tools
    data_quality_tools
    transaction_quality_tools
    data_cleaning_tools
    transaction_feature_tools
    modeling_tools
    metric_tools
    strategy_tools
    explain_tools
    report_tools
    trace_tools
    ↓
Storage:
    artifacts/{project_name}/
    data/
    MLflow，可选
```

---

## 4. 技术栈

### 4.1 MVP 技术栈

| 模块 | 技术 |
|---|---|
| 前端 | Streamlit |
| Agent 编排 | LangGraph |
| LLM 接口 | OpenAI 兼容接口 / 本地大模型接口 |
| 建模 | AutoGluon TabularPredictor |
| 指标计算 | scikit-learn |
| 数据处理 | pandas / numpy |
| 实验记录 | 本地 JSON + 可选 MLflow |
| 报告 | Markdown |
| 文件存储 | 本地 artifacts 目录 |
| 测试 | pytest |

### 4.2 设计原则

1. Agent 负责理解、规划、解释、判断和生成结构化建议。
2. Tool 负责真实的数据处理、特征生成、建模、评估和保存。
3. Agent 不允许执行任意代码。
4. Agent 只能调用权限白名单中的工具。
5. 关键节点必须进入人在回路确认。
6. 原始数据只读，不覆盖。
7. 所有中间结果保存为文件。
8. 所有 Agent 行动写入 `agent_trace.json`。
9. 所有人工选择写入 `human_confirmations.json`。
10. 所有清洗动作写入 `cleaning_log.json`。

---

## 5. 端到端主流程

```text
Start
  ↓
FileUpload
  ↓
Optional DataDictionaryUpload
  ↓
DataIntakeAgent
  ↓
MetadataExtractorTool
  ↓
DataDictionaryParserAgent 或 FieldSemanticParserAgent
  ↓
HumanReviewGate_FieldSemantics
  ↓
DataTypeClassifierAgent
  ↓
HumanReviewGate_DataType
  ↓
DataQualityAgent / TransactionQualityAgent
  ↓
HumanReviewGate_DataQuality
  ↓
DataCleaningPlannerAgent
  ↓
HumanReviewGate_CleaningPlan
  ↓
DataCleaningExecutorTool
  ↓
DataQualityRecheckAgent
  ↓
PipelineRouterAgent
  ↓
不同 Pipeline
```

---

## 6. 数据上传与数据字典设计

### 6.1 上传文件类型

支持上传：

```text
CSV
Excel
Parquet，V1 可选
```

支持上传一个或多个数据文件。

---

### 6.2 数据字典上传

用户可选上传数据字典文件。

数据字典建议包含以下字段：

| 字段 | 含义 |
|---|---|
| table_name | 表名 |
| column_name | 字段名 |
| column_cn_name | 中文名称 |
| business_meaning | 业务含义 |
| data_type | 数据类型 |
| value_range | 取值范围 |
| enum_mapping | 枚举解释 |
| is_label | 是否标签字段 |
| is_id | 是否 ID 字段 |
| is_time | 是否时间字段 |
| is_sensitive | 是否敏感字段 |
| available_time | 字段可获得时间 |
| source_system | 来源系统 |
| remark | 备注 |

示例：

```csv
table_name,column_name,column_cn_name,business_meaning,data_type,is_label,is_id,is_time,available_time
loan_application,bad_flag,是否逾期90天,放款后90天内是否逾期,int,yes,no,no,after_loan
loan_application,apply_date,申请日期,客户提交贷款申请日期,date,no,no,yes,application_time
transaction_flow,txn_amt,交易金额,单笔账户交易金额,float,no,no,no,transaction_time
transaction_flow,txn_time,交易时间,账户交易发生时间,datetime,no,no,yes,transaction_time
```

---

### 6.3 数据字典优先原则

```text
如果用户上传数据字典：
    优先使用数据字典解析字段语义。

如果用户未上传数据字典：
    使用大模型基于字段名、字段类型、缺失率、唯一值比例、脱敏样例值进行字段解析。

如果大模型字段解析置信度低：
    进入人工字段配置页面。
```

---

## 7. 字段语义解析

### 7.1 DataDictionaryParserAgent

职责：

```text
读取数据字典
解析字段业务含义
识别 label、ID、时间字段、敏感字段、可用时间
生成字段语义表
```

输出示例：

```json
{
  "dictionary_uploaded": true,
  "parsed_columns": [
    {
      "table_name": "loan_application",
      "column_name": "bad_flag",
      "business_meaning": "放款后90天内是否逾期",
      "role": "label",
      "available_time": "after_loan",
      "risk_level": "normal"
    },
    {
      "table_name": "loan_application",
      "column_name": "dpd30",
      "business_meaning": "30天逾期天数",
      "role": "post_loan_feature",
      "available_time": "after_loan",
      "risk_level": "high"
    }
  ],
  "warnings": []
}
```

---

### 7.2 FieldSemanticParserAgent

当用户跳过数据字典时启用。

职责：

```text
基于字段名、字段类型、字段画像和脱敏样例值，解析字段语义。
```

需要识别：

```text
label 字段
客户 ID 字段
账户 ID 字段
贷款 ID 字段
观察点时间字段
交易时间字段
金额字段
交易方向字段
敏感字段
疑似标签泄露字段
疑似贷后表现字段
```

输出示例：

```json
{
  "dictionary_uploaded": false,
  "field_semantics": {
    "customer_id": {
      "role": "customer_key",
      "confidence": 0.95
    },
    "apply_date": {
      "role": "observation_time",
      "confidence": 0.91
    },
    "bad_flag": {
      "role": "label",
      "confidence": 0.88
    },
    "dpd30": {
      "role": "possible_leakage_feature",
      "confidence": 0.92,
      "risk_level": "high"
    }
  },
  "need_human_review": true
}
```

---

### 7.3 HumanReviewGate_FieldSemantics

页面展示：

| 字段名 | 推断含义 | 字段角色 | 置信度 | 风险等级 |
|---|---|---|---:|---|
| bad_flag | 是否坏客户 | label | 0.91 | 正常 |
| dpd30 | 30天逾期 | 疑似泄露字段 | 0.92 | 高 |
| apply_date | 申请日期 | 观察点时间 | 0.89 | 正常 |

用户可操作：

```text
确认字段解析
修改字段角色
重新上传数据字典
跳过并进入人工配置
终止流程
```

---

## 8. 数据类型识别与 Pipeline 路由

### 8.1 可识别数据类型

| 类型 | 说明 | Pipeline |
|---|---|---|
| structured_modeling_table | 普通结构化建模宽表 | structured_modeling_pipeline |
| transaction_flow_table | 交易流水明细表 | transaction_feature_pipeline |
| main_table | 多表建模主表 | main_plus_transaction_pipeline 的主表 |
| auxiliary_table | 辅助信息表 | manual 或后续扩展 |
| unknown_or_ambiguous | 不确定数据 | manual_configuration_pipeline |

---

### 8.2 DataTypeClassifierAgent

职责：

```text
基于字段语义、元数据、字段类型、样例值、唯一值比例和规则打分，判断文件类型。
```

识别策略：

```text
规则识别器先打分
+
大模型基于元数据做语义判断
+
规则结果和大模型结果融合
+
低置信度或不一致时进入人工确认
```

输出示例：

```json
{
  "file_name": "transaction_flow.csv",
  "detected_data_type": "transaction_flow_table",
  "confidence": 0.93,
  "reasoning_summary": "字段中包含 account_id、transaction_time、transaction_amount、debit_credit_flag，且同一账户对应多笔记录，符合交易流水明细表特征。",
  "recommended_pipeline": "transaction_feature_pipeline",
  "detected_roles": {
    "account_key": "account_id",
    "customer_key": "customer_id",
    "transaction_time_col": "transaction_time",
    "amount_col": "transaction_amount",
    "direction_col": "debit_credit_flag",
    "label_col": null,
    "base_time_col": null
  },
  "warnings": [
    "当前文件未发现 label 字段，不能单独用于监督学习建模。",
    "需要与主建模表按 account_id 或 customer_id 关联。"
  ],
  "need_human_review": true
}
```

---

### 8.3 HumanReviewGate_DataType

页面展示：

| 文件名 | 识别类型 | 置信度 | 推荐角色 | 推荐 Pipeline |
|---|---|---:|---|---|
| loan_application.csv | 普通结构化建模表 | 0.89 | 主表 | 结构化建模 |
| transaction_flow.csv | 交易流水表 | 0.93 | 流水表 | 流水特征 |

用户可操作：

```text
确认识别结果
修改文件角色
修改关键字段映射
进入人工配置
终止流程
```

---

## 9. 数据质量分析

### 9.1 DataQualityAgent

职责：

```text
在建模或特征生成前，对数据进行质量分析，不修改数据。
```

分析内容：

```text
重复行
主键重复
缺失值
高缺失字段
异常值
字段类型错误
常数列
唯一值过高字段
label 缺失
label 分布不均衡
时间字段解析问题
主表和流水表关联覆盖率
```

输出示例：

```json
{
  "duplicate_analysis": {
    "duplicate_rows": 120,
    "duplicate_rate": 0.012,
    "duplicate_key_rows": 86,
    "suggestion": "建议删除完全重复行，主键重复样本需人工确认。"
  },
  "missing_analysis": {
    "high_missing_columns": [
      {
        "column": "work_years",
        "missing_rate": 0.42,
        "suggestion": "缺失率较高，建议保留并增加缺失指示变量，或人工决定是否删除。"
      }
    ]
  },
  "outlier_analysis": {
    "outlier_columns": [
      {
        "column": "income",
        "method": "IQR",
        "outlier_rate": 0.035,
        "suggestion": "建议进行上下 1% 分位数缩尾。"
      }
    ]
  },
  "type_analysis": {
    "type_mismatch_columns": [
      {
        "column": "apply_date",
        "current_type": "object",
        "suggested_type": "datetime"
      }
    ]
  },
  "overall_quality_score": 78,
  "need_cleaning_review": true
}
```

---

### 9.2 TransactionQualityAgent

如果数据被识别为交易流水表，需要额外分析：

```text
transaction_id 是否重复
account_id 是否缺失
transaction_time 是否缺失
transaction_amount 是否异常
debit_credit_flag 是否可识别
负金额比例
交易时间范围
同一账户时间序列是否异常
与主表 account_id / customer_id 关联覆盖率
```

输出示例：

```json
{
  "transaction_quality": {
    "duplicate_transaction_id_rate": 0.003,
    "missing_account_id_rate": 0.001,
    "missing_txn_time_rate": 0.0,
    "negative_amount_rate": 0.12,
    "direction_coverage_rate": 0.98,
    "join_coverage_with_main_table": 0.86,
    "suggestion": "交易流水质量基本可用，但需确认负金额是否代表支出方向。"
  }
}
```

---

### 9.3 HumanReviewGate_DataQuality

用户根据质量分析结果选择处理方式。

重复数据：

```text
删除完全重复行
保留重复行
按主键保留最新一条
手动下载检查
```

缺失值：

```text
数值字段中位数填充
数值字段均值填充
类别字段填充 Unknown
增加缺失指示变量
删除高缺失字段
暂不处理
```

异常值：

```text
1%-99% 分位数缩尾
0.5%-99.5% 分位数缩尾
IQR 截断
保留异常值
手动选择字段处理
```

类型修正：

```text
接受自动类型转换
手动修改
跳过
```

---

## 10. 数据清洗设计

### 10.1 DataCleaningPlannerAgent

职责：

```text
根据数据质量报告生成可选清洗方案，但不直接执行。
```

输出示例：

```json
{
  "cleaning_plan": {
    "duplicate_strategy": {
      "action": "drop_exact_duplicate_rows",
      "need_human_confirm": true
    },
    "missing_strategy": {
      "numeric": "median_impute_with_missing_indicator",
      "categorical": "fill_unknown",
      "high_missing_columns": "manual_review"
    },
    "outlier_strategy": {
      "numeric": "winsorize_1_99",
      "manual_columns": ["income", "loan_amount"]
    },
    "type_fix_strategy": {
      "apply_date": "convert_to_datetime",
      "bad_flag": "convert_to_int"
    },
    "protected_columns": [
      "bad_flag",
      "customer_id",
      "loan_id"
    ]
  },
  "warnings": [
    "label 字段不会被填充或缩尾处理。",
    "原始文件不会被覆盖，清洗后数据将保存为 cleaned_dataset.csv。"
  ]
}
```

---

### 10.2 HumanReviewGate_CleaningPlan

用户确认后才执行清洗。

可选项：

```text
重复行：删除 / 保留 / 手动处理
主键重复：保留第一条 / 保留最新一条 / 不处理
数值缺失：中位数填充 / 均值填充 / 增加缺失指示 / 不处理
类别缺失：填充 Unknown / 众数填充 / 不处理
高缺失字段：删除 / 保留 / 手动选择
异常值：缩尾 / IQR 截断 / 保留
字段类型：自动转换 / 手动选择 / 不处理
```

---

### 10.3 DataCleaningExecutorTool

职责：

```text
根据用户确认后的清洗方案执行确定性清洗。
```

只允许执行白名单动作：

```text
删除完全重复行
按主键去重
缺失值填充
增加缺失指示变量
异常值缩尾
异常值截断
字段类型转换
删除高缺失字段
删除常数列
保留原始数据版本
```

输出示例：

```json
{
  "cleaned_data_path": "artifacts/project/cleaned_dataset.csv",
  "cleaning_log_path": "artifacts/project/cleaning_log.json",
  "before_shape": [10000, 52],
  "after_shape": [9880, 57],
  "actions_applied": [
    "drop_exact_duplicate_rows",
    "convert_apply_date_to_datetime",
    "median_impute_income",
    "add_missing_indicator_work_years",
    "winsorize_income_1_99"
  ]
}
```

---

## 11. 交易流水特征工程

### 11.1 TransactionFeatureAgent

职责：

```text
针对交易流水数据表，按照账户和交易日维度生成新特征，并基于观察点生成历史窗口特征。
```

交易流水表字段示例：

```text
transaction_id
account_id
customer_id
transaction_time
transaction_date
transaction_amount
transaction_type
debit_credit_flag
merchant_category
channel
balance_after_txn
counterparty_account
city
device_id
```

主表字段示例：

```text
customer_id
account_id
apply_date
loan_amount
age
income
bad_flag
```

关联约束：

```text
loan_application.account_id = transaction_flow.account_id
transaction_flow.transaction_time < loan_application.apply_date
```

---

### 11.2 账户日维度特征

粒度：

```text
account_id + transaction_date
```

生成特征：

| 特征 | 含义 |
|---|---|
| daily_txn_count | 当日交易笔数 |
| daily_txn_amt_sum | 当日交易总额 |
| daily_txn_amt_mean | 当日平均交易金额 |
| daily_txn_amt_max | 当日最大单笔金额 |
| daily_txn_amt_std | 当日交易金额标准差 |
| daily_income_sum | 当日收入金额 |
| daily_expense_sum | 当日支出金额 |
| daily_income_count | 当日收入笔数 |
| daily_expense_count | 当日支出笔数 |
| daily_net_inflow | 当日净流入 |
| daily_income_expense_ratio | 当日收入支出比 |
| daily_unique_counterparty_count | 当日不同交易对手数 |
| daily_unique_channel_count | 当日渠道数量 |
| daily_night_txn_count | 当日夜间交易笔数 |
| daily_large_txn_count | 当日大额交易笔数 |

---

### 11.3 观察点窗口特征

窗口：

```text
近 7 天
近 14 天
近 30 天
近 60 天
近 90 天
近 180 天
```

核心规则：

```text
只统计 transaction_time < apply_date 的交易。
```

特征组：

#### 活跃度类

```text
txn_count_7d
txn_count_30d
active_days_30d
active_days_ratio_30d
avg_daily_txn_count_30d
```

#### 金额规模类

```text
txn_amt_sum_30d
txn_amt_mean_30d
txn_amt_max_30d
txn_amt_std_30d
txn_amt_median_30d
```

#### 收入支出类

```text
income_sum_30d
income_count_30d
income_mean_30d
expense_sum_30d
expense_count_30d
expense_mean_30d
income_expense_ratio_30d
net_inflow_sum_30d
```

#### 波动性类

```text
txn_amt_cv_30d
income_cv_30d
expense_cv_30d
daily_net_inflow_std_30d
balance_std_30d
```

#### 行为偏好类

```text
night_txn_ratio_30d
weekend_txn_ratio_30d
online_channel_ratio_30d
atm_channel_ratio_30d
```

#### 大额交易类

```text
large_txn_count_30d
large_txn_amt_sum_30d
large_txn_ratio_30d
max_single_txn_amt_90d
```

#### 对手方分散度类

```text
unique_counterparty_count_30d
counterparty_concentration_30d
top1_counterparty_amt_ratio_30d
```

---

### 11.4 HumanReviewGate_Feature

流水特征生成前，用户确认特征方案。

选项：

```text
生成全部交易流水特征
只生成基础交易活跃度和金额特征
跳过交易流水特征
手动选择特征组
```

---

## 12. 风险字段检查

### 12.1 RiskGuardAgent

职责：

```text
检查疑似标签泄露字段、时间穿越字段、ID 类字段、高唯一值字段、敏感字段。
```

疑似泄露关键词：

```text
bad
overdue
default
dpd
逾期
违约
催收
还款结果
结清
核销
风险结果
collection
writeoff
settlement
repay_result
```

输出示例：

```json
{
  "can_continue": true,
  "high_risk_columns": ["dpd30", "collection_status"],
  "drop_recommendations": ["customer_id", "dpd30"],
  "warnings": [
    "字段 dpd30 可能包含贷后表现信息，申请评分模型中不建议使用。"
  ]
}
```

---

## 13. 建模、评估与策略分析

### 13.1 ModelingAgent

职责：

```text
调用 AutoGluon TabularPredictor 训练二分类模型。
```

核心配置：

```python
from autogluon.tabular import TabularPredictor

predictor = TabularPredictor(
    label=label_col,
    problem_type="binary",
    eval_metric="roc_auc",
    path=model_path
)

predictor.fit(
    train_data=train_df,
    time_limit=time_limit,
    presets="medium_quality"
)
```

---

### 13.2 EvaluationAgent

计算指标：

```text
AUC
KS
Accuracy
Precision
Recall
F1
混淆矩阵
分类报告
```

KS 函数：

```python
def compute_ks(y_true, y_score):
    data = pd.DataFrame({"y": y_true, "score": y_score})
    data = data.sort_values("score", ascending=False)
    data["bad"] = data["y"]
    data["good"] = 1 - data["y"]
    data["cum_bad_rate"] = data["bad"].cumsum() / data["bad"].sum()
    data["cum_good_rate"] = data["good"].cumsum() / data["good"].sum()
    return float((data["cum_bad_rate"] - data["cum_good_rate"]).abs().max())
```

---

### 13.3 StrategyAgent

规则：

```text
预测坏账概率 >= threshold → 拒绝
预测坏账概率 < threshold → 通过
```

输出阈值策略表：

| 阈值 | 通过率 | 拒绝率 | 通过客群坏账率 | 坏样本捕获率 |
|---:|---:|---:|---:|---:|
| 0.20 | 82% | 18% | 3.8% | 42% |
| 0.30 | 88% | 12% | 4.6% | 31% |
| 0.40 | 92% | 8% | 5.2% | 21% |

---

## 14. 人在回路控制节点

系统必须保留以下人工确认节点：

| 节点 | 作用 |
|---|---|
| HumanReviewGate_FieldSemantics | 确认字段语义解析结果 |
| HumanReviewGate_DataType | 确认数据类型、文件角色和 Pipeline |
| HumanReviewGate_DataQuality | 确认数据质量问题 |
| HumanReviewGate_CleaningPlan | 确认数据清洗方案 |
| HumanReviewGate_TableRole | 多表情况下确认主表、流水表和辅助表 |
| HumanReviewGate_Risk | 确认风险字段删除建议 |
| HumanReviewGate_Feature | 确认交易流水特征方案 |
| HumanReviewGate_Result | 确认模型评估结果和报告生成 |

所有人工确认结果保存到：

```text
artifacts/{project_name}/human_confirmations.json
```

记录字段：

```json
{
  "confirmation_type": "cleaning_plan",
  "timestamp": "2026-05-06 10:30:00",
  "user_decision": "accept_with_modification",
  "details": {
    "duplicate_strategy": "drop_exact_duplicate_rows",
    "missing_numeric": "median_with_indicator",
    "outlier_strategy": "winsorize_1_99"
  }
}
```

---

## 15. ReAct 推理行动控制

每个 Agent 必须遵循以下结构：

```text
Reasoning Summary
→ Action
→ Observation
→ Decision
→ Next Node
```

注意：

```text
不暴露完整链式思考，只展示简要 reasoning_summary。
Agent 不能执行任意代码。
Agent 只能调用权限白名单中的工具。
每次行动必须写入 agent_trace.json。
```

Agent trace 示例：

```json
{
  "agent_name": "DataQualityAgent",
  "reasoning_summary": "需要在建模前检查重复、缺失、异常和字段类型问题。",
  "action": "analyze_data_quality",
  "action_input_summary": {
    "file": "loan_application.csv",
    "label_col": "bad_flag"
  },
  "observation_summary": "发现120行完全重复，2个字段缺失率超过30%，3个数值字段存在异常值。",
  "decision": "进入数据质量人工确认节点，等待用户选择清洗策略。",
  "next_node": "HumanReviewGate_DataQuality",
  "status": "need_human_review",
  "timestamp": "2026-05-06 10:30:00"
}
```

---

## 16. 工具权限白名单

`core/permissions.py`：

```python
AGENT_TOOL_PERMISSIONS = {
    "DataIntakeAgent": [
        "save_uploaded_file",
        "inspect_file_format",
        "extract_file_metadata"
    ],
    "DataDictionaryParserAgent": [
        "parse_data_dictionary",
        "validate_dictionary_columns",
        "map_dictionary_to_dataset"
    ],
    "FieldSemanticParserAgent": [
        "parse_field_semantics_by_llm",
        "merge_dictionary_and_llm_semantics"
    ],
    "DataTypeClassifierAgent": [
        "classify_data_type_by_rules",
        "classify_data_type_by_llm",
        "merge_data_type_classification"
    ],
    "DataQualityAgent": [
        "analyze_duplicates",
        "analyze_missing_values",
        "analyze_outliers",
        "analyze_type_mismatch",
        "analyze_label_quality",
        "analyze_key_quality",
        "generate_data_quality_report"
    ],
    "TransactionQualityAgent": [
        "analyze_transaction_quality"
    ],
    "DataCleaningPlannerAgent": [
        "build_cleaning_plan"
    ],
    "DataCleaningExecutorTool": [
        "execute_cleaning_plan"
    ],
    "TransactionFeatureAgent": [
        "infer_transaction_schema",
        "build_account_daily_features",
        "build_account_window_features",
        "validate_transaction_feature_cutoff",
        "profile_generated_features"
    ],
    "RiskGuardAgent": [
        "detect_leakage_columns",
        "detect_id_columns",
        "detect_high_missing_columns",
        "detect_time_leakage_candidates"
    ],
    "ModelingAgent": [
        "train_autogluon_binary"
    ],
    "EvaluationAgent": [
        "evaluate_binary_model"
    ],
    "StrategyAgent": [
        "build_threshold_table"
    ],
    "ExplainAgent": [
        "compute_feature_importance"
    ],
    "ReportAgent": [
        "generate_markdown_report"
    ],
    "PipelineRouterAgent": [
        "route_pipeline"
    ]
}


def check_tool_permission(agent_name: str, tool_name: str) -> bool:
    return tool_name in AGENT_TOOL_PERMISSIONS.get(agent_name, [])
```

---

## 17. State 设计

`core/state.py`：

```python
from typing import TypedDict, Optional, List, Dict, Any

class UploadedFileState(TypedDict, total=False):
    file_id: str
    file_name: str
    file_path: str
    file_format: str
    n_rows: int
    n_cols: int
    columns: List[str]
    column_profiles: Dict[str, Any]
    sample_values_masked: Dict[str, Any]
    rule_detected_type: str
    llm_detected_type: str
    final_detected_type: str
    confidence: float
    assigned_role: str
    recommended_pipeline: str
    detected_keys: Dict[str, Any]
    warnings: List[str]


class RiskModelingProjectState(TypedDict, total=False):
    project_name: str

    uploaded_files: List[UploadedFileState]

    data_dictionary_path: Optional[str]
    dictionary_uploaded: bool
    dictionary_parse_result: Dict[str, Any]
    field_semantics: Dict[str, Any]

    data_type_classification_result: Dict[str, Any]
    pipeline_type: str

    data_quality_report: Dict[str, Any]
    transaction_quality_report: Optional[Dict[str, Any]]

    cleaning_plan: Dict[str, Any]
    cleaning_user_decision: Dict[str, Any]
    cleaned_data_paths: Dict[str, str]
    cleaning_log_path: str
    quality_recheck_report: Dict[str, Any]

    human_confirmations: List[Dict[str, Any]]

    main_data_path: Optional[str]
    cleaned_main_data_path: Optional[str]
    label_col: Optional[str]
    id_col: Optional[str]
    account_col: Optional[str]
    customer_col: Optional[str]
    time_col: Optional[str]
    positive_label: int

    transaction_data_path: Optional[str]
    cleaned_transaction_data_path: Optional[str]
    transaction_schema: Dict[str, Any]
    transaction_feature_plan: Dict[str, Any]
    transaction_daily_feature_path: Optional[str]
    transaction_window_feature_path: Optional[str]

    modeling_data_path: Optional[str]
    train_path: Optional[str]
    test_path: Optional[str]
    model_path: Optional[str]
    leaderboard_path: Optional[str]
    predictions_path: Optional[str]
    threshold_table_path: Optional[str]
    feature_importance_path: Optional[str]
    metrics: Dict[str, Any]
    report_path: Optional[str]

    agent_trace: List[Dict[str, Any]]
    warnings: List[str]
    errors: List[str]
    next_step: str
```

---

## 18. 推荐项目目录结构

```text
risk_modeling_agent_mvp/
├── app.py
├── requirements.txt
├── README.md
├── config.yaml
│
├── agents/
│   ├── __init__.py
│   ├── data_intake_agent.py
│   ├── data_dictionary_parser_agent.py
│   ├── field_semantic_parser_agent.py
│   ├── data_type_classifier_agent.py
│   ├── data_quality_agent.py
│   ├── transaction_quality_agent.py
│   ├── data_cleaning_planner_agent.py
│   ├── pipeline_router_agent.py
│   ├── task_understanding_agent.py
│   ├── table_relation_agent.py
│   ├── risk_guard_agent.py
│   ├── transaction_schema_agent.py
│   ├── transaction_feature_agent.py
│   ├── feature_merge_agent.py
│   ├── feature_planner_agent.py
│   ├── modeling_agent.py
│   ├── evaluation_agent.py
│   ├── strategy_agent.py
│   ├── explain_agent.py
│   └── report_agent.py
│
├── pipelines/
│   ├── __init__.py
│   ├── structured_modeling_pipeline.py
│   ├── transaction_feature_pipeline.py
│   ├── main_plus_transaction_pipeline.py
│   └── manual_configuration_pipeline.py
│
├── tools/
│   ├── __init__.py
│   ├── file_tools.py
│   ├── metadata_tools.py
│   ├── dictionary_tools.py
│   ├── field_semantic_tools.py
│   ├── data_type_tools.py
│   ├── data_quality_tools.py
│   ├── transaction_quality_tools.py
│   ├── data_cleaning_tools.py
│   ├── schema_tools.py
│   ├── relation_tools.py
│   ├── transaction_feature_tools.py
│   ├── split_tools.py
│   ├── modeling_tools.py
│   ├── metric_tools.py
│   ├── strategy_tools.py
│   ├── explain_tools.py
│   ├── report_tools.py
│   ├── mlflow_tools.py
│   └── trace_tools.py
│
├── core/
│   ├── __init__.py
│   ├── state.py
│   ├── schemas.py
│   ├── permissions.py
│   ├── constants.py
│   └── exceptions.py
│
├── artifacts/
│   └── .gitkeep
│
├── data/
│   └── .gitkeep
│
└── tests/
    ├── test_dictionary_parser.py
    ├── test_field_semantic_parser.py
    ├── test_data_type_rules.py
    ├── test_pipeline_router.py
    ├── test_metadata_extractor.py
    ├── test_data_quality.py
    ├── test_cleaning_plan.py
    ├── test_cleaning_executor.py
    ├── test_transaction_quality.py
    ├── test_transaction_daily_features.py
    ├── test_transaction_window_features.py
    ├── test_cutoff_validation.py
    ├── test_ks.py
    └── test_threshold.py
```

---

## 19. 输出文件规范

所有结果保存到：

```text
artifacts/{project_name}/
```

建议输出：

```text
raw_files/
metadata/
field_semantics.json
data_type_classification.json
human_confirmations.json
data_quality_report.json
cleaning_plan.json
cleaning_log.json
quality_recheck_report.json
cleaned_{file_name}.csv
transaction_daily_features.csv
transaction_window_features.csv
modeling_dataset.csv
train.csv
test.csv
model/
leaderboard.csv
metrics.json
threshold_table.csv
feature_importance.csv
agent_trace.json
model_report.md
```

---

## 20. Streamlit 页面设计

### 页面 1：项目与文件上传

字段：

```text
项目名称
上传一个或多个数据文件
上传数据字典，可选
是否启用大模型字段解析
LLM API Key / Base URL，可选
开始识别
```

---

### 页面 2：字段语义解析

展示：

```text
字段名
推断含义
字段角色
置信度
风险等级
是否来自数据字典
```

操作：

```text
确认字段解析
修改字段角色
重新上传数据字典
使用大模型重新解析
```

---

### 页面 3：数据类型识别

展示：

```text
文件名
识别类型
置信度
推荐角色
推荐 Pipeline
识别原因
```

操作：

```text
确认识别结果
修改文件角色
修改关键字段
进入人工配置
```

---

### 页面 4：数据质量分析

展示：

```text
重复情况
缺失情况
异常情况
类型问题
标签质量
主键质量
时间字段质量
交易流水质量
多表关联覆盖率
```

---

### 页面 5：清洗方案确认

用户选择：

```text
重复行处理策略
缺失值处理策略
异常值处理策略
字段类型转换策略
高缺失字段处理策略
常数列处理策略
主键重复处理策略
```

按钮：

```text
执行清洗并进入下一步
跳过清洗
终止流程
```

---

### 页面 6：Pipeline 路由确认

展示：

```text
系统建议进入哪个 Pipeline
原因
即将执行的步骤
```

---

### 页面 7：建模 / 特征生成过程

展示 Agent 执行过程：

```text
DataQualityAgent：已完成质量分析
DataCleaningPlannerAgent：已生成清洗方案
DataCleaningExecutorTool：已完成清洗
TransactionFeatureAgent：已生成窗口特征
ModelingAgent：已完成模型训练
EvaluationAgent：AUC=..., KS=...
ReportAgent：已生成报告
```

---

### 页面 8：结果展示

展示：

```text
模型排行榜
核心指标
阈值策略表
特征重要性
Agent 总结
报告下载
日志下载
```

---

## 21. requirements.txt

```txt
streamlit>=1.30.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
autogluon.tabular>=1.1.0
langgraph>=0.2.0
langchain>=0.2.0
langchain-openai>=0.1.0
mlflow>=2.10.0
pyyaml>=6.0.0
matplotlib>=3.7.0
joblib>=1.3.0
pydantic>=2.0.0
openpyxl>=3.1.0
pytest>=7.0.0
```

MVP 可以支持：

```text
MOCK_AGENT_MODE=True
```

在不调用真实大模型的情况下，用规则和模板跑完整流程。

---

## 22. Codex / Claude Code 开发提示词

请将下面内容直接复制给 Codex 或 Claude Code：

```text
你是一名资深 Python 机器学习工程师、风控建模专家和 Agent 系统架构师。请开发一个风控建模 Agent MVP 项目。

项目名称：risk_modeling_agent_mvp

目标：
开发一个本地可运行的 Streamlit 应用。用户上传一个或多个 CSV/Excel 数据文件，并可选上传数据字典后，系统通过多 Agent 工作流完成字段语义解析、数据类型识别、人在回路数据质量分析与清洗确认、Pipeline 路由、交易流水特征生成、AutoML 建模、评估、阈值策略分析、解释和报告生成。

核心原则：
1. 数据字典优先。如果用户上传数据字典，则优先使用数据字典解析字段语义。
2. 如果用户跳过数据字典，则使用大模型基于字段名、字段类型、缺失率、唯一值比例和脱敏样例值解析字段语义。
3. 不允许把全量原始数据传给 LLM。
4. Agent 只负责理解、规划、解释、判断和生成结构化建议。
5. Python Tool 负责真实的数据处理、清洗、特征生成、建模、评估和文件保存。
6. Agent 不允许执行任意代码。
7. Agent 只能调用权限白名单中的工具。
8. 所有关键节点必须有人在回路确认。
9. 原始数据只读，不覆盖。
10. 所有中间结果、清洗日志、人工确认和 Agent trace 必须保存到 artifacts/{project_name}/。

技术栈：
- Streamlit
- LangGraph
- pandas
- scikit-learn
- AutoGluon TabularPredictor
- 可选 MLflow
- Markdown 报告
- pytest

请按以下目录结构生成完整代码：

risk_modeling_agent_mvp/
├── app.py
├── requirements.txt
├── README.md
├── config.yaml
├── agents/
│   ├── __init__.py
│   ├── data_intake_agent.py
│   ├── data_dictionary_parser_agent.py
│   ├── field_semantic_parser_agent.py
│   ├── data_type_classifier_agent.py
│   ├── data_quality_agent.py
│   ├── transaction_quality_agent.py
│   ├── data_cleaning_planner_agent.py
│   ├── pipeline_router_agent.py
│   ├── task_understanding_agent.py
│   ├── table_relation_agent.py
│   ├── risk_guard_agent.py
│   ├── transaction_schema_agent.py
│   ├── transaction_feature_agent.py
│   ├── feature_merge_agent.py
│   ├── feature_planner_agent.py
│   ├── modeling_agent.py
│   ├── evaluation_agent.py
│   ├── strategy_agent.py
│   ├── explain_agent.py
│   └── report_agent.py
├── pipelines/
│   ├── __init__.py
│   ├── structured_modeling_pipeline.py
│   ├── transaction_feature_pipeline.py
│   ├── main_plus_transaction_pipeline.py
│   └── manual_configuration_pipeline.py
├── tools/
│   ├── __init__.py
│   ├── file_tools.py
│   ├── metadata_tools.py
│   ├── dictionary_tools.py
│   ├── field_semantic_tools.py
│   ├── data_type_tools.py
│   ├── data_quality_tools.py
│   ├── transaction_quality_tools.py
│   ├── data_cleaning_tools.py
│   ├── schema_tools.py
│   ├── relation_tools.py
│   ├── transaction_feature_tools.py
│   ├── split_tools.py
│   ├── modeling_tools.py
│   ├── metric_tools.py
│   ├── strategy_tools.py
│   ├── explain_tools.py
│   ├── report_tools.py
│   ├── mlflow_tools.py
│   └── trace_tools.py
├── core/
│   ├── __init__.py
│   ├── state.py
│   ├── schemas.py
│   ├── permissions.py
│   ├── constants.py
│   └── exceptions.py
├── artifacts/
│   └── .gitkeep
├── data/
│   └── .gitkeep
└── tests/
    ├── test_dictionary_parser.py
    ├── test_field_semantic_parser.py
    ├── test_data_type_rules.py
    ├── test_pipeline_router.py
    ├── test_metadata_extractor.py
    ├── test_data_quality.py
    ├── test_cleaning_plan.py
    ├── test_cleaning_executor.py
    ├── test_transaction_quality.py
    ├── test_transaction_daily_features.py
    ├── test_transaction_window_features.py
    ├── test_cutoff_validation.py
    ├── test_ks.py
    └── test_threshold.py

必须实现的流程：
Upload
→ Optional DataDictionaryUpload
→ DataIntakeAgent
→ MetadataExtractorTool
→ DataDictionaryParserAgent 或 FieldSemanticParserAgent
→ HumanReviewGate_FieldSemantics
→ DataTypeClassifierAgent
→ HumanReviewGate_DataType
→ DataQualityAgent / TransactionQualityAgent
→ HumanReviewGate_DataQuality
→ DataCleaningPlannerAgent
→ HumanReviewGate_CleaningPlan
→ DataCleaningExecutorTool
→ DataQualityRecheckAgent
→ PipelineRouterAgent
→ StructuredModelingPipeline / TransactionFeaturePipeline / MainPlusTransactionPipeline / ManualConfigurationPipeline

必须实现的数据字典能力：
- 支持上传 CSV 或 Excel 数据字典。
- 数据字典字段包括 table_name、column_name、column_cn_name、business_meaning、data_type、value_range、enum_mapping、is_label、is_id、is_time、is_sensitive、available_time、source_system、remark。
- 如果数据字典存在，优先使用数据字典解析字段语义。
- 如果数据字典不存在，使用大模型或 MOCK_AGENT_MODE 规则模板解析字段语义。

必须实现的数据类型识别能力：
- structured_modeling_table
- transaction_flow_table
- main_table
- auxiliary_table
- unknown_or_ambiguous
- 规则识别和大模型识别结果不一致或置信度低于 0.8 时，进入人工配置。

必须实现的数据质量能力：
- 重复行分析
- 主键重复分析
- 缺失值分析
- 高缺失字段分析
- 异常值分析
- 字段类型错误分析
- 常数列分析
- 唯一值过高字段分析
- label 缺失分析
- label 分布不均衡分析
- 时间字段解析问题分析
- 主表和流水表关联覆盖率分析

必须实现的数据清洗能力：
- 清洗方案由 DataCleaningPlannerAgent 生成，但不能直接执行。
- 用户必须在 HumanReviewGate_CleaningPlan 确认后，DataCleaningExecutorTool 才能执行。
- 支持删除完全重复行、按主键去重、缺失填充、缺失指示变量、异常值缩尾、IQR 截断、字段类型转换、删除高缺失字段、删除常数列。
- 原始数据只读，清洗后保存为 cleaned_{file_name}.csv。
- 保存 cleaning_log.json 和 quality_recheck_report.json。

必须实现的交易流水特征能力：
- 识别 transaction_id、account_id、customer_id、transaction_time、transaction_amount、debit_credit_flag、transaction_type、channel、merchant_category、balance_after_txn、counterparty_account。
- 按 account_id + transaction_date 生成账户日维度特征。
- 以主表 apply_date 作为观察点，生成近 7、14、30、60、90、180 天窗口特征。
- 必须保证 transaction_time < apply_date，防止时间穿越。
- 生成 transaction_daily_features.csv 和 transaction_window_features.csv。

必须实现的建模能力：
- 使用 AutoGluon TabularPredictor 进行二分类建模。
- 计算 AUC、KS、Accuracy、Precision、Recall、F1、混淆矩阵。
- 生成阈值策略表，阈值从 0.05 到 0.95，步长 0.05。
- score >= threshold 表示拒绝，score < threshold 表示通过。
- 输出 feature_importance.csv、leaderboard.csv、metrics.json、threshold_table.csv、model_report.md。

人在回路要求：
- HumanReviewGate_FieldSemantics
- HumanReviewGate_DataType
- HumanReviewGate_DataQuality
- HumanReviewGate_CleaningPlan
- HumanReviewGate_TableRole
- HumanReviewGate_Risk
- HumanReviewGate_Feature
- HumanReviewGate_Result
- 所有人工选择写入 human_confirmations.json。

ReAct 控制要求：
每个 Agent 必须遵循：
reasoning_summary → action → observation → decision → next_node。
不展示完整链式思考，只保存和展示简要 reasoning_summary。
每个 Agent 只能调用 core/permissions.py 白名单中的工具。
每次 Agent 行动写入 agent_trace.json。

请保证：
1. 项目可以通过 pip install -r requirements.txt 安装依赖。
2. 项目可以通过 streamlit run app.py 启动。
3. MOCK_AGENT_MODE=True 时，不调用真实大模型也能跑完整流程。
4. README 中写清楚安装、运行和使用方式。
5. 测试用例覆盖数据字典解析、字段语义解析、数据类型识别、数据质量、清洗计划、清洗执行、流水特征、KS、阈值表。
```

---

## 23. MVP 验收标准

### 23.1 功能验收

```text
可以上传一个或多个数据文件
可以上传或跳过数据字典
可以解析字段语义
可以识别普通结构化表和交易流水表
可以人工确认字段角色和数据类型
可以进行数据质量分析
可以选择清洗策略
可以执行清洗并保存日志
可以生成交易流水账户日特征
可以生成交易流水窗口特征
可以合并主表和流水特征
可以训练二分类模型
可以输出 AUC、KS 和阈值策略表
可以生成 Markdown 报告
```

### 23.2 工程验收

```text
streamlit run app.py 可以启动
所有原始数据只读
所有中间文件保存到 artifacts/{project_name}/
agent_trace.json 完整记录 Agent 行动
human_confirmations.json 完整记录人工确认
cleaning_log.json 完整记录清洗动作
核心函数有 pytest 测试
MOCK_AGENT_MODE=True 可跑通完整流程
```

### 23.3 风控业务验收

报告必须回答：

```text
这份数据是什么类型？
字段语义是否明确？
是否存在数据字典？
数据质量有什么问题？
用户选择了哪些清洗策略？
是否存在标签泄露或时间穿越风险？
交易流水是否按观察点正确截断？
生成了哪些交易流水特征？
模型效果如何？
阈值策略下通过率、拒绝率和坏账率如何？
当前模型是否适合进一步验证？
下一轮优化建议是什么？
```

---

## 24. 后续 V1/V2 升级方向

### V1

```text
WOE / IV
自动分箱
评分卡模型
OOT 验证
PSI / CSI
MLflow 实验管理
FastAPI 推理接口
Docker 部署
```

### V2

```text
Agent 自动提出特征优化建议
Agent 自动提出清洗策略对比实验
Champion-Challenger 模型机制
模型监控与漂移预警
多租户和权限管理
模型审批流
特征库 Feast
任务调度 Airflow / Prefect
```

---

## 25. 参考技术文档

- LangGraph 官方文档：https://docs.langchain.com/oss/python/langgraph/overview
- AutoGluon TabularPredictor 官方文档：https://auto.gluon.ai/stable/api/autogluon.tabular.TabularPredictor.html
- Streamlit 官方文档：https://docs.streamlit.io/

