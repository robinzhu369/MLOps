中文 | [English](README.md)

# 风控建模 Agent MVP

端到端多智能体系统，自动化信贷风控与反欺诈建模流程——从原始数据接入、特征工程、AutoML 训练到评估报告生成。

## 功能特性

- **自动管道路由** — 自动识别输入数据类型（有标签建模表、交易流水表或两者兼有），并路由至对应流程
- **多智能体协作** — 基于 LangGraph ReAct 模式，由多个 Agent 分别负责字段解析、数据质量分析、清洗、特征工程、建模和报告生成
- **人机协同 (HITL)** — 在关键决策点（字段语义确认、清洗方案、建模配置）需用户确认
- **全链路可追溯** — 所有 Agent 动作、人工决策和数据变换均有日志记录
- **时间泄漏防护** — 交易特征严格使用观察点之前的数据

## 系统架构

```
Streamlit UI → LangGraph 编排器 → Agents → Tools → 存储 (artifacts/)
```

Agent 负责推理和规划；Tool 执行实际数据操作。每个 Agent 只能调用白名单内的工具。

### 管道场景

| 场景 | 输入 | 输出 |
|------|------|------|
| A | 有标签的结构化建模表 | 完整 AutoML 流程 → 模型 + 评估报告 |
| B | 交易流水表（无标签） | 特征工程 → 画像导出 |
| C | 主表 + 交易流水 | 特征关联回主表 → AutoML 流程 |

## 技术栈

| 层级 | 技术 |
|------|------|
| 前端 | Streamlit |
| Agent 编排 | LangGraph (ReAct 模式) |
| LLM 接口 | OpenAI 兼容 API |
| 建模 | AutoGluon TabularPredictor |
| 评估指标 | scikit-learn |
| 数据处理 | pandas / numpy |
| 实验追踪 | 本地 JSON + 可选 MLflow |
| 报告 | Markdown |
| 测试 | pytest |

## 项目结构

```
├── app.py                  # Streamlit 入口
├── config.yaml             # 项目配置
├── agents/                 # LangGraph Agent 定义
│   ├── data_intake_agent.py
│   ├── field_semantic_parser_agent.py
│   ├── data_type_classifier_agent.py
│   ├── data_quality_agent.py
│   ├── data_cleaning_planner_agent.py
│   ├── pipeline_router_agent.py
│   ├── transaction_feature_agent.py
│   ├── risk_guard_agent.py
│   ├── modeling_agent.py
│   ├── evaluation_agent.py
│   ├── strategy_agent.py
│   ├── explain_agent.py
│   └── report_agent.py
├── tools/                  # Agent 调用的工具实现
├── core/                   # 共享状态、Schema、常量、异常
├── tests/                  # pytest 测试套件
├── artifacts/              # 生成产物（模型、报告、特征）
├── data/                   # 上传的源数据（只读）
└── docs/                   # 技术方案文档
```

## 快速开始

### 环境要求

- Python 3.10+
- OpenAI 兼容 API Key

### 安装依赖

```bash
pip install -r requirements.txt
```

### 配置

编辑 `config.yaml` 设置 LLM 端点和建模参数：

```yaml
llm:
  api_base: "https://api.openai.com/v1"
  model: "gpt-4o"
  temperature: 0.0
```

### 启动

```bash
streamlit run app.py
```

### 运行测试

```bash
pytest
```

## 设计原则

1. **原始数据只读** — 所有变换生成新文件，不覆盖源数据
2. **禁止任意代码执行** — Agent 通过工具白名单操作
3. **完整审计链** — `agent_trace.json`、`human_confirmations.json`、`cleaning_log.json`
4. **时间泄漏防护** — 交易特征仅使用观察点之前的数据

## 文档

- 技术方案（中文）：[`docs/risk_modeling_agent.md`](docs/risk_modeling_agent.md)
- 任务拆解：[`docs/task_breakdown.md`](docs/task_breakdown.md)

## 许可证

私有项目，保留所有权利。
