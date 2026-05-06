[中文版](README_zh.md) | English

# Risk Modeling Agent MVP (风控建模 Agent MVP)

An end-to-end multi-agent system that automates credit risk and fraud modeling workflows — from raw data ingestion through feature engineering, AutoML training, and evaluation report generation.

## Features

- **Automated pipeline routing** — detects whether input is a labeled modeling table, transaction flow, or both, and routes accordingly
- **Multi-agent orchestration** — LangGraph-based ReAct agents handle field parsing, data quality analysis, cleaning, feature engineering, modeling, and reporting
- **Human-in-the-loop** — user confirmation at critical decision points (field semantics, cleaning plan, modeling config)
- **Full traceability** — all agent actions, human decisions, and data transformations are logged
- **Time-leakage prevention** — transaction features enforce temporal ordering relative to observation dates

## Architecture

```
Streamlit UI → LangGraph Orchestrator → Agents → Tools → Storage (artifacts/)
```

Agents reason and plan; tools execute actual data operations. Each agent can only invoke whitelisted tools.

### Pipeline Scenarios

| Scenario | Input | Output |
|----------|-------|--------|
| A | Structured table with label | Full AutoML pipeline → model + evaluation report |
| B | Transaction flow (no label) | Feature engineering → profile export |
| C | Main table + transaction flow | Join features → AutoML pipeline |

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | Streamlit |
| Agent Orchestration | LangGraph (ReAct pattern) |
| LLM Interface | OpenAI-compatible API |
| Modeling | AutoGluon TabularPredictor |
| Metrics | scikit-learn |
| Data Processing | pandas / numpy |
| Experiment Tracking | Local JSON + optional MLflow |
| Reports | Markdown |
| Testing | pytest |

## Project Structure

```
├── app.py                  # Streamlit entry point
├── config.yaml             # Project configuration
├── agents/                 # LangGraph agent definitions
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
├── tools/                  # Tool implementations invoked by agents
├── core/                   # Shared state, schemas, constants, exceptions
├── tests/                  # pytest test suite
├── artifacts/              # Generated outputs (models, reports, features)
├── data/                   # Uploaded source data (read-only)
└── docs/                   # Technical specification
```

## Getting Started

### Prerequisites

- Python 3.10+
- An OpenAI-compatible API key

### Installation

```bash
pip install -r requirements.txt
```

### Configuration

Edit `config.yaml` to set your LLM endpoint and modeling parameters:

```yaml
llm:
  api_base: "https://api.openai.com/v1"
  model: "gpt-4o"
  temperature: 0.0
```

### Running

```bash
streamlit run app.py
```

### Testing

```bash
pytest
```

## Design Principles

1. **Original data is read-only** — all transformations produce new files
2. **No arbitrary code execution** — agents operate through a tool whitelist
3. **Full audit trail** — `agent_trace.json`, `human_confirmations.json`, `cleaning_log.json`
4. **Time-leakage prevention** — transaction features use only pre-observation data

## Documentation

- Technical specification (中文): [`docs/risk_modeling_agent.md`](docs/risk_modeling_agent.md)
- Task breakdown: [`docs/task_breakdown.md`](docs/task_breakdown.md)

## License

Private — all rights reserved.
