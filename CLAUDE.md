# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Risk Modeling Agent MVP (风控建模 Agent MVP) — an end-to-end system that takes uploaded data files through field parsing, data quality analysis, cleaning, feature engineering, AutoML modeling, and evaluation report generation. Designed for credit risk / fraud scenarios.

## Planned Tech Stack

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

## Architecture

The system uses a multi-agent pipeline orchestrated by LangGraph:

```
Streamlit UI → LangGraph Orchestrator → Agents → Tools → Storage (artifacts/)
```

Key agents: DataIntakeAgent, FieldSemanticParserAgent, DataTypeClassifierAgent, DataQualityAgent, DataCleaningPlannerAgent, PipelineRouterAgent, TransactionFeatureAgent, RiskGuardAgent, ModelingAgent, EvaluationAgent, StrategyAgent, ReportAgent.

Each agent can only invoke whitelisted tools. Agents reason and plan; tools execute actual data operations.

## Core Design Principles

- **Human-in-the-loop (HITL):** User confirmation required at critical decision points (field semantics, data type classification, cleaning plan, modeling config).
- **Original data is read-only.** All transformations produce new files; never overwrite source data.
- **Full traceability:** All agent actions → `agent_trace.json`, human decisions → `human_confirmations.json`, cleaning ops → `cleaning_log.json`.
- **No arbitrary code execution by agents.** Tool whitelist enforced.
- **Time-leakage prevention:** Transaction features must use only data before the observation point (`transaction_time < apply_date`).

## Pipeline Routing

The system routes to different pipelines based on detected data type:
- **Scenario A:** Structured modeling table (has label) → full AutoML pipeline
- **Scenario B:** Transaction flow table (no label) → feature engineering + profile export only
- **Scenario C:** Main table + transaction flow → join features back to main table, then AutoML

## Storage Layout

```
artifacts/{project_name}/
  data/          # uploaded and cleaned data
  features/      # generated feature tables
  models/        # trained models
  reports/       # evaluation reports
```

## Development Notes

- Technical specification: `docs/risk_modeling_agent.md` (authoritative design doc, in Chinese)
- The project is in pre-implementation phase — no source code exists yet
- When implementing, use Python with the stack listed above
- Tests should use pytest
- Language in code/comments: English. Documentation may be in Chinese.
