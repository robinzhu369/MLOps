"""Page 7: Agent 执行过程展示."""

import time

import streamlit as st

_PIPELINE_MODULE = {
    "structured_modeling_pipeline": "pipelines.structured_modeling_pipeline",
    "transaction_feature_pipeline": "pipelines.transaction_feature_pipeline",
    "main_plus_transaction_pipeline": "pipelines.main_plus_transaction_pipeline",
}

_AGENT_LABELS = {
    "DataIntake": "DataIntakeAgent：文件上传与元数据提取",
    "FieldSemantic": "FieldSemanticParserAgent：字段语义解析",
    "DataType": "DataTypeClassifierAgent：数据类型识别",
    "DataQuality": "DataQualityAgent：数据质量分析",
    "TransactionQuality": "TransactionQualityAgent：流水质量分析",
    "CleaningPlan": "DataCleaningPlannerAgent：清洗方案生成",
    "CleaningExecute": "DataCleaningExecutor：执行数据清洗",
    "RiskGuard": "RiskGuardAgent：风险字段检查",
    "TransactionFeature": "TransactionFeatureAgent：交易特征生成",
    "FeatureMerge": "FeatureMerge：特征合并",
    "TimeLeakageCheck": "TimeLeakageCheck：时间穿越校验",
    "Modeling": "ModelingAgent：AutoML 建模",
    "Evaluation": "EvaluationAgent：模型评估",
    "Strategy": "StrategyAgent：阈值策略分析",
    "Explain": "ExplainAgent：特征重要性解释",
    "Report": "ReportAgent：报告生成",
}


def _run_pipeline():
    """Run the selected pipeline and stream progress to session state."""
    import importlib

    state = st.session_state.pipeline_state
    pipeline_type = state.get("pipeline_type", "structured_modeling_pipeline")
    module_name = _PIPELINE_MODULE.get(pipeline_type, _PIPELINE_MODULE["structured_modeling_pipeline"])

    mod = importlib.import_module(module_name)
    app = mod.compile_pipeline()

    thread_config = {"configurable": {"thread_id": st.session_state.project_name or "default"}}

    # Build initial input — reuse already-computed state from earlier pages
    initial_input = {k: v for k, v in state.items() if not k.startswith("_")}

    execution_log = []
    final_state = None

    try:
        for event in app.stream(initial_input, config=thread_config, stream_mode="updates"):
            for node_name, node_output in event.items():
                label = _AGENT_LABELS.get(node_name, node_name)
                entry = {"node": node_name, "label": label, "status": "completed", "output": node_output}
                execution_log.append(entry)
                st.session_state.execution_log = execution_log

            # Capture latest state snapshot
            snapshot = app.get_state(thread_config)
            if snapshot and snapshot.values:
                final_state = snapshot.values

    except Exception as exc:
        execution_log.append({
            "node": "ERROR",
            "label": f"执行出错：{exc}",
            "status": "error",
            "output": {},
        })
        st.session_state.execution_log = execution_log

    if final_state:
        st.session_state.pipeline_state.update(final_state)

    st.session_state.run_complete = True


def render():
    st.header("⚙️ 页面 7：建模 / 特征生成过程")

    if not st.session_state.pipeline_confirmed:
        st.warning("请先确认 Pipeline 路由（页面 6）。")
        if st.button("← 返回 Pipeline 路由"):
            st.session_state.page = 6
            st.rerun()
        return

    state = st.session_state.pipeline_state
    pipeline_type = state.get("pipeline_type", "")
    st.caption(f"当前 Pipeline：{pipeline_type}")

    # Start button
    if not st.session_state.run_complete and not st.session_state.execution_log:
        if st.button("▶ 开始执行", type="primary"):
            with st.spinner("Pipeline 执行中，请稍候…"):
                _run_pipeline()
            st.rerun()
        return

    # Show execution log
    log = st.session_state.execution_log
    if log:
        st.subheader("执行日志")
        for entry in log:
            status = entry.get("status", "")
            label = entry.get("label", entry.get("node", ""))
            if status == "error":
                st.error(f"❌ {label}")
            else:
                st.success(f"✅ {label}")

    # Also show agent trace from disk
    project_name = state.get("project_name", "")
    if project_name:
        try:
            from tools.trace_tools import read_agent_trace
            trace = read_agent_trace(project_name)
            if trace:
                with st.expander(f"Agent Trace（共 {len(trace)} 条）"):
                    import pandas as pd
                    df = pd.DataFrame(trace)
                    cols = [c for c in ["timestamp", "agent_name", "action", "status", "decision"] if c in df.columns]
                    st.dataframe(df[cols], use_container_width=True, hide_index=True)
        except Exception:
            pass

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.session_state.run_complete:
            if st.button("📈 查看结果", type="primary"):
                st.session_state.page = 8
                st.rerun()
    with col2:
        if st.button("🔄 重新执行"):
            st.session_state.execution_log = []
            st.session_state.run_complete = False
            st.rerun()
