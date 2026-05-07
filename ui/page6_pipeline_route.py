"""Page 6: Pipeline 路由确认."""

import streamlit as st

_PIPELINE_STEPS = {
    "structured_modeling_pipeline": [
        "DataIntake", "FieldSemantic", "DataType", "DataQuality",
        "CleaningPlan", "CleaningExecute", "RiskGuard",
        "Modeling", "Evaluation", "Strategy", "Explain", "Report",
    ],
    "transaction_feature_pipeline": [
        "DataIntake", "FieldSemantic", "DataType", "TransactionQuality",
        "CleaningPlan", "CleaningExecute", "TransactionFeature", "Report",
    ],
    "main_plus_transaction_pipeline": [
        "DataIntake", "FieldSemantic", "DataType", "DataQuality",
        "CleaningPlan", "CleaningExecute", "TransactionFeature",
        "FeatureMerge", "TimeLeakageCheck", "RiskGuard",
        "Modeling", "Evaluation", "Strategy", "Explain", "Report",
    ],
    "manual_configuration_pipeline": ["需要人工配置"],
}

_PIPELINE_NAMES = {
    "structured_modeling_pipeline": "场景 A：结构化建模 Pipeline",
    "transaction_feature_pipeline": "场景 B：交易流水特征 Pipeline",
    "main_plus_transaction_pipeline": "场景 C：主表 + 流水表联合建模 Pipeline",
    "manual_configuration_pipeline": "人工配置 Pipeline",
}


def _determine_pipeline():
    import agents.pipeline_router_agent as router

    state = st.session_state.pipeline_state
    with st.spinner("正在确定 Pipeline 路由…"):
        result = router.run(state)
        state["pipeline_route"] = result
        state["pipeline_type"] = result.get("pipeline_type", "")
        st.session_state.pipeline_state = state


def render():
    st.header("🔀 页面 6：Pipeline 路由确认")

    state = st.session_state.pipeline_state
    if not state.get("data_type_classification_result"):
        st.warning("请先完成数据类型识别（页面 3）。")
        if st.button("← 返回数据类型"):
            st.session_state.page = 3
            st.rerun()
        return

    if not state.get("pipeline_route"):
        _determine_pipeline()
        st.rerun()

    route = state.get("pipeline_route", {})
    pipeline_type = route.get("pipeline_type", "")
    reasoning = route.get("reasoning", "")
    file_roles = route.get("file_roles", {})

    pipeline_name = _PIPELINE_NAMES.get(pipeline_type, pipeline_type)
    steps = _PIPELINE_STEPS.get(pipeline_type, [])

    st.success(f"**系统建议：** {pipeline_name}")
    if reasoning:
        st.info(f"**原因：** {reasoning}")

    st.markdown("---")

    # File roles
    if file_roles:
        st.subheader("文件角色")
        import pandas as pd
        rows = []
        for fname, info in file_roles.items():
            rows.append({
                "文件名": fname,
                "识别类型": info.get("type", ""),
                "角色": info.get("role", ""),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # Pipeline steps
    st.subheader("即将执行的步骤")
    step_cols = st.columns(min(len(steps), 6))
    for i, step in enumerate(steps):
        with step_cols[i % len(step_cols)]:
            st.markdown(f"**{i+1}.** {step}")

    st.markdown("---")

    # Allow manual override
    with st.expander("手动修改 Pipeline（可选）"):
        from core.constants import PipelineType
        pipeline_options = [p.value for p in PipelineType]
        pipeline_labels = [_PIPELINE_NAMES.get(p, p) for p in pipeline_options]
        current_idx = pipeline_options.index(pipeline_type) if pipeline_type in pipeline_options else 0
        selected_label = st.selectbox(
            "选择 Pipeline",
            options=pipeline_labels,
            index=current_idx,
        )
        selected_pipeline = pipeline_options[pipeline_labels.index(selected_label)]
        if selected_pipeline != pipeline_type:
            if st.button("应用修改"):
                state["pipeline_type"] = selected_pipeline
                state["pipeline_route"]["pipeline_type"] = selected_pipeline
                st.session_state.pipeline_state = state
                st.rerun()

    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ 确认并开始执行", type="primary"):
            st.session_state.pipeline_confirmed = True
            st.session_state.page = 7
            st.rerun()
    with col2:
        if st.button("← 返回清洗方案"):
            st.session_state.page = 5
            st.rerun()
