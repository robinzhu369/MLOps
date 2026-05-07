"""Page 5: 清洗方案确认."""

import pandas as pd
import streamlit as st


_STRATEGY_LABELS = {
    "drop_exact_duplicates": "删除完全重复行",
    "drop_key_duplicates": "删除主键重复行",
    "drop_high_missing_columns": "删除高缺失列",
    "add_missing_indicator": "添加缺失指示变量",
    "fill_missing_values": "填充缺失值",
    "winsorize_outliers": "异常值缩尾处理",
    "convert_types": "字段类型转换",
    "drop_constant_columns": "删除常数列",
}


def _run_cleaning_plan():
    import agents.data_cleaning_planner_agent as agent

    state = st.session_state.pipeline_state
    with st.spinner("正在生成清洗方案…"):
        plan = agent.run(state)
        state["cleaning_plan"] = plan
        st.session_state.pipeline_state = state
        # Initialise decisions: all steps enabled by default
        decisions = {}
        for step in plan.get("cleaning_steps", []):
            decisions[step["action"]] = True
        st.session_state.cleaning_decisions = decisions


def _run_cleaning_execute():
    import tools.data_cleaning_tools as cleaning

    state = st.session_state.pipeline_state
    plan = state.get("cleaning_plan", {})
    decisions = st.session_state.cleaning_decisions

    # Filter steps by user decisions
    active_steps = [
        s for s in plan.get("cleaning_steps", [])
        if decisions.get(s["action"], True)
    ]
    plan_to_execute = {**plan, "cleaning_steps": active_steps}
    state["cleaning_plan"] = plan_to_execute
    state["cleaning_user_decision"] = decisions

    with st.spinner("正在执行数据清洗…"):
        data_path = (
            state.get("main_data_path")
            or (state.get("uploaded_files") or [{}])[0].get("file_path", "")
        )
        result = cleaning.execute_cleaning_plan(
            data_path=data_path,
            cleaning_plan=plan_to_execute,
            project_name=state.get("project_name", "default"),
        )
        state["cleaned_data_path"] = result.get("output_path")
        state["cleaning_log_path"] = result.get("cleaning_log_path", "")
        st.session_state.pipeline_state = state


def render():
    st.header("🧹 页面 5：清洗方案确认")

    state = st.session_state.pipeline_state
    if not state.get("data_quality_report"):
        st.warning("请先完成数据质量分析（页面 4）。")
        if st.button("← 返回质量分析"):
            st.session_state.page = 4
            st.rerun()
        return

    if not state.get("cleaning_plan"):
        _run_cleaning_plan()
        st.rerun()

    plan = state.get("cleaning_plan", {})
    steps = plan.get("cleaning_steps", [])

    if plan.get("error"):
        st.error(f"清洗方案生成失败：{plan['error']}")
        return

    st.info(f"系统建议 **{len(steps)}** 个清洗步骤。请选择要执行的步骤，然后点击执行。")

    protected = plan.get("protected_columns", [])
    if protected:
        st.caption(f"受保护列（不会被修改）：{', '.join(protected)}")

    st.markdown("---")

    decisions = st.session_state.cleaning_decisions

    if steps:
        st.subheader("清洗步骤选择")
        for step in steps:
            action = step["action"]
            label = _STRATEGY_LABELS.get(action, action)
            reason = step.get("reason", "")
            col1, col2 = st.columns([1, 4])
            with col1:
                enabled = st.checkbox(
                    label,
                    value=decisions.get(action, True),
                    key=f"step_{action}",
                )
                decisions[action] = enabled
            with col2:
                st.caption(reason)
        st.session_state.cleaning_decisions = decisions
    else:
        st.success("数据质量良好，无需清洗步骤。")

    st.markdown("---")
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("✅ 执行清洗并进入下一步", type="primary"):
            _run_cleaning_execute()
            st.success("清洗完成！")
            st.session_state.page = 6
            st.rerun()

    with col2:
        if st.button("⏭️ 跳过清洗"):
            state["cleaning_user_decision"] = {"skipped": True}
            # Use original data path as cleaned path
            data_path = state.get("main_data_path") or (
                state.get("uploaded_files", [{}])[0].get("file_path")
            )
            state["cleaned_data_path"] = data_path
            st.session_state.pipeline_state = state
            st.session_state.page = 6
            st.rerun()

    with col3:
        if st.button("← 返回质量分析"):
            st.session_state.page = 4
            st.rerun()
