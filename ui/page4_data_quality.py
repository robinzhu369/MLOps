"""Page 4: 数据质量分析结果展示."""

import pandas as pd
import streamlit as st


def _run_quality():
    import agents.data_quality_agent as agent

    state = st.session_state.pipeline_state
    with st.spinner("正在进行数据质量分析…"):
        report = agent.run(state)
        state["data_quality_report"] = report
        st.session_state.pipeline_state = state


def _metric_card(label: str, value, delta=None):
    st.metric(label=label, value=value, delta=delta)


def render():
    st.header("📊 页面 4：数据质量分析")

    state = st.session_state.pipeline_state
    if not state.get("data_type_classification_result"):
        st.warning("请先完成数据类型识别（页面 3）。")
        if st.button("← 返回数据类型"):
            st.session_state.page = 3
            st.rerun()
        return

    if not state.get("data_quality_report"):
        _run_quality()
        st.rerun()

    report = state.get("data_quality_report", {})

    if report.get("error"):
        st.error(f"质量分析出错：{report['error']}")
        return

    # ---- Overview metrics ----
    st.subheader("总览")
    n_rows = report.get("n_rows", 0)
    n_cols = report.get("n_cols", 0)
    dup = report.get("duplicate_analysis", {})
    missing = report.get("missing_analysis", {})

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("总行数", f"{n_rows:,}")
    with col2:
        st.metric("总列数", n_cols)
    with col3:
        dup_rows = dup.get("duplicate_rows", 0)
        st.metric("重复行", dup_rows, delta=f"-{dup_rows}" if dup_rows else None,
                  delta_color="inverse")
    with col4:
        missing_cols = len(missing.get("high_missing_columns", []))
        st.metric("高缺失列", missing_cols, delta=f"-{missing_cols}" if missing_cols else None,
                  delta_color="inverse")

    st.markdown("---")

    # ---- Tabs for each dimension ----
    tabs = st.tabs([
        "重复情况", "缺失情况", "异常情况", "类型问题",
        "标签质量", "主键质量", "时间字段质量",
    ])

    # Tab 0: Duplicates
    with tabs[0]:
        st.subheader("重复情况")
        if dup:
            st.json(dup)
        else:
            st.info("无重复分析数据。")

    # Tab 1: Missing
    with tabs[1]:
        st.subheader("缺失情况")
        cols_info = missing.get("columns", [])
        if cols_info:
            df_miss = pd.DataFrame(cols_info)
            if "missing_rate" in df_miss.columns:
                df_miss["missing_rate"] = df_miss["missing_rate"].apply(
                    lambda x: f"{x:.1%}"
                )
            st.dataframe(df_miss, use_container_width=True, hide_index=True)
            high_miss = missing.get("high_missing_columns", [])
            if high_miss:
                st.warning(f"高缺失列（>30%）：{', '.join(high_miss)}")
        else:
            st.success("无缺失值问题。")

    # Tab 2: Outliers
    with tabs[2]:
        st.subheader("异常情况")
        outlier = report.get("outlier_analysis", {})
        if outlier:
            cols_out = outlier.get("columns", [])
            if cols_out:
                df_out = pd.DataFrame(cols_out)
                st.dataframe(df_out, use_container_width=True, hide_index=True)
            else:
                st.success("未检测到异常值。")
        else:
            st.info("无异常值分析数据。")

    # Tab 3: Type issues
    with tabs[3]:
        st.subheader("类型问题")
        type_issues = report.get("type_mismatch_analysis", {})
        if type_issues:
            issues = type_issues.get("issues", [])
            if issues:
                st.dataframe(pd.DataFrame(issues), use_container_width=True, hide_index=True)
            else:
                st.success("无类型问题。")
        else:
            st.info("无类型分析数据。")

    # Tab 4: Label quality
    with tabs[4]:
        st.subheader("标签质量")
        label_q = report.get("label_quality", {})
        if label_q:
            st.json(label_q)
        else:
            st.info("无标签质量数据（可能无 label 列）。")

    # Tab 5: Key quality
    with tabs[5]:
        st.subheader("主键质量")
        key_q = report.get("key_quality", {})
        if key_q:
            st.json(key_q)
        else:
            st.info("无主键质量数据。")

    # Tab 6: Time quality
    with tabs[6]:
        st.subheader("时间字段质量")
        time_q = report.get("time_quality", {})
        if time_q:
            st.json(time_q)
        else:
            st.info("无时间字段质量数据。")

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ 进入清洗方案确认", type="primary"):
            st.session_state.page = 5
            st.rerun()
    with col2:
        if st.button("← 返回数据类型"):
            st.session_state.page = 3
            st.rerun()
