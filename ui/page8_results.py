"""Page 8: 结果展示."""

import os

import streamlit as st


def _fmt(v):
    try:
        return f"{float(v):.4f}"
    except Exception:
        return str(v)


def _render_report_download(state: dict, project_name: str):
    st.subheader("报告下载")
    report_path = state.get("report_path")
    if report_path and os.path.exists(report_path):
        with open(report_path, "r", encoding="utf-8") as f:
            content = f.read()
        st.markdown(content[:3000] + ("…（截断）" if len(content) > 3000 else ""))
        st.download_button(
            "📥 下载完整报告（Markdown）",
            data=content,
            file_name=os.path.basename(report_path),
            mime="text/markdown",
        )
    else:
        # Try to find report in artifacts
        artifact_dir = os.path.join("artifacts", project_name, "reports")
        if os.path.isdir(artifact_dir):
            md_files = [f for f in os.listdir(artifact_dir) if f.endswith(".md")]
            if md_files:
                path = os.path.join(artifact_dir, md_files[0])
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read()
                st.download_button(
                    "📥 下载报告（Markdown）",
                    data=content,
                    file_name=md_files[0],
                    mime="text/markdown",
                )
                return
        st.info("暂无报告文件。")


def _render_log_download(project_name: str):
    st.subheader("日志下载")
    import json

    artifact_dir = os.path.join("artifacts", project_name)
    log_files = {
        "Agent Trace": "agent_trace.json",
        "人工确认记录": "human_confirmations.json",
        "清洗日志": "cleaning_log.json",
    }
    for label, fname in log_files.items():
        path = os.path.join(artifact_dir, fname)
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            st.download_button(
                f"📥 下载 {label}",
                data=content,
                file_name=fname,
                mime="application/json",
                key=f"dl_{fname}",
            )
        else:
            st.caption(f"{label}：暂无数据")


def render():
    st.header("📈 页面 8：结果展示")

    if not st.session_state.run_complete:
        st.warning("请先完成 Pipeline 执行（页面 7）。")
        if st.button("← 返回执行过程"):
            st.session_state.page = 7
            st.rerun()
        return

    state = st.session_state.pipeline_state
    project_name = state.get("project_name", "")
    pipeline_type = state.get("pipeline_type", "")
    is_modeling = "transaction_feature" not in pipeline_type

    if is_modeling:
        tabs = st.tabs(["核心指标", "模型排行榜", "阈值策略", "特征重要性", "报告下载", "日志下载"])

        # Tab 0: Core metrics
        with tabs[0]:
            st.subheader("核心评估指标")
            metrics = state.get("metrics", {})
            if metrics:
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("AUC", _fmt(metrics.get("auc", "N/A")))
                with col2:
                    st.metric("KS", _fmt(metrics.get("ks", "N/A")))
                with col3:
                    st.metric("Gini", _fmt(metrics.get("gini", "N/A")))
                with col4:
                    st.metric("Accuracy", _fmt(metrics.get("accuracy", "N/A")))

                extra_keys = [k for k in metrics if k not in ("auc", "ks", "gini", "accuracy")]
                if extra_keys:
                    import pandas as pd
                    extra_df = pd.DataFrame(
                        [{"指标": k, "值": _fmt(metrics[k])} for k in extra_keys]
                    )
                    st.dataframe(extra_df, use_container_width=True, hide_index=True)
            else:
                st.info("暂无评估指标数据。")

        # Tab 1: Leaderboard
        with tabs[1]:
            st.subheader("模型排行榜")
            leaderboard_path = state.get("leaderboard_path")
            if leaderboard_path and os.path.exists(leaderboard_path):
                import pandas as pd
                lb = pd.read_csv(leaderboard_path)
                st.dataframe(lb, use_container_width=True, hide_index=True)
            else:
                st.info("暂无排行榜数据。")

        # Tab 2: Threshold strategy
        with tabs[2]:
            st.subheader("阈值策略表")
            threshold_table = state.get("threshold_table")
            threshold_path = state.get("threshold_table_path")
            if threshold_table:
                import pandas as pd
                st.dataframe(pd.DataFrame(threshold_table), use_container_width=True, hide_index=True)
            elif threshold_path and os.path.exists(threshold_path):
                import pandas as pd
                st.dataframe(pd.read_csv(threshold_path), use_container_width=True, hide_index=True)
            else:
                st.info("暂无阈值策略数据。")

        # Tab 3: Feature importance
        with tabs[3]:
            st.subheader("特征重要性")
            fi = state.get("feature_importance")
            fi_path = state.get("feature_importance_path")
            if fi:
                import pandas as pd
                df_fi = pd.DataFrame(fi)
                if "importance" in df_fi.columns and "feature" in df_fi.columns:
                    df_fi = df_fi.sort_values("importance", ascending=False).head(30)
                    try:
                        import matplotlib.pyplot as plt
                        fig, ax = plt.subplots(figsize=(8, max(4, len(df_fi) * 0.3)))
                        ax.barh(df_fi["feature"], df_fi["importance"])
                        ax.invert_yaxis()
                        ax.set_xlabel("Importance")
                        ax.set_title("Top Feature Importance")
                        st.pyplot(fig)
                        plt.close(fig)
                    except Exception:
                        st.dataframe(df_fi, use_container_width=True, hide_index=True)
                else:
                    st.dataframe(df_fi, use_container_width=True, hide_index=True)
            elif fi_path and os.path.exists(fi_path):
                import pandas as pd
                st.dataframe(pd.read_csv(fi_path), use_container_width=True, hide_index=True)
            else:
                st.info("暂无特征重要性数据。")

        # Tab 4: Report download
        with tabs[4]:
            _render_report_download(state, project_name)

        # Tab 5: Log download
        with tabs[5]:
            _render_log_download(project_name)

    else:
        # Transaction feature pipeline
        tabs = st.tabs(["特征文件", "报告下载", "日志下载"])

        with tabs[0]:
            st.subheader("生成的特征文件")
            daily_path = state.get("transaction_daily_feature_path")
            window_path = state.get("transaction_window_feature_path")
            for label, path in [("日维度特征", daily_path), ("窗口特征", window_path)]:
                if path and os.path.exists(path):
                    import pandas as pd
                    st.markdown(f"**{label}**：`{path}`")
                    df = pd.read_csv(path)
                    st.dataframe(df.head(20), use_container_width=True, hide_index=True)
                    with open(path, "rb") as f:
                        st.download_button(
                            f"📥 下载 {label}",
                            data=f,
                            file_name=os.path.basename(path),
                            mime="text/csv",
                            key=f"dl_{label}",
                        )
                else:
                    st.info(f"{label}：暂无数据。")

        with tabs[1]:
            _render_report_download(state, project_name)

        with tabs[2]:
            _render_log_download(project_name)

    st.markdown("---")
    if st.button("← 返回执行过程"):
        st.session_state.page = 7
        st.rerun()
