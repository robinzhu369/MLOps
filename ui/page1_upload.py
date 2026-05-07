"""Page 1: 项目与文件上传."""

import os
import tempfile

import streamlit as st


def _nav_next():
    st.session_state.page = 2


def render():
    st.header("📁 页面 1：项目与文件上传")

    col_left, col_right = st.columns([2, 1])

    with col_left:
        # Project name
        project_name = st.text_input(
            "项目名称 *",
            value=st.session_state.project_name,
            placeholder="例如：loan_risk_2024",
        )
        st.session_state.project_name = project_name.strip()

        # Data files
        data_files = st.file_uploader(
            "上传数据文件（支持 CSV / Excel，可多选）*",
            type=["csv", "xlsx", "xls"],
            accept_multiple_files=True,
            key="uploader_data",
        )
        if data_files:
            st.session_state.uploaded_data_files = data_files

        # Data dictionary (optional)
        dict_file = st.file_uploader(
            "上传数据字典（可选，CSV / Excel）",
            type=["csv", "xlsx", "xls"],
            key="uploader_dict",
        )
        if dict_file:
            st.session_state.uploaded_dict_file = dict_file

    with col_right:
        st.subheader("LLM 配置")
        use_llm = st.checkbox(
            "启用大模型字段解析",
            value=st.session_state.use_llm,
        )
        st.session_state.use_llm = use_llm

        if use_llm:
            st.session_state.llm_api_key = st.text_input(
                "API Key",
                value=st.session_state.llm_api_key,
                type="password",
            )
            st.session_state.llm_base_url = st.text_input(
                "Base URL（可选）",
                value=st.session_state.llm_base_url,
                placeholder="https://api.openai.com/v1",
            )
            st.session_state.llm_model = st.text_input(
                "模型名称",
                value=st.session_state.llm_model,
            )

    st.markdown("---")

    # Validation and start
    ready = bool(
        st.session_state.project_name
        and st.session_state.uploaded_data_files
    )

    if not ready:
        st.info("请填写项目名称并上传至少一个数据文件。")

    if st.button("🚀 开始识别", disabled=not ready, type="primary"):
        _run_intake()


def _run_intake():
    """Save uploaded files to disk and run DataIntakeAgent."""
    import agents.data_intake_agent as intake_agent

    project_name = st.session_state.project_name
    data_files = st.session_state.uploaded_data_files

    with st.spinner("正在处理上传文件…"):
        uploaded_states = []
        saved_paths = []

        for uf in data_files:
            # Write to a temp file so agents can read it
            suffix = os.path.splitext(uf.name)[1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uf.getvalue())
                tmp_path = tmp.name

            saved_paths.append(tmp_path)

            fake_state = {"project_name": project_name}
            uf.seek(0)
            file_state = intake_agent.run(fake_state, uf, uf.name)
            uploaded_states.append(file_state)

        # Handle optional data dictionary
        dict_path = None
        if st.session_state.uploaded_dict_file:
            df = st.session_state.uploaded_dict_file
            suffix = os.path.splitext(df.name)[1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(df.getvalue())
                dict_path = tmp.name

        st.session_state.pipeline_state = {
            "project_name": project_name,
            "uploaded_files": uploaded_states,
            "_pending_files": saved_paths,
            "data_dictionary_path": dict_path,
            "dictionary_uploaded": dict_path is not None,
            "errors": [],
            "warnings": [],
            "agent_trace": [],
        }

    st.success(f"已上传 {len(uploaded_states)} 个文件，正在进入字段语义解析…")
    st.session_state.page = 2
    st.rerun()
