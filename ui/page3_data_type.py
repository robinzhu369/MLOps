"""Page 3: 数据类型识别结果展示与确认."""

import pandas as pd
import streamlit as st


def _run_data_type():
    import agents.data_type_classifier_agent as agent
    from ui.page2_field_semantics import _build_llm_call

    state = st.session_state.pipeline_state
    llm_call = _build_llm_call()

    with st.spinner("正在识别数据类型…"):
        result = agent.run(state, llm_call=llm_call)
        state.update(result)
        st.session_state.pipeline_state = state
        # Initialise edits
        st.session_state.data_type_edits = {
            c["file_name"]: c.copy()
            for c in result.get("classifications", [])
        }


def render():
    st.header("🗂️ 页面 3：数据类型识别")

    state = st.session_state.pipeline_state
    if not state.get("field_semantics"):
        st.warning("请先完成字段语义解析（页面 2）。")
        if st.button("← 返回字段语义"):
            st.session_state.page = 2
            st.rerun()
        return

    # Run agent if not yet done
    if not state.get("data_type_classification_result"):
        _run_data_type()
        st.rerun()

    result = state.get("data_type_classification_result", {})
    classifications = result.get("classifications", [])
    edits = st.session_state.data_type_edits

    if not classifications:
        st.error("数据类型识别失败，请检查上传文件。")
        return

    # Overall pipeline recommendation
    recommended_pipeline = result.get("recommended_pipeline", "")
    pipeline_reason = result.get("pipeline_reason", "")
    if recommended_pipeline:
        st.success(f"**推荐 Pipeline：** {recommended_pipeline}")
        if pipeline_reason:
            st.caption(pipeline_reason)

    st.markdown("---")

    from core.constants import DataType
    dtype_options = [d.value for d in DataType]

    rows = []
    for c in classifications:
        rows.append({
            "文件名": c.get("file_name", ""),
            "识别类型": c.get("detected_data_type", ""),
            "置信度": round(float(c.get("confidence", 0)), 2),
            "推荐角色": c.get("assigned_role", ""),
            "推荐 Pipeline": c.get("recommended_pipeline", ""),
            "识别原因": c.get("reason", ""),
        })

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("修改文件角色与关键字段")

    with st.expander("展开编辑（可选）"):
        for file_name, info in edits.items():
            st.markdown(f"**{file_name}**")
            col1, col2 = st.columns(2)
            with col1:
                current_type = info.get("detected_data_type", dtype_options[0])
                new_type = st.selectbox(
                    "数据类型",
                    options=dtype_options,
                    index=dtype_options.index(current_type) if current_type in dtype_options else 0,
                    key=f"dtype_{file_name}",
                )
                edits[file_name]["detected_data_type"] = new_type
            with col2:
                label_col = st.text_input(
                    "Label 列名（如有）",
                    value=info.get("label_col", ""),
                    key=f"label_{file_name}",
                )
                edits[file_name]["label_col"] = label_col
            st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("✅ 确认识别结果，进入下一步", type="primary"):
            # Merge edits back
            updated_classifications = list(edits.values())
            st.session_state.pipeline_state["data_type_classification_result"]["classifications"] = (
                updated_classifications
            )
            # Set label_col from edits if provided
            for info in updated_classifications:
                if info.get("label_col"):
                    st.session_state.pipeline_state["label_col"] = info["label_col"]
            st.session_state.page = 4
            st.rerun()

    with col2:
        if st.button("🔄 重新识别"):
            st.session_state.pipeline_state["data_type_classification_result"] = {}
            st.session_state.data_type_edits = {}
            st.rerun()

    with col3:
        if st.button("← 返回字段语义"):
            st.session_state.page = 2
            st.rerun()
