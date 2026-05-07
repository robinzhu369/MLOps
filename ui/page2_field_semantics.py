"""Page 2: 字段语义解析结果展示与确认."""

import pandas as pd
import streamlit as st


def _build_llm_call():
    """Build an LLM call function from session state config."""
    if not st.session_state.use_llm:
        return None
    try:
        from openai import OpenAI
        client = OpenAI(
            api_key=st.session_state.llm_api_key or "sk-placeholder",
            base_url=st.session_state.llm_base_url or None,
        )
        model = st.session_state.llm_model or "gpt-4o"

        def llm_call(prompt: str) -> str:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )
            return resp.choices[0].message.content

        return llm_call
    except Exception:
        return None


def _run_field_semantic():
    import agents.field_semantic_parser_agent as agent
    import agents.data_dictionary_parser_agent as dict_agent

    state = st.session_state.pipeline_state
    llm_call = _build_llm_call()

    with st.spinner("正在解析字段语义…"):
        if state.get("dictionary_uploaded") and state.get("data_dictionary_path"):
            dict_result = dict_agent.run(state)
            state.update(dict_result)
            result = agent.run_with_dictionary_merge(state, dict_result, llm_call=llm_call)
        else:
            result = agent.run(state, llm_call=llm_call)

        state.update(result)
        st.session_state.pipeline_state = state
        # Initialise edits from parsed result
        st.session_state.field_semantics_edits = {
            k: v.copy() for k, v in result.get("field_semantics", {}).items()
        }


def render():
    st.header("🔍 页面 2：字段语义解析")

    state = st.session_state.pipeline_state
    if not state.get("uploaded_files"):
        st.warning("请先完成文件上传（页面 1）。")
        if st.button("← 返回上传"):
            st.session_state.page = 1
            st.rerun()
        return

    # Run agent if not yet done
    if not state.get("field_semantics"):
        _run_field_semantic()
        st.rerun()

    field_semantics = st.session_state.field_semantics_edits or state.get("field_semantics", {})

    if not field_semantics:
        st.error("字段语义解析失败，请检查上传文件格式。")
        return

    st.info(f"共解析 **{len(field_semantics)}** 个字段。置信度低于 0.8 或高风险字段需人工确认。")

    # Build editable dataframe
    rows = []
    for key, info in field_semantics.items():
        rows.append({
            "字段键": key,
            "字段名": info.get("field_name", key.split(":")[-1]),
            "推断含义": info.get("inferred_meaning", ""),
            "字段角色": info.get("field_role", "normal"),
            "置信度": round(float(info.get("confidence", 0)), 2),
            "风险等级": info.get("risk_level", "normal"),
            "来源": "数据字典" if info.get("from_dictionary") else "LLM/规则",
        })

    df = pd.DataFrame(rows)

    # Highlight low-confidence / high-risk rows
    def _highlight(row):
        if row["风险等级"] == "high":
            return ["background-color: #ffe0e0"] * len(row)
        if row["置信度"] < 0.8:
            return ["background-color: #fff8e0"] * len(row)
        return [""] * len(row)

    st.dataframe(
        df.style.apply(_highlight, axis=1),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("---")
    st.subheader("修改字段角色")

    from core.constants import FieldRole
    role_options = [r.value for r in FieldRole]

    with st.expander("展开编辑字段角色（可选）"):
        edits = st.session_state.field_semantics_edits
        changed = False
        for key in list(edits.keys()):
            info = edits[key]
            current_role = info.get("field_role", "normal")
            new_role = st.selectbox(
                f"{key}",
                options=role_options,
                index=role_options.index(current_role) if current_role in role_options else 0,
                key=f"role_{key}",
            )
            if new_role != current_role:
                edits[key]["field_role"] = new_role
                changed = True
        if changed:
            st.session_state.field_semantics_edits = edits

    st.markdown("---")
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("✅ 确认字段解析，进入下一步", type="primary"):
            # Merge edits back into state
            st.session_state.pipeline_state["field_semantics"] = (
                st.session_state.field_semantics_edits
            )
            st.session_state.page = 3
            st.rerun()

    with col2:
        if st.button("🔄 重新解析"):
            st.session_state.pipeline_state["field_semantics"] = {}
            st.session_state.field_semantics_edits = {}
            st.rerun()

    with col3:
        if st.button("← 返回上传"):
            st.session_state.page = 1
            st.rerun()
