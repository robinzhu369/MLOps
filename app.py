"""风控建模 Agent MVP — Streamlit 主入口."""

import streamlit as st

st.set_page_config(
    page_title="风控建模 Agent MVP",
    page_icon="🏦",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------

def _init_state():
    defaults = {
        "page": 1,
        "project_name": "",
        "pipeline_app": None,
        "thread_id": None,
        "pipeline_state": {},
        "uploaded_data_files": [],
        "uploaded_dict_file": None,
        "use_llm": False,
        "llm_api_key": "",
        "llm_base_url": "",
        "llm_model": "gpt-4o",
        "field_semantics_edits": {},
        "data_type_edits": {},
        "cleaning_decisions": {},
        "pipeline_confirmed": False,
        "execution_log": [],
        "run_complete": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_state()

# ---------------------------------------------------------------------------
# Page routing
# ---------------------------------------------------------------------------

from ui.page1_upload import render as page1
from ui.page2_field_semantics import render as page2
from ui.page3_data_type import render as page3
from ui.page4_data_quality import render as page4
from ui.page5_cleaning import render as page5
from ui.page6_pipeline_route import render as page6
from ui.page7_execution import render as page7
from ui.page8_results import render as page8

PAGES = {
    1: ("📁 上传", page1),
    2: ("🔍 字段语义", page2),
    3: ("🗂️ 数据类型", page3),
    4: ("📊 质量分析", page4),
    5: ("🧹 清洗方案", page5),
    6: ("🔀 Pipeline 路由", page6),
    7: ("⚙️ 执行过程", page7),
    8: ("📈 结果", page8),
}

# Sidebar navigation
with st.sidebar:
    st.title("🏦 风控建模 Agent")
    st.markdown("---")
    current = st.session_state.page
    for num, (label, _) in PAGES.items():
        is_current = num == current
        prefix = "▶ " if is_current else "  "
        st.markdown(f"{prefix}**{num}. {label}**" if is_current else f"{num}. {label}")
    st.markdown("---")
    if st.session_state.project_name:
        st.caption(f"项目：{st.session_state.project_name}")

# Render current page
_, render_fn = PAGES[st.session_state.page]
render_fn()
