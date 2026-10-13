import streamlit as st
import pandas as pd
from agent import ask_agent

st.set_page_config(page_title="Data Analysis Agent", page_icon="📊",
                   layout="wide")

st.title("📊 Data Analysis Agent")
st.write(
    "Upload a CSV and ask a question. The agent selects an analysis tool "
    "and computes results from your data."
)

@st.cache_data
def sample_data():
    return pd.DataFrame({
        "region": ["North", "South", "East", "West", "North", "South",
                   "East", "West", "North", "South", "East", "West"],
        "product": ["A", "B", "A", "C", "B", "A", "C", "B", "A", "C", "B", "A"],
        "sales": [1200, 950, 1430, 780, 1100, 1580, 890, 1320, 1700, 990, 1450, 1820],
        "units": [12, 10, 14, 8, 11, 16, 9, 13, 17, 10, 15, 18],
        "customer_satisfaction": [4.2, 3.8, 4.5, 3.6, 4.0, 4.7, 3.9, 4.1, 4.8, 4.0, 4.3, 4.9],
    })

uploaded = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded:
    try:
        data = pd.read_csv(uploaded)
        if data.empty or len(data.columns) == 0:
            st.error("The uploaded CSV is empty.")
            st.stop()
    except Exception as exc:
        st.error(f"Could not read CSV: {exc}")
        st.stop()
else:
    data = sample_data()
    st.info("Using the built-in sample dataset. Upload a CSV to analyse your own data.")

st.subheader("Dataset preview")
st.caption(f"{len(data)} rows · {len(data.columns)} columns")
st.dataframe(data.head(10), use_container_width=True)
st.write("**Available columns:**", ", ".join(map(str, data.columns)))

st.subheader("Ask a question")
question = st.text_input(
    "Question",
    placeholder="e.g. Show statistics for sales",
    key="question",
)

if st.button("Analyse", type="primary") and question.strip():
    tool, result = ask_agent(question, data)
    st.markdown(f"**Tool selected:** `{tool}`")
    st.markdown("**Computed result**")
    st.json(result, expanded=True)

st.divider()
st.caption(
    "This version uses transparent rule-based tool selection, not an LLM. "
    "Results depend on supported question patterns. Correlation does not imply causation."
)
