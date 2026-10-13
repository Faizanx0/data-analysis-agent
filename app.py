
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Data Analysis Agent",
    page_icon="📊",
    layout="wide"
)

# ---------- HEADER ----------
st.title("📊 Data Analysis Agent")
st.caption("Explore your data. Discover patterns. Get insights.")

uploaded_file = st.file_uploader(
    "Upload your dataset",
    type=["csv"]
)

if uploaded_file is None:
    st.info("Upload a CSV file to begin your analysis.")
    st.stop()

try:
    data = pd.read_csv(uploaded_file)
except Exception as error:
    st.error(f"Unable to read this file: {error}")
    st.stop()

if data.empty:
    st.warning("This CSV file has no data to display.")
    st.stop()

numeric_cols = data.select_dtypes(
    include="number"
).columns.tolist()

categorical_cols = data.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()

# ---------- SUMMARY CARDS ----------
st.subheader("Dataset Overview")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Records", f"{len(data):,}")
c2.metric("Total Features", len(data.columns))
c3.metric("Missing Values", f"{int(data.isna().sum().sum()):,}")
c4.metric("Duplicate Rows", f"{int(data.duplicated().sum()):,}")

# ---------- DATA PREVIEW ----------
with st.expander("View dataset preview"):
    st.dataframe(data.head(10), use_container_width=True)

# ---------- VISUAL ANALYSIS ----------
st.subheader("Visual Analytics")

tab1, tab2, tab3 = st.tabs([
    "Numeric Distribution",
    "Category Breakdown",
    "Correlation"
])

with tab1:
    if numeric_cols:
        selected_num = st.selectbox(
            "Choose a numeric feature",
            numeric_cols
        )

        values = pd.to_numeric(
            data[selected_num], errors="coerce"
        ).dropna()

        if not values.empty:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.hist(values, bins=20, edgecolor="white")
            ax.set_title(f"Distribution of {selected_num}")
            ax.set_xlabel(selected_num)
            ax.set_ylabel("Frequency")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            st.caption(
                f"Average: {values.mean():.2f} · "
                f"Median: {values.median():.2f} · "
                f"Minimum: {values.min():.2f} · "
                f"Maximum: {values.max():.2f}"
            )
        else:
            st.info("This feature has no valid numeric values.")
    else:
        st.info("No numeric features are available in this dataset.")

with tab2:
    if categorical_cols:
        selected_cat = st.selectbox(
            "Choose a category",
            categorical_cols
        )

        top_n = st.slider(
            "Number of categories to display",
            min_value=3,
            max_value=10,
            value=5
        )

        counts = (
            data[selected_cat]
            .fillna("Missing")
            .astype(str)
            .value_counts()
            .head(top_n)
        )

        chart_choice = st.radio(
            "Chart type",
            ["Pie Chart", "Bar Chart"],
            horizontal=True
        )

        if chart_choice == "Pie Chart":
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.pie(
                counts.values,
                labels=counts.index,
                autopct="%1.1f%%",
                startangle=90
            )
            ax.set_title(f"Breakdown of {selected_cat}")
            ax.axis("equal")
        else:
            fig, ax = plt.subplots(figsize=(8, 4))
            counts.sort_values().plot(
                kind="barh", ax=ax
            )
            ax.set_title(f"Most common values in {selected_cat}")
            ax.set_xlabel("Count")
            ax.set_ylabel("")
        
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.caption(
            "Charts show the most frequent categories selected above."
        )
    else:
        st.info(
            "No categorical features are available for a pie chart."
        )

with tab3:
    if len(numeric_cols) >= 2:
        corr = data[numeric_cols].corr()

        fig, ax = plt.subplots(figsize=(8, 5))
        image = ax.imshow(
            corr, aspect="auto", vmin=-1, vmax=1
        )
        ax.set_xticks(range(len(numeric_cols)))
        ax.set_yticks(range(len(numeric_cols)))
        ax.set_xticklabels(
            numeric_cols, rotation=45, ha="right"
        )
        ax.set_yticklabels(numeric_cols)
        ax.set_title("Numeric Feature Correlations")
        fig.colorbar(image, ax=ax, label="Correlation")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.info(
            "At least two numeric features are needed for correlation."
        )

# ---------- AUTOMATIC SUMMARY ----------
st.subheader("Quick Insights")

if numeric_cols:
    summary_col = st.selectbox(
        "Summarize a numeric feature",
        numeric_cols,
        key="summary_column"
    )
    values = pd.to_numeric(
        data[summary_col], errors="coerce"
    ).dropna()

    if not values.empty:
        st.write(
            f"**{summary_col}:** average "
            f"{values.mean():.2f}, median "
            f"{values.median():.2f}, based on "
            f"{len(values):,} valid records."
        )

if categorical_cols:
    summary_cat = st.selectbox(
        "Most common category",
        categorical_cols,
        key="summary_category"
    )
    common = data[summary_cat].value_counts(
        dropna=False
    ).head(1)

    if not common.empty:
        st.write(
            f"**{summary_cat}:** most frequent value is "
            f"`{common.index[0]}` "
            f"({common.iloc[0]:,} records)."
        )

# ---------- ASK YOUR DATA ----------
st.subheader("Ask Your Data")

st.caption(
    "Try questions about statistics, missing values, "
    "common categories, or correlations."
)

question = st.text_input(
    "Your question",
    placeholder="e.g. What are the statistics for Age?"
)

if st.button("Analyze", type="primary") and question.strip():
    q = question.lower().replace("_", " ")

    def find_column(columns):
        for col in columns:
            if col.lower().replace("_", " ") in q:
                return col
        return None

    if "missing" in q or "null values" in q:
        report = pd.DataFrame({
            "Missing Count": data.isna().sum(),
            "Missing Percentage": (
                data.isna().mean() * 100
            ).round(2)
        }).sort_values("Missing Count", ascending=False)
        st.dataframe(report, use_container_width=True)

    elif "correlation" in q:
        col_a = find_column(numeric_cols)
        col_b = None

        if col_a:
            col_b = find_column(
                [c for c in numeric_cols if c != col_a]
            )

        if col_a and col_b:
            result = data[[col_a, col_b]].corr().iloc[0, 1]
            st.write(
                f"Correlation between **{col_a}** and "
                f"**{col_b}**: **{result:.3f}**"
            )
        else:
            st.info(
                "Mention two numeric feature names in your question."
            )

    elif any(word in q for word in [
        "statistics", "average", "mean", "median",
        "minimum", "maximum", "describe"
    ]):
        col = find_column(numeric_cols)

        if col:
            st.write(f"Statistics for **{col}**")
            st.dataframe(
                data[col].describe().to_frame().round(2),
                use_container_width=True
            )
        else:
            st.info(
                "Please include a numeric feature name, such as Age."
            )

    elif any(word in q for word in [
        "most common", "common values", "frequency"
    ]):
        col = find_column(categorical_cols)

        if col:
            st.dataframe(
                data[col].value_counts(dropna=False)
                .head(10).to_frame("Count"),
                use_container_width=True
            )
        else:
            st.info(
                "Please include a categorical feature name."
            )

    elif any(word in q for word in [
        "rows", "columns", "overview"
    ]):
        st.write(
            f"This dataset contains {len(data):,} rows and "
            f"{len(data.columns)} columns."
        )

    else:
        st.info(
            "Try a supported question about statistics, "
            "missing values, common values, or correlation."
        )

# ---------- DOWNLOADS ----------
st.subheader("Export Results")

left, right = st.columns(2)

with left:
    st.download_button(
        "Download CSV",
        data=data.to_csv(index=False).encode("utf-8"),
        file_name="analyzed_data.csv",
        mime="text/csv"
    )

with right:
    st.download_button(
        "Download JSON",
        data=data.to_json(
            orient="records", indent=2
        ).encode("utf-8"),
        file_name="analyzed_data.json",
        mime="application/json"
    )

st.divider()
st.caption("Data Analysis Agent · Built with Python and Streamlit")
