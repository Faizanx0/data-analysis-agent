
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Data Analysis Agent",
    page_icon="📊",
    layout="wide"
)

# ---------- LAYOUT SETTINGS ----------
st.title("📊 Data Analysis Agent")
st.caption("Explore your data. Discover patterns. Get insights.")

with st.sidebar:
    st.subheader("Display Settings")
    chart_size = st.select_slider(
        "Chart size",
        options=["Compact", "Medium", "Large"],
        value="Compact"
    )

    chart_width = {
        "Compact": 4,
        "Medium": 5.5,
        "Large": 7
    }[chart_size]

    chart_height = {
        "Compact": 2.5,
        "Medium": 3.5,
        "Large": 4.5
    }[chart_size]

    st.caption("Change the size of charts shown below.")

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

# ---------- DATASET OVERVIEW ----------
st.subheader("Dataset Overview")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Records", f"{len(data):,}")
c2.metric("Total Features", len(data.columns))
c3.metric("Missing Values", f"{int(data.isna().sum().sum()):,}")
c4.metric("Duplicate Rows", f"{int(data.duplicated().sum()):,}")

with st.expander("View dataset preview"):
    st.dataframe(data.head(10), use_container_width=True, height=260)

# ---------- VISUAL ANALYTICS ----------
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
            fig, ax = plt.subplots(
                figsize=(chart_width, chart_height)
            )
            ax.hist(values, bins=20, edgecolor="white")
            ax.set_title(f"Distribution of {selected_num}")
            ax.set_xlabel(selected_num)
            ax.set_ylabel("Frequency")
            fig.tight_layout()
            st.pyplot(fig, use_container_width=False)
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
            fig, ax = plt.subplots(
                figsize=(chart_width, chart_height)
            )
            ax.pie(
                counts.values,
                labels=counts.index,
                autopct="%1.1f%%",
                startangle=90,
                textprops={"fontsize": 8}
            )
            ax.set_title(f"Breakdown of {selected_cat}")
            ax.axis("equal")
        else:
            fig, ax = plt.subplots(
                figsize=(chart_width, chart_height)
            )
            counts.sort_values().plot(kind="barh", ax=ax)
            ax.set_title(f"Most common values in {selected_cat}")
            ax.set_xlabel("Count")
            ax.set_ylabel("")
            ax.tick_params(axis="y", labelsize=8)

        fig.tight_layout()
        st.pyplot(fig, use_container_width=False)
        plt.close(fig)

        st.caption("Showing the most frequent categories.")
    else:
        st.info("No categorical features are available for a pie chart.")

with tab3:
    if len(numeric_cols) >= 2:
        corr = data[numeric_cols].corr()

        # Limit heatmap height for datasets with many numeric columns.
        heatmap_height = min(
            max(chart_height, len(numeric_cols) * 0.35),
            8
        )
        fig, ax = plt.subplots(
            figsize=(chart_width, heatmap_height)
        )
        image = ax.imshow(
            corr, aspect="auto", vmin=-1, vmax=1
        )
        ax.set_xticks(range(len(numeric_cols)))
        ax.set_yticks(range(len(numeric_cols)))
        ax.set_xticklabels(
            numeric_cols, rotation=45, ha="right", fontsize=7
        )
        ax.set_yticklabels(numeric_cols, fontsize=7)
        ax.set_title("Numeric Feature Correlations")
        fig.colorbar(image, ax=ax, label="Correlation")
        fig.tight_layout()
        st.pyplot(fig, use_container_width=False)
        plt.close(fig)
    else:
        st.info("At least two numeric features are needed for correlation.")

# ---------- ASK YOUR DATA ----------
st.subheader("Ask Your Data")
st.caption("Click a suggested question or type your own question.")

suggestions = [
    "How many rows and columns are in the dataset?",
    "Are there any missing values?"
]

for col in numeric_cols[:3]:
    suggestions.append(f"What are the statistics for {col}?")

for col in categorical_cols[:2]:
    suggestions.append(f"What are the most common values in {col}?")

if len(numeric_cols) >= 2:
    suggestions.append(
        f"What is the correlation between {numeric_cols[0]} and {numeric_cols[1]}?"
    )

if numeric_cols and categorical_cols:
    suggestions.append(
        f"What is the average {numeric_cols[0]} by {categorical_cols[0]}?"
    )

if "chosen_question" not in st.session_state:
    st.session_state.chosen_question = ""

# Two compact columns of suggested-question buttons
for start in range(0, len(suggestions), 2):
    button_cols = st.columns(2)
    for offset, suggestion in enumerate(suggestions[start:start + 2]):
        if button_cols[offset].button(
            suggestion,
            key=f"suggestion_{start}_{offset}",
            use_container_width=True
        ):
            st.session_state.chosen_question = suggestion

question = st.text_input(
    "Your question",
    key="chosen_question",
    placeholder="e.g. What are the statistics for Age?"
)

analyze = st.button("Analyze", type="primary")

# Results appear below the input, across the full page width.
if analyze and question.strip():
    q = question.lower().replace("_", " ")

    def find_column(columns):
        # Prefer longer column names to avoid partial-name matches.
        for col in sorted(columns, key=len, reverse=True):
            if col.lower().replace("_", " ") in q:
                return col
        return None

    st.divider()
    st.subheader("Answer")

    if "missing" in q or "null values" in q:
        report = pd.DataFrame({
            "Column": data.columns,
            "Missing Count": data.isna().sum().values,
            "Missing Percentage": (
                data.isna().mean().values * 100
            ).round(2)
        }).sort_values("Missing Count", ascending=False)

        st.write(
            f"Total missing cells: **{int(data.isna().sum().sum()):,}**"
        )
        st.dataframe(
            report,
            use_container_width=True,
            height=300,
            hide_index=True
        )

    elif "correlation" in q:
        col_a = find_column(numeric_cols)
        col_b = None

        if col_a:
            col_b = find_column([
                c for c in numeric_cols if c != col_a
            ])

        if col_a and col_b:
            result = data[[col_a, col_b]].corr().iloc[0, 1]
            st.metric("Correlation", f"{result:.3f}")
            st.write(f"Between **{col_a}** and **{col_b}**.")

            if pd.notna(result):
                if result > 0:
                    st.write("The relationship is positive.")
                elif result < 0:
                    st.write("The relationship is negative.")
                else:
                    st.write("The linear correlation is zero.")
            st.caption(
                "Correlation describes association, not proof of causation."
            )
        else:
            st.info("Please mention two numeric feature names.")

    elif any(word in q for word in [
        "statistics", "average", "mean", "median",
        "minimum", "maximum", "describe"
    ]):
        col = find_column(numeric_cols)
        group_col = find_column(categorical_cols)

        if col and " by " in q and group_col:
            grouped = (
                data.groupby(group_col, dropna=False)[col]
                .mean()
                .sort_values(ascending=False)
                .reset_index(name="Average")
            )
            st.dataframe(
                grouped,
                use_container_width=True,
                height=300,
                hide_index=True
            )
        elif col:
            report = data[col].describe().round(2).rename(
                "Value"
            ).to_frame()
            st.dataframe(
                report,
                use_container_width=False,
                height=300
            )
        else:
            st.info("Please include a numeric feature name.")

    elif any(word in q for word in [
        "most common", "common values", "frequency"
    ]):
        col = find_column(categorical_cols)

        if col:
            report = (
                data[col].value_counts(dropna=False)
                .head(10)
                .rename_axis("Value")
                .reset_index(name="Count")
            )
            st.dataframe(
                report,
                use_container_width=True,
                height=300,
                hide_index=True
            )
        else:
            st.info("Please include a categorical feature name.")

    elif any(word in q for word in [
        "rows", "columns", "overview"
    ]):
        st.write(
            f"This dataset contains **{len(data):,} rows** "
            f"and **{len(data.columns)} columns**."
        )

    else:
        st.info(
            "Try a suggested question or ask about statistics, "
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
        mime="text/csv",
        use_container_width=True
    )

with right:
    st.download_button(
        "Download JSON",
        data=data.to_json(
            orient="records", indent=2
        ).encode("utf-8"),
        file_name="analyzed_data.json",
        mime="application/json",
        use_container_width=True
    )

st.divider()
st.caption("Data Analysis Agent · Built with Python and Streamlit")
