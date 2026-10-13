
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

# ---------- DISPLAY SETTINGS ----------
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

    st.caption("Choose how large your charts appear.")

# ---------- FILE UPLOAD ----------
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

# Reset the previous question when a different file is uploaded.
# The content hash also detects different files with the same name and size.
file_signature = (
    uploaded_file.name,
    uploaded_file.size,
    hash(uploaded_file.getvalue())
)

if st.session_state.get("file_signature") != file_signature:
    st.session_state.file_signature = file_signature
    st.session_state.chosen_question = ""
    st.session_state.last_question = ""
    st.session_state.analysis_result = None

# ---------- IDENTIFY COLUMN TYPES ----------
numeric_cols = data.select_dtypes(
    include="number"
).columns.tolist()

categorical_cols = data.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()

# Include dates stored as text when they can be parsed reliably.
date_cols = []
for col in data.columns:
    if col in numeric_cols:
        continue

    if pd.api.types.is_datetime64_any_dtype(data[col]):
        date_cols.append(col)
    elif data[col].dtype == "object":
        non_null = data[col].dropna()

        if len(non_null) > 0:
            sample = non_null.head(30)
            parsed = pd.to_datetime(sample, errors="coerce")
            if parsed.notna().mean() >= 0.8:
                date_cols.append(col)

# ---------- DATASET OVERVIEW ----------
st.subheader("Dataset Overview")

c1, c2, c3, c4 = st.columns(4)

c1.metric("Total Records", f"{len(data):,}")
c2.metric("Total Features", len(data.columns))
c3.metric("Missing Values", f"{int(data.isna().sum().sum()):,}")
c4.metric("Duplicate Rows", f"{int(data.duplicated().sum()):,}")

with st.expander("View dataset preview"):
    st.dataframe(
        data.head(10),
        use_container_width=True,
        height=260
    )

# ---------- VISUAL ANALYTICS ----------
st.subheader("Visual Analytics")

tab1, tab2, tab3 = st.tabs([
    "Numeric Distribution",
    "Category Breakdown",
    "Correlation"
])

# ---------- NUMERIC DISTRIBUTION ----------
with tab1:
    if numeric_cols:
        selected_num = st.selectbox(
            "Choose a numeric feature",
            numeric_cols,
            key="distribution_column"
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
        st.info("No numeric features are available.")

# ---------- CATEGORY BREAKDOWN ----------
with tab2:
    if categorical_cols:
        selected_cat = st.selectbox(
            "Choose a category",
            categorical_cols,
            key="category_column"
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

            counts.sort_values().plot(
                kind="barh",
                ax=ax
            )

            ax.set_title(f"Most common values in {selected_cat}")
            ax.set_xlabel("Count")
            ax.set_ylabel("")
            ax.tick_params(axis="y", labelsize=8)

        fig.tight_layout()
        st.pyplot(fig, use_container_width=False)
        plt.close(fig)

        st.caption("Showing the most frequent categories.")
    else:
        st.info("No categorical features are available.")

# ---------- CORRELATION ----------
with tab3:
    if len(numeric_cols) >= 2:
        corr = data[numeric_cols].corr()

        heatmap_height = min(
            max(chart_height, len(numeric_cols) * 0.35),
            8
        )

        fig, ax = plt.subplots(
            figsize=(chart_width, heatmap_height)
        )

        image = ax.imshow(
            corr,
            aspect="auto",
            vmin=-1,
            vmax=1
        )

        ax.set_xticks(range(len(numeric_cols)))
        ax.set_yticks(range(len(numeric_cols)))

        ax.set_xticklabels(
            numeric_cols,
            rotation=45,
            ha="right",
            fontsize=7
        )

        ax.set_yticklabels(numeric_cols, fontsize=7)
        ax.set_title("Numeric Feature Correlations")

        fig.colorbar(image, ax=ax, label="Correlation")
        fig.tight_layout()

        st.pyplot(fig, use_container_width=False)
        plt.close(fig)
    else:
        st.info("At least two numeric features are needed for correlation.")

# ---------- DYNAMIC QUESTION SUGGESTIONS ----------
st.subheader("Ask Your Data")
st.caption(
    "Suggested questions update automatically based on the uploaded dataset."
)

suggestions = [
    "How many rows and columns are in the dataset?",
    "Are there any missing values?",
    "How many duplicate rows are there?"
]

# Questions for every numeric column
for col in numeric_cols:
    suggestions.extend([
        f"What are the statistics for {col}?",
        f"What is the average {col}?",
        f"What is the total {col}?",
        f"What is the highest {col}?",
        f"What is the lowest {col}?"
    ])

# Questions for every categorical column
for col in categorical_cols:
    suggestions.append(
        f"What are the most common values in {col}?"
    )

# Grouping questions using actual column names
if numeric_cols and categorical_cols:
    for num_col in numeric_cols[:3]:
        for cat_col in categorical_cols[:3]:
            suggestions.append(
                f"What is the average {num_col} by {cat_col}?"
            )
            suggestions.append(
                f"What is the total {num_col} by {cat_col}?"
            )

# Correlation questions
if len(numeric_cols) >= 2:
    suggestions.append(
        f"What is the correlation between "
        f"{numeric_cols[0]} and {numeric_cols[1]}?"
    )

# Make questions unique while preserving order
suggestions = list(dict.fromkeys(suggestions))

# Keep the suggestion list manageable for wide datasets
suggestions = suggestions[:30]

# Show the suggestion buttons in two columns
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
    placeholder="e.g. What is the highest Revenue?"
)

analyze = st.button("Analyze", type="primary")

# ---------- QUESTION ANSWER ENGINE ----------
if analyze and question.strip():
    q = question.lower().replace("_", " ")
    result_kind = None
    result_data = None
    answer_text = None
    info_text = None

    def find_column(columns, query):
        # Match full column names, preferring longer names.
        normalized_query = query.lower().replace("_", " ")

        for col in sorted(columns, key=len, reverse=True):
            normalized_col = col.lower().replace("_", " ")
            if normalized_col in normalized_query:
                return col

        return None

    # Dataset-level questions
    if "missing" in q or "null values" in q:
        result_kind = "missing"

    elif "duplicate" in q:
        answer_text = (
            f"This dataset contains **{int(data.duplicated().sum()):,}** "
            "duplicate rows."
        )

    elif (
        "rows" in q
        or "columns" in q
        or "overview" in q
        or "size of the dataset" in q
    ):
        answer_text = (
            f"This dataset contains **{len(data):,} rows** "
            f"and **{len(data.columns)} columns**."
        )

    # Correlation questions
    elif "correlation" in q:
        col_a = find_column(numeric_cols, q)
        col_b = None

        if col_a:
            remaining_cols = [
                col for col in numeric_cols if col != col_a
            ]
            col_b = find_column(remaining_cols, q)

        if col_a and col_b:
            correlation = data[[col_a, col_b]].corr().iloc[0, 1]

            if pd.isna(correlation):
                info_text = "Correlation could not be calculated for these columns."
            else:
                answer_text = (
                    f"Correlation between **{col_a}** and "
                    f"**{col_b}** is **{correlation:.3f}**."
                )

                if correlation > 0:
                    answer_text += " This is a positive relationship."
                elif correlation < 0:
                    answer_text += " This is a negative relationship."
                else:
                    answer_text += " The linear correlation is zero."

                info_text = (
                    "Correlation indicates association, not proof of causation."
                )
        else:
            info_text = (
                "Please select a suggested correlation question or mention "
                "two numeric column names."
            )

    # Grouped numeric questions
    elif " by " in q and (
        "average" in q
        or "mean" in q
        or "total" in q
        or "sum" in q
    ):
        num_col = find_column(numeric_cols, q)
        cat_col = find_column(categorical_cols, q)

        if num_col and cat_col:
            grouped_data = data.groupby(
                cat_col,
                dropna=False
            )[num_col]

            if "average" in q or "mean" in q:
                grouped = grouped_data.mean()
                label = "Average"
            else:
                grouped = grouped_data.sum()
                label = "Total"

            result_data = (
                grouped.sort_values(ascending=False)
                .reset_index(name=label)
            )
            result_kind = "table"
        else:
            info_text = (
                "Please include a numeric column and a category column."
            )

    # Numeric-column questions
    elif any(word in q for word in [
        "statistics", "average", "mean", "median",
        "minimum", "maximum", "highest", "lowest",
        "total", "sum", "describe"
    ]):
        col = find_column(numeric_cols, q)

        if col:
            values = pd.to_numeric(
                data[col], errors="coerce"
            ).dropna()

            if values.empty:
                info_text = f"No valid numeric values were found in {col}."
            elif "highest" in q or "maximum" in q:
                answer_text = f"Highest **{col}**: **{values.max():,.2f}**"
            elif "lowest" in q or "minimum" in q:
                answer_text = f"Lowest **{col}**: **{values.min():,.2f}**"
            elif "total" in q or "sum" in q:
                answer_text = f"Total **{col}**: **{values.sum():,.2f}**"
            elif "average" in q or "mean" in q:
                answer_text = f"Average **{col}**: **{values.mean():,.2f}**"
            elif "median" in q:
                answer_text = f"Median **{col}**: **{values.median():,.2f}**"
            else:
                result_data = (
                    values.describe()
                    .round(2)
                    .rename("Value")
                    .to_frame()
                )
                result_kind = "table"
        else:
            info_text = "Please include a numeric column name."

    # Categorical-column questions
    elif any(word in q for word in [
        "most common", "common values", "frequency"
    ]):
        col = find_column(categorical_cols, q)

        if col:
            result_data = (
                data[col]
                .value_counts(dropna=False)
                .head(10)
                .rename_axis("Value")
                .reset_index(name="Count")
            )
            result_kind = "table"
        else:
            info_text = "Please include a categorical column name."

    else:
        info_text = (
            "I couldn't identify that question yet. Try one of the suggested "
            "questions, or ask about a column's statistics, total, average, "
            "highest or lowest value, missing values, common values, "
            "correlation, or averages grouped by category."
        )

    # ---------- DISPLAY ANSWER ----------
    st.divider()
    st.subheader("Answer")

    if answer_text:
        st.markdown(answer_text)

    if info_text:
        st.info(info_text)

    if result_kind == "missing":
        report = pd.DataFrame({
            "Column": data.columns,
            "Missing Count": data.isna().sum().values,
            "Missing Percentage": (
                data.isna().mean().values * 100
            ).round(2)
        }).sort_values("Missing Count", ascending=False)

        st.write(
            f"Total missing cells: "
            f"**{int(data.isna().sum().sum()):,}**"
        )
        st.dataframe(
            report,
            use_container_width=True,
            height=300,
            hide_index=True
        )

    elif result_kind == "table":
        st.dataframe(
            result_data,
            use_container_width=True,
            height=300,
            hide_index=True
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
            orient="records",
            indent=2
        ).encode("utf-8"),
        file_name="analyzed_data.json",
        mime="application/json",
        use_container_width=True
    )

st.divider()
st.caption("Data Analysis Agent · Built with Python and Streamlit")
