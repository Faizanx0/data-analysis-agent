
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

# ---------- UPLOAD DATA ----------
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

# Detect when a different CSV is uploaded
file_signature = (
    uploaded_file.name,
    uploaded_file.size,
    hash(uploaded_file.getvalue())
)

if st.session_state.get("file_signature") != file_signature:
    st.session_state.file_signature = file_signature
    st.session_state.chosen_question = ""

# ---------- IDENTIFY COLUMN TYPES ----------
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
st.caption("Choose a suggestion or type your own question.")

suggestions = [
    "How many rows and columns are in the dataset?",
    "Are there any missing values?"
]

if numeric_cols:
    suggestions.append(
        f"What are the statistics for {numeric_cols[0]}?"
    )
    suggestions.append(
        f"What is the highest {numeric_cols[0]}?"
    )

if categorical_cols:
    suggestions.append(
        f"What are the most common values in {categorical_cols[0]}?"
    )

if numeric_cols and categorical_cols:
    suggestions.append(
        f"What is the average {numeric_cols[0]} by {categorical_cols[0]}?"
    )
elif len(numeric_cols) >= 2:
    suggestions.append(
        f"What is the correlation between {numeric_cols[0]} and {numeric_cols[1]}?"
    )

# Display no more than six unique suggestions
suggestions = list(dict.fromkeys(suggestions))[:6]

# Display clickable suggestions in two columns
for start in range(0, len(suggestions), 2):
    button_cols = st.columns(2)

    for offset, suggestion in enumerate(suggestions[start:start + 2]):
        if button_cols[offset].button(
            suggestion,
            key=f"suggestion_{start}_{offset}",
            use_container_width=True
        ):
            st.session_state.chosen_question = suggestion

# IMPORTANT: Define question and analyze BEFORE using them below
question = st.text_input(
    "Your question",
    key="chosen_question",
    placeholder="e.g. What is the highest Revenue?"
)

analyze = st.button("Analyze", type="primary")

# ---------- QUESTION ANSWER ENGINE ----------
if analyze and question.strip():
    q = question.lower().replace("_", " ")

    answer_text = None
    info_text = None
    result_data = None
    result_kind = None

    def find_column(columns, query):
        """Find a mentioned column, preferring longer column names."""
        normalized_query = query.lower().replace("_", " ")

        for col in sorted(columns, key=len, reverse=True):
            normalized_col = col.lower().replace("_", " ")
            if normalized_col in normalized_query:
                return col

        return None

    # Dataset overview
    if (
        "rows" in q
        or "columns" in q
        or "overview" in q
        or "size of the dataset" in q
    ):
        answer_text = (
            f"This dataset contains **{len(data):,} rows** "
            f"and **{len(data.columns)} columns**."
        )

    # Missing values
    elif "missing" in q or "null values" in q:
        result_kind = "missing"

    # Duplicate rows
    elif "duplicate" in q:
        answer_text = (
            f"This dataset contains **{int(data.duplicated().sum()):,}** "
            "duplicate rows."
        )

    # Correlation
    elif "correlation" in q:
        col_a = find_column(numeric_cols, q)
        col_b = None

        if col_a:
            col_b = find_column(
                [col for col in numeric_cols if col != col_a],
                q
            )

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
                "Please mention two numeric column names or choose "
                "a suggested correlation question."
            )

    # Average or total by category
    elif " by " in q and any(
        word in q for word in ["average", "mean", "total", "sum"]
    ):
        num_col = find_column(numeric_cols, q)
        cat_col = find_column(categorical_cols, q)

        if num_col and cat_col:
            grouped = data.groupby(
                cat_col, dropna=False
            )[num_col]

            if "average" in q or "mean" in q:
                grouped_result = grouped.mean()
                label = "Average"
            else:
                grouped_result = grouped.sum()
                label = "Total"

            result_data = (
                grouped_result.sort_values(ascending=False)
                .reset_index(name=label)
            )
            result_kind = "table"
        else:
            info_text = (
                "Please mention a numeric column and a category column."
            )

    # Numeric statistics and individual values
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
                answer_text = (
                    f"Highest **{col}**: **{values.max():,.2f}**"
                )

            elif "lowest" in q or "minimum" in q:
                answer_text = (
                    f"Lowest **{col}**: **{values.min():,.2f}**"
                )

            elif "total" in q or "sum" in q:
                answer_text = (
                    f"Total **{col}**: **{values.sum():,.2f}**"
                )

            elif "average" in q or "mean" in q:
                answer_text = (
                    f"Average **{col}**: **{values.mean():,.2f}**"
                )

            elif "median" in q:
                answer_text = (
                    f"Median **{col}**: **{values.median():,.2f}**"
                )

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

    # Most common categorical values
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
            "Try a suggested question or ask about statistics, highest "
            "or lowest values, totals, averages, missing values, "
            "common values, or correlation."
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
        }).sort_values(
            "Missing Count",
            ascending=False
        )

        st.write(
            f"Total missing cells: **{int(data.isna().sum().sum()):,}**"
        )

        st.dataframe(
            report,
            use_container_width=True,
            height=300,
            hide_index=True
        )

    elif result_kind == "table" and result_data is not None:
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
