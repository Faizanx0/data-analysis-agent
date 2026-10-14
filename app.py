
import hashlib

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st


# ---------- PAGE CONFIGURATION ----------
st.set_page_config(
    page_title="Data Analysis Agent",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Data Analysis Agent")
st.caption("Explore your data. Discover patterns. Get insights.")


# ---------- SESSION STATE ----------
if "file_signature" not in st.session_state:
    st.session_state.file_signature = None

if "chosen_question" not in st.session_state:
    st.session_state.chosen_question = ""


# ---------- DISPLAY SETTINGS ----------
with st.sidebar:
    st.subheader("Display Settings")

    chart_size = st.select_slider(
        "Chart size",
        options=["Compact", "Medium", "Large"],
        value="Medium",
    )

    chart_width, chart_height = {
        "Compact": (4, 2.5),
        "Medium": (5.5, 3.5),
        "Large": (7, 4.5),
    }[chart_size]


# ---------- UPLOAD DATA ----------
uploaded_file = st.file_uploader(
    "Upload your dataset",
    type=["csv"],
)

if uploaded_file is None:
    st.info("Upload a CSV file to begin your analysis.")
    st.stop()

file_bytes = uploaded_file.getvalue()
file_signature = hashlib.sha256(file_bytes).hexdigest()

if st.session_state.file_signature != file_signature:
    st.session_state.file_signature = file_signature
    st.session_state.chosen_question = ""

try:
    original_data = pd.read_csv(uploaded_file)
except Exception as error:
    st.error(f"Unable to read this CSV file: {error}")
    st.stop()

if original_data.empty:
    st.warning("This CSV file has no data to display.")
    st.stop()

# Work on a copy so that filters do not alter the source dataset.
data = original_data.copy()


# ---------- IDENTIFY COLUMN TYPES ----------
# Identify numeric columns and also detect numeric-looking text columns.
numeric_cols = []

for col in data.columns:
    converted = pd.to_numeric(data[col], errors="coerce")

    if converted.notna().any():
        # Preserve genuine text categories such as product IDs.
        # Only classify a column as numeric if its original dtype is
        # numeric or all non-missing values can be converted.
        original_non_null = data[col].notna().sum()

        if (
            pd.api.types.is_numeric_dtype(data[col])
            or converted.notna().sum() == original_non_null
        ):
            numeric_cols.append(col)
            data[col] = converted

categorical_cols = [
    col
    for col in data.columns
    if col not in numeric_cols
    and (
        pd.api.types.is_object_dtype(data[col])
        or isinstance(data[col].dtype, pd.CategoricalDtype)
        or pd.api.types.is_bool_dtype(data[col])
    )
]


# ---------- DATA FILTERING ----------
with st.sidebar:
    st.subheader("Filter Data")

    filter_cols = st.multiselect(
        "Select columns to filter",
        options=data.columns.tolist(),
    )

    for col in filter_cols:
        if col in numeric_cols:
            values = data[col].dropna()

            if values.empty:
                st.caption(f"{col} has no valid numeric values.")
                continue

            min_val = float(values.min())
            max_val = float(values.max())

            if min_val < max_val:
                selected_range = st.slider(
                    f"Range for {col}",
                    min_value=min_val,
                    max_value=max_val,
                    value=(min_val, max_val),
                    key=f"range_{col}",
                )

                data = data[
                    data[col].between(
                        selected_range[0],
                        selected_range[1],
                    )
                    | data[col].isna()
                ]

            else:
                st.caption(
                    f"{col} has only one distinct numeric value."
                )

        elif col in categorical_cols:
            unique_vals = data[col].dropna().unique().tolist()

            if unique_vals:
                selected_vals = st.multiselect(
                    f"Values for {col}",
                    options=unique_vals,
                    default=unique_vals,
                    key=f"category_filter_{col}",
                )

                include_missing = st.checkbox(
                    f"Include missing {col}",
                    value=True,
                    key=f"missing_filter_{col}",
                )

                mask = data[col].isin(selected_vals)

                if include_missing:
                    mask = mask | data[col].isna()

                data = data[mask]


if data.empty:
    st.warning("No records match the applied filter criteria.")
    st.stop()


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
        height=260,
    )


# ---------- VISUAL ANALYTICS ----------
st.subheader("Visual Analytics")

tab1, tab2, tab3 = st.tabs(
    [
        "Numeric Distribution",
        "Category Breakdown",
        "Correlation",
    ]
)


# ---------- NUMERIC DISTRIBUTION ----------
with tab1:
    if numeric_cols:
        selected_num = st.selectbox(
            "Choose a numeric feature",
            numeric_cols,
            key="distribution_column",
        )

        values = data[selected_num].dropna()

        if not values.empty:
            fig, ax = plt.subplots(
                figsize=(chart_width, chart_height)
            )

            ax.hist(
                values,
                bins=20,
                edgecolor="white",
            )

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
            key="category_column",
        )

        top_n = st.slider(
            "Number of categories to display",
            min_value=3,
            max_value=10,
            value=5,
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
            horizontal=True,
        )

        fig, ax = plt.subplots(
            figsize=(chart_width, chart_height)
        )

        if chart_choice == "Pie Chart":
            ax.pie(
                counts.values,
                labels=counts.index,
                autopct="%1.1f%%",
                startangle=90,
                textprops={"fontsize": 8},
            )

            ax.set_title(f"Breakdown of {selected_cat}")
            ax.axis("equal")

        else:
            counts.sort_values().plot(
                kind="barh",
                ax=ax,
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
            8,
        )

        fig, ax = plt.subplots(
            figsize=(chart_width, heatmap_height)
        )

        image = ax.imshow(
            corr,
            aspect="auto",
            vmin=-1,
            vmax=1,
            cmap="coolwarm",
        )

        ax.set_xticks(range(len(numeric_cols)))
        ax.set_yticks(range(len(numeric_cols)))

        ax.set_xticklabels(
            numeric_cols,
            rotation=45,
            ha="right",
            fontsize=7,
        )

        ax.set_yticklabels(numeric_cols, fontsize=7)
        ax.set_title("Numeric Feature Correlations")

        fig.colorbar(image, ax=ax, label="Correlation")
        fig.tight_layout()

        st.pyplot(fig, use_container_width=False)
        plt.close(fig)

        st.caption(
            "Pearson correlation measures linear association, "
            "not causation."
        )
    else:
        st.info(
            "At least two numeric features are needed for correlation."
        )


# ---------- QUESTION SUGGESTIONS ----------
st.subheader("Ask Your Data")
st.caption("Choose a suggestion or type your own question.")

suggestions = [
    "How many rows and columns are in the dataset?",
    "Are there any missing values?",
    "How many duplicate rows are there?",
]

if numeric_cols:
    first_num = numeric_cols[0]

    suggestions.extend([
        f"What are the statistics for {first_num}?",
        f"What is the highest {first_num}?",
        f"What is the average {first_num}?",
    ])

if categorical_cols:
    suggestions.append(
        f"What are the most common values in {categorical_cols[0]}?"
    )

if numeric_cols and categorical_cols:
    suggestions.append(
        f"What is the average {numeric_cols[0]} "
        f"by {categorical_cols[0]}?"
    )

if len(numeric_cols) >= 2:
    suggestions.append(
        f"What is the correlation between {numeric_cols[0]} "
        f"and {numeric_cols[1]}?"
    )

suggestions = list(dict.fromkeys(suggestions))[:6]

for start in range(0, len(suggestions), 2):
    button_cols = st.columns(2)

    for offset, suggestion in enumerate(suggestions[start:start + 2]):
        if button_cols[offset].button(
            suggestion,
            key=f"suggestion_{start}_{offset}",
            use_container_width=True,
        ):
            st.session_state.chosen_question = suggestion


question = st.text_input(
    "Your question",
    key="chosen_question",
    placeholder="e.g. What is the highest Revenue?",
)

analyze = st.button("Analyze", type="primary")


# ---------- QUESTION ANSWER ENGINE ----------
def find_column(columns, query):
    """Find a column mentioned in a question."""
    normalized_query = query.lower().replace("_", " ")

    for col in sorted(columns, key=len, reverse=True):
        normalized_col = str(col).lower().replace("_", " ")

        if normalized_col in normalized_query:
            return col

    return None


if analyze and question.strip():
    q = question.lower().replace("_", " ")

    answer_text = None
    info_text = None
    result_data = None
    result_kind = None

    # Dataset overview
    if any(term in q for term in [
        "rows",
        "columns",
        "overview",
        "size of the dataset",
        "dataset size",
    ]):
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
                q,
            )

        if col_a and col_b:
            pair = data[[col_a, col_b]].dropna()

            if len(pair) < 2:
                info_text = (
                    "At least two complete observations are required."
                )
            else:
                correlation = pair[col_a].corr(pair[col_b])

                if pd.isna(correlation):
                    info_text = (
                        "Correlation could not be calculated. "
                        "One of the columns may be constant."
                    )
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
                        "Correlation indicates association, "
                        "not proof of causation."
                    )
        else:
            info_text = (
                "Please mention two numeric column names."
            )

    # Aggregation by category
    elif " by " in q and any(
        word in q
        for word in ["average", "mean", "total", "sum"]
    ):
        num_col = find_column(numeric_cols, q)
        cat_col = find_column(categorical_cols, q)

        if num_col and cat_col:
            grouped = data.groupby(
                cat_col,
                dropna=False,
                observed=True,
            )[num_col]

            if "average" in q or "mean" in q:
                grouped_result = grouped.mean()
                label = "Average"
            else:
                grouped_result = grouped.sum(min_count=1)
                label = "Total"

            result_data = (
                grouped_result
                .sort_values(ascending=False, na_position="last")
                .reset_index(name=label)
            )

            result_kind = "table"
        else:
            info_text = (
                "Please mention a numeric column and a category column."
            )

    # Numeric statistics and individual values
    elif any(word in q for word in [
        "statistics",
        "average",
        "mean",
        "median",
        "minimum",
        "maximum",
        "highest",
        "lowest",
        "total",
        "sum",
        "describe",
    ]):
        col = find_column(numeric_cols, q)

        if col:
            values = data[col].dropna()

            if values.empty:
                info_text = (
                    f"No valid numeric values were found in {col}."
                )

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
                stats = values.describe()

                result_data = pd.DataFrame({
                    "Statistic": [
                        "Count",
                        "Average (Mean)",
                        "Standard Deviation",
                        "Minimum",
                        "25th Percentile",
                        "Median (50th Percentile)",
                        "75th Percentile",
                        "Maximum",
                    ],
                    "Value": [
                        stats["count"],
                        stats["mean"],
                        stats["std"],
                        stats["min"],
                        stats["25%"],
                        stats["50%"],
                        stats["75%"],
                        stats["max"],
                    ],
                }).round(2)

                result_kind = "table"

        else:
            info_text = "Please include a numeric column name."

    # Most common categorical values
    elif any(word in q for word in [
        "most common",
        "common values",
        "frequency",
        "frequent",
    ]):
        col = find_column(categorical_cols, q)

        if col:
            result_data = (
                data[col]
                .fillna("Missing")
                .astype(str)
                .value_counts()
                .head(10)
                .rename_axis("Value")
                .reset_index(name="Count")
            )

            result_kind = "table"
        else:
            info_text = "Please include a categorical column name."

    else:
        info_text = (
            "Try a suggested question or ask about statistics, "
            "highest or lowest values, totals, averages, missing "
            "values, common values, duplicates, or correlation."
        )

    # ---------- DISPLAY ANSWER ----------
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
            ).round(2),
        }).sort_values(
            "Missing Count",
            ascending=False,
        ).reset_index(drop=True)

        st.write(
            f"Total missing cells: "
            f"**{int(data.isna().sum().sum()):,}**"
        )

        st.dataframe(
            report,
            use_container_width=True,
            height=300,
            hide_index=True,
        )

    elif result_kind == "table" and result_data is not None:
        st.dataframe(
            result_data,
            use_container_width=True,
            height=300,
            hide_index=True,
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
        use_container_width=True,
    )

with right:
    st.download_button(
        "Download JSON",
        data=data.to_json(
            orient="records",
            indent=2,
            date_format="iso",
        ).encode("utf-8"),
        file_name="analyzed_data.json",
        mime="application/json",
        use_container_width=True,
    )


st.divider()
st.caption("Data Analysis Agent · Built with Python and Streamlit")
