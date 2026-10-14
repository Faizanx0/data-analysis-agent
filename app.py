
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Data Analysis Agent",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Data Analysis Agent")
st.caption("Explore your data. Discover patterns. Get insights.")


# =========================================================
# TOOL 1: DESCRIPTIVE STATISTICS
# =========================================================

def descriptive_statistics(df, column):
    """Calculate descriptive statistics for a numeric column."""

    if column not in df.columns:
        return None

    values = pd.to_numeric(
        df[column], errors="coerce"
    ).dropna()

    if values.empty:
        return None

    stats = values.describe()

    return pd.DataFrame({
        "Statistic": [
            "Count",
            "Mean",
            "Standard Deviation",
            "Minimum",
            "25th Percentile",
            "Median",
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


# =========================================================
# TOOL 2: MISSING-VALUE ANALYSIS
# =========================================================

def missing_value_analysis(df):
    """Calculate missing counts and percentages per column."""

    report = pd.DataFrame({
        "Column": df.columns,
        "Missing Count": df.isna().sum().values,
        "Missing Percentage": (
            df.isna().mean().values * 100
        ).round(2),
    })

    return report.sort_values(
        "Missing Count",
        ascending=False,
    ).reset_index(drop=True)


# =========================================================
# TOOL 3: CORRELATION ANALYSIS
# =========================================================

def correlation_analysis(df, column_a=None, column_b=None):
    """Calculate pairwise correlation or a full correlation matrix."""

    numeric_data = df.select_dtypes(include="number")

    if column_a is not None and column_b is not None:
        if (
            column_a not in numeric_data.columns
            or column_b not in numeric_data.columns
        ):
            return None

        pair = numeric_data[[column_a, column_b]].dropna()

        if len(pair) < 2:
            return None

        return float(
            pair[column_a].corr(pair[column_b])
        )

    if numeric_data.shape[1] < 2:
        return None

    return numeric_data.corr().round(3)


# =========================================================
# TOOL 4: GROUP-BY ANALYSIS
# =========================================================

def groupby_analysis(
    df,
    category_column,
    numeric_column,
    operation="mean",
):
    """Aggregate a numeric column by a category."""

    if (
        category_column not in df.columns
        or numeric_column not in df.columns
    ):
        return None

    if operation not in {"mean", "sum", "count"}:
        return None

    values = pd.to_numeric(
        df[numeric_column], errors="coerce"
    )

    working_data = df[[category_column]].copy()
    working_data[numeric_column] = values

    grouped = working_data.groupby(
        category_column,
        dropna=False,
    )[numeric_column]

    result = getattr(grouped, operation)()

    labels = {
        "mean": "Average",
        "sum": "Total",
        "count": "Count",
    }

    return (
        result.sort_values(ascending=False)
        .reset_index(name=labels[operation])
    )


# =========================================================
# TOOL 5: DATA FILTERING
# =========================================================

def filter_data(df, column, operator, value):
    """Filter rows using numeric or text conditions."""

    if column not in df.columns:
        return None

    series = df[column]

    if operator in {">", ">=", "<", "<="}:
        numeric_series = pd.to_numeric(
            series, errors="coerce"
        )

        try:
            number = float(value)
        except (TypeError, ValueError):
            return None

        conditions = {
            ">": numeric_series > number,
            ">=": numeric_series >= number,
            "<": numeric_series < number,
            "<=": numeric_series <= number,
        }

        return df[conditions[operator]].copy()

    if operator == "equals":
        mask = (
            series.astype(str).str.casefold()
            == str(value).casefold()
        )
        return df[mask].copy()

    if operator == "contains":
        mask = series.astype(str).str.contains(
            re.escape(str(value)),
            case=False,
            na=False,
        )
        return df[mask].copy()

    return None


# =========================================================
# QUESTION-ROUTING HELPERS
# =========================================================

def find_column(columns, question):
    """Find a column mentioned in a question."""

    normalized_question = (
        question.lower().replace("_", " ")
    )

    # Longest column names first to avoid partial matches.
    for column in sorted(
        columns, key=lambda item: len(str(item)), reverse=True
    ):
        normalized_column = (
            str(column).lower().replace("_", " ")
        )

        if normalized_column in normalized_question:
            return column

    return None


def parse_filter_question(question, df):
    """
    Recognize simple filters, for example:
    Sales > 1000
    Region equals West
    Product contains phone
    """

    column = find_column(df.columns.tolist(), question)

    if column is None:
        return None, None, None

    escaped_column = re.escape(
        str(column).replace("_", " ")
    )

    # Numeric comparisons.
    for symbol in [">=", "<=", ">", "<"]:
        pattern = (
            escaped_column
            + r"\s*"
            + re.escape(symbol)
            + r"\s*(-?\d+(?:\.\d+)?)"
        )

        match = re.search(
            pattern, question, flags=re.IGNORECASE
        )

        if match:
            return column, symbol, match.group(1)

    # Text equals.
    match = re.search(
        escaped_column
        + r"\s+(?:equals?|is)\s+['\"]?(.+?)['\"]?\s*$",
        question,
        flags=re.IGNORECASE,
    )

    if match:
        return column, "equals", match.group(1).strip()

    # Text contains.
    match = re.search(
        escaped_column
        + r"\s+contains?\s+['\"]?(.+?)['\"]?\s*$",
        question,
        flags=re.IGNORECASE,
    )

    if match:
        return column, "contains", match.group(1).strip()

    return None, None, None


# =========================================================
# SIDEBAR SETTINGS
# =========================================================

with st.sidebar:
    st.subheader("Display Settings")

    chart_size = st.select_slider(
        "Chart size",
        options=["Compact", "Medium", "Large"],
        value="Compact",
    )

    chart_width = {
        "Compact": 4,
        "Medium": 5.5,
        "Large": 7,
    }[chart_size]

    chart_height = {
        "Compact": 2.5,
        "Medium": 3.5,
        "Large": 4.5,
    }[chart_size]


# =========================================================
# CSV UPLOAD
# =========================================================

uploaded_file = st.file_uploader(
    "Upload your dataset",
    type=["csv"],
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

# Clear the previous question when a new file is uploaded.
file_signature = (
    uploaded_file.name,
    uploaded_file.size,
    hash(uploaded_file.getvalue()),
)

if st.session_state.get("file_signature") != file_signature:
    st.session_state.file_signature = file_signature
    st.session_state.chosen_question = ""

numeric_cols = data.select_dtypes(
    include="number"
).columns.tolist()

categorical_cols = data.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()


# =========================================================
# DATASET OVERVIEW
# =========================================================

st.subheader("Dataset Overview")

c1, c2, c3, c4 = st.columns(4)

c1.metric("Total Records", f"{len(data):,}")
c2.metric("Total Features", len(data.columns))
c3.metric(
    "Missing Values",
    f"{int(data.isna().sum().sum()):,}",
)
c4.metric(
    "Duplicate Rows",
    f"{int(data.duplicated().sum()):,}",
)

with st.expander("View dataset preview"):
    st.dataframe(
        data.head(10),
        use_container_width=True,
        height=260,
    )


# =========================================================
# VISUAL ANALYTICS
# =========================================================

st.subheader("Visual Analytics")

tab1, tab2, tab3 = st.tabs([
    "Numeric Distribution",
    "Category Breakdown",
    "Correlation",
])


# Numeric distribution
with tab1:
    if numeric_cols:
        selected_num = st.selectbox(
            "Choose a numeric feature",
            numeric_cols,
            key="distribution_column",
        )

        values = pd.to_numeric(
            data[selected_num], errors="coerce"
        ).dropna()

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
            st.info("This column has no valid numeric values.")
    else:
        st.info("No numeric columns are available.")


# Category breakdown
with tab2:
    if categorical_cols:
        selected_cat = st.selectbox(
            "Choose a category",
            categorical_cols,
            key="category_column",
        )

        top_n = st.slider(
            "Number of categories",
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
    else:
        st.info("No categorical columns are available.")


# Correlation heatmap
with tab3:
    corr_matrix = correlation_analysis(data)

    if isinstance(corr_matrix, pd.DataFrame):
        fig, ax = plt.subplots(
            figsize=(
                chart_width,
                min(max(chart_height, len(numeric_cols) * 0.35), 8),
            )
        )

        image = ax.imshow(
            corr_matrix,
            aspect="auto",
            vmin=-1,
            vmax=1,
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
    else:
        st.info(
            "At least two numeric columns are needed for correlation."
        )


# =========================================================
# ASK YOUR DATA
# =========================================================

st.subheader("Ask Your Data")
st.caption("Use the actual column names from your CSV.")

suggestions = [
    "How many rows and columns are in the dataset?",
    "Are there any missing values?",
    "How many duplicate rows are there?",
]

if numeric_cols:
    suggestions.extend([
        f"What are the statistics for {numeric_cols[0]}?",
        f"What is the highest {numeric_cols[0]}?",
    ])

if categorical_cols:
    suggestions.append(
        f"What are the most common values in {categorical_cols[0]}?"
    )

if numeric_cols and categorical_cols:
    suggestions.append(
        f"What is the average {numeric_cols[0]} by {categorical_cols[0]}?"
    )

if len(numeric_cols) >= 2:
    suggestions.append(
        f"What is the correlation between {numeric_cols[0]} "
        f"and {numeric_cols[1]}?"
    )

suggestions = list(dict.fromkeys(suggestions))[:6]

for start in range(0, len(suggestions), 2):
    button_cols = st.columns(2)

    for offset, suggestion in enumerate(
        suggestions[start:start + 2]
    ):
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


# =========================================================
# AGENT TOOL ROUTER
# =========================================================

if analyze and question.strip():
    q = question.lower().replace("_", " ")

    answer_text = None
    info_text = None
    result_data = None

    # TOOL 2: Missing values
    if (
        "missing" in q
        or "null values" in q
        or "nulls" in q
    ):
        result_data = missing_value_analysis(data)

        answer_text = (
            f"Found **{int(data.isna().sum().sum()):,} "
            "missing cells** across the dataset."
        )

    # Duplicate count
    elif "duplicate" in q:
        answer_text = (
            f"This dataset contains "
            f"**{int(data.duplicated().sum()):,} duplicate rows**."
        )

    # TOOL 3: Correlation
    elif "correlation" in q:
        col_a = find_column(numeric_cols, q)

        col_b = (
            find_column(
                [col for col in numeric_cols if col != col_a],
                q,
            )
            if col_a
            else None
        )

        if col_a and col_b:
            correlation = correlation_analysis(
                data, col_a, col_b
            )

            if correlation is None or pd.isna(correlation):
                info_text = (
                    "Correlation could not be calculated for "
                    "these columns."
                )
            else:
                answer_text = (
                    f"Correlation between **{col_a}** and "
                    f"**{col_b}** is **{correlation:.3f}**."
                )
                info_text = (
                    "Correlation measures association; it does "
                    "not prove causation."
                )
        else:
            matrix = correlation_analysis(data)

            if isinstance(matrix, pd.DataFrame):
                result_data = matrix
                info_text = (
                    "Here is the full numeric correlation matrix."
                )
            else:
                info_text = (
                    "Mention two numeric column names. At least "
                    "two numeric columns are required."
                )

    # TOOL 4: Group-by aggregation
    elif " by " in q and any(
        word in q
        for word in ["average", "mean", "total", "sum", "count"]
    ):
        num_col = find_column(numeric_cols, q)
        cat_col = find_column(categorical_cols, q)

        if num_col and cat_col:
            if "average" in q or "mean" in q:
                operation = "mean"
            elif "total" in q or "sum" in q:
                operation = "sum"
            else:
                operation = "count"

            result_data = groupby_analysis(
                data,
                cat_col,
                num_col,
                operation,
            )

            answer_text = (
                f"{operation.title()} of **{num_col}** "
                f"grouped by **{cat_col}**."
            )
        else:
            info_text = (
                "Mention one numeric column and one category "
                "column, for example: Average Sales by Region."
            )

    # TOOL 5: Filtering
    elif any(
        operator in q
        for operator in [">=", "<=", ">", "<", " equals ", " is ", " contains "]
    ):
        col, operator, value = parse_filter_question(
            question, data
        )

        if col is not None and operator is not None:
            result_data = filter_data(
                data, col, operator, value
            )

            if result_data is not None:
                answer_text = (
                    f"Filter returned **{len(result_data):,} rows** "
                    f"where **{col} {operator} {value}**."
                )
            else:
                info_text = (
                    "Unable to apply this filter. Check the "
                    "column name and condition."
                )
        else:
            info_text = (
                "Try a question like `Sales > 1000`, "
                "`Region equals West`, or `Product contains phone`."
            )

    # TOOL 1: Descriptive statistics and numeric queries
    elif any(
        word in q
        for word in [
            "statistics", "describe", "average", "mean",
            "median", "minimum", "maximum", "highest",
            "lowest", "total", "sum",
        ]
    ):
        col = find_column(numeric_cols, q)

        if col:
            values = pd.to_numeric(
                data[col], errors="coerce"
            ).dropna()

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
                result_data = descriptive_statistics(data, col)
                answer_text = (
                    f"Descriptive statistics for **{col}**."
                )
        else:
            info_text = (
                "Please include a numeric column name."
            )

    # Most common categorical values
    elif any(
        word in q
        for word in ["most common", "common values", "frequency"]
    ):
        col = find_column(categorical_cols, q)

        if col:
            result_data = (
                data[col]
                .value_counts(dropna=False)
                .head(10)
                .rename_axis("Value")
                .reset_index(name="Count")
            )
        else:
            info_text = (
                "Please include a categorical column name."
            )

    # Dataset size
    elif any(
        word in q
        for word in ["rows", "columns", "overview", "dataset size"]
    ):
        answer_text = (
            f"This dataset contains **{len(data):,} rows** "
            f"and **{len(data.columns)} columns**."
        )

    else:
        info_text = (
            "Try statistics, missing values, correlation, "
            "average/total by category, common values, duplicates, "
            "or a filter like `Sales > 1000`."
        )

    # Display result
    st.subheader("Answer")

    if answer_text:
        st.markdown(answer_text)

    if info_text:
        st.info(info_text)

    if result_data is not None:
        st.dataframe(
            result_data,
            use_container_width=True,
            height=320,
            hide_index=True,
        )


# =========================================================
# EXPORTS
# =========================================================

st.subheader("Export Dataset")

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
        ).encode("utf-8"),
        file_name="analyzed_data.json",
        mime="application/json",
        use_container_width=True,
    )


# =========================================================
# TOOL DOCUMENTATION IN APP
# =========================================================

with st.expander("About the five analysis tools"):
    st.markdown("""
    **1. Descriptive statistics**
    - Count, mean, standard deviation, quartiles, minimum, maximum.

    **2. Missing-value analysis**
    - Missing cell count and missing percentage by column.

    **3. Correlation analysis**
    - Pairwise correlation or a full numeric correlation matrix.

    **4. Group-by analysis**
    - Calculate average, total, or count by category.

    **5. Data filtering**
    - Numeric comparisons: `>`, `>=`, `<`, `<=`.
    - Text filters: `equals` and `contains`.
    """)

st.divider()
st.caption(
    "Data Analysis Agent · Python, Pandas, NumPy, "
    "Matplotlib, and Streamlit"
)
