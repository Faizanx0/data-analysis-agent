
import re

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="Data Analysis Agent",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Data Analysis Agent")
st.caption("Explore your data. Discover patterns. Get insights.")


def normalize_name(value):
    return re.sub(r"[^a-z0-9]+", " ", str(value).lower()).strip()


def numeric_columns(df):
    result = []

    for column in df.columns:
        values = pd.to_numeric(df[column], errors="coerce")
        original_count = df[column].notna().sum()

        if original_count and values.notna().sum() >= original_count * 0.8:
            result.append(column)

    return result


def find_column(columns, question):
    q = normalize_name(question)

    for column in sorted(columns, key=lambda item: len(str(item)), reverse=True):
        name = normalize_name(column)

        if name and re.search(r"(?<!\w)" + re.escape(name) + r"(?!\w)", q):
            return column

    return None


# Tool 1: Descriptive statistics
def descriptive_statistics(df, column):
    values = pd.to_numeric(df[column], errors="coerce").dropna()

    if values.empty:
        return None

    stats = values.describe()

    return pd.DataFrame({
        "Statistic": [
            "Count", "Mean", "Standard Deviation", "Minimum",
            "25th Percentile", "Median", "75th Percentile", "Maximum",
        ],
        "Value": [
            stats["count"], stats["mean"], stats["std"], stats["min"],
            stats["25%"], stats["50%"], stats["75%"], stats["max"],
        ],
    }).round(2)


# Tool 2: Missing-value analysis
def missing_value_analysis(df):
    return pd.DataFrame({
        "Column": df.columns,
        "Missing Count": df.isna().sum().values,
        "Missing Percentage": (df.isna().mean().values * 100).round(2),
    }).sort_values("Missing Count", ascending=False).reset_index(drop=True)


# Tool 3: Correlation analysis
def correlation_analysis(df, column_a=None, column_b=None):
    columns = numeric_columns(df)
    numeric_data = df[columns].apply(pd.to_numeric, errors="coerce")

    if column_a is not None and column_b is not None:
        if column_a not in numeric_data or column_b not in numeric_data:
            return None

        pair = numeric_data[[column_a, column_b]].dropna()

        if len(pair) < 2:
            return None

        return float(pair[column_a].corr(pair[column_b]))

    if numeric_data.shape[1] < 2:
        return None

    return numeric_data.corr().round(3)


# Tool 4: Group-by analysis
def groupby_analysis(df, category_column, numeric_column, operation="mean"):
    if category_column not in df.columns or numeric_column not in df.columns:
        return None

    if operation not in {"mean", "sum", "count"}:
        return None

    working = df[[category_column]].copy()
    working[numeric_column] = pd.to_numeric(
        df[numeric_column], errors="coerce"
    )

    grouped = working.groupby(category_column, dropna=False)[numeric_column]
    result = getattr(grouped, operation)()

    label = {"mean": "Average", "sum": "Total", "count": "Count"}[operation]

    return result.sort_values(ascending=False).reset_index(name=label)


# Tool 5: Data filtering
def filter_data(df, column, operator, value):
    if column not in df.columns:
        return None

    series = df[column]

    if operator in {">", ">=", "<", "<="}:
        numeric_series = pd.to_numeric(series, errors="coerce")

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

        return df.loc[conditions[operator]].copy()

    if operator == "equals":
        mask = (
            series.astype(str).str.strip().str.casefold()
            == str(value).strip().casefold()
        )
        return df.loc[mask].copy()

    if operator == "contains":
        mask = series.astype(str).str.contains(
            re.escape(str(value)), case=False, na=False
        )
        return df.loc[mask].copy()

    return None


def parse_filter_question(question, df):
    column = find_column(df.columns.tolist(), question)

    if column is None:
        return None, None, None

    # Match the actual column name, allowing spaces and underscores.
    column_pattern = re.escape(str(column)).replace(r"\ ", r"[\s_]+")
    text = question.strip()

    for operator in (">=", "<=", ">", "<"):
        match = re.search(
            column_pattern + r"\s*" + re.escape(operator)
            + r"\s*(-?\d+(?:\.\d+)?)",
            text,
            flags=re.IGNORECASE,
        )

        if match:
            return column, operator, match.group(1)

    match = re.search(
        column_pattern + r"\s+(?:equals?|equal to|is)\s+(.+?)\s*$",
        text,
        flags=re.IGNORECASE,
    )

    if match:
        return column, "equals", match.group(1).strip().strip("'\"")

    match = re.search(
        column_pattern + r"\s+contains?\s+(.+?)\s*$",
        text,
        flags=re.IGNORECASE,
    )

    if match:
        return column, "contains", match.group(1).strip().strip("'\"")

    return None, None, None


with st.sidebar:
    st.subheader("Display Settings")
    chart_size = st.select_slider(
        "Chart size",
        options=["Compact", "Medium", "Large"],
        value="Compact",
    )
    chart_width = {"Compact": 4, "Medium": 5.5, "Large": 7}[chart_size]
    chart_height = {"Compact": 2.5, "Medium": 3.5, "Large": 4.5}[chart_size]


uploaded_file = st.file_uploader("Upload your dataset", type=["csv"])

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

file_signature = (
    uploaded_file.name,
    uploaded_file.size,
    hash(uploaded_file.getvalue()),
)

if st.session_state.get("file_signature") != file_signature:
    st.session_state.file_signature = file_signature
    st.session_state.chosen_question = ""

numeric_cols = numeric_columns(data)
categorical_cols = [
    column for column in data.columns
    if column not in numeric_cols
]

st.subheader("Dataset Overview")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Records", f"{len(data):,}")
c2.metric("Total Features", len(data.columns))
c3.metric("Missing Values", f"{int(data.isna().sum().sum()):,}")
c4.metric("Duplicate Rows", f"{int(data.duplicated().sum()):,}")

with st.expander("View dataset preview"):
    st.dataframe(data.head(10), use_container_width=True, height=260)


st.subheader("Visual Analytics")
tab1, tab2, tab3 = st.tabs([
    "Numeric Distribution", "Category Breakdown", "Correlation"
])

with tab1:
    if numeric_cols:
        selected_num = st.selectbox(
            "Choose a numeric feature", numeric_cols, key="distribution_column"
        )
        values = pd.to_numeric(data[selected_num], errors="coerce").dropna()

        if not values.empty:
            fig, ax = plt.subplots(figsize=(chart_width, chart_height))
            ax.hist(values, bins=20, edgecolor="white")
            ax.set_title(f"Distribution of {selected_num}")
            ax.set_xlabel(selected_num)
            ax.set_ylabel("Frequency")
            fig.tight_layout()
            st.pyplot(fig, use_container_width=False)
            plt.close(fig)

            st.caption(
                f"Average: {values.mean():.2f} · Median: {values.median():.2f} · "
                f"Minimum: {values.min():.2f} · Maximum: {values.max():.2f}"
            )
        else:
            st.info("This column has no valid numeric values.")
    else:
        st.info("No numeric columns are available.")


with tab2:
    if categorical_cols:
        selected_cat = st.selectbox(
            "Choose a category", categorical_cols, key="category_column"
        )
        top_n = st.slider("Number of categories", 3, 10, 5)
        counts = (
            data[selected_cat].fillna("Missing").astype(str)
            .value_counts().head(top_n)
        )
        chart_choice = st.radio(
            "Chart type", ["Pie Chart", "Bar Chart"], horizontal=True
        )

        fig, ax = plt.subplots(figsize=(chart_width, chart_height))

        if chart_choice == "Pie Chart":
            ax.pie(
                counts.values, labels=counts.index,
                autopct="%1.1f%%", startangle=90,
                textprops={"fontsize": 8},
            )
            ax.set_title(f"Breakdown of {selected_cat}")
            ax.axis("equal")
        else:
            counts.sort_values().plot(kind="barh", ax=ax)
            ax.set_title(f"Most common values in {selected_cat}")
            ax.set_xlabel("Count")
            ax.set_ylabel("")
            ax.tick_params(axis="y", labelsize=8)

        fig.tight_layout()
        st.pyplot(fig, use_container_width=False)
        plt.close(fig)
    else:
        st.info("No categorical columns are available.")


with tab3:
    if len(numeric_cols) >= 2:
        corr_matrix = data[numeric_cols].apply(
            pd.to_numeric, errors="coerce"
        ).corr().round(3)

        fig, ax = plt.subplots(
            figsize=(
                chart_width,
                min(max(chart_height, len(numeric_cols) * 0.35), 8),
            )
        )
        image = ax.imshow(corr_matrix, aspect="auto", vmin=-1, vmax=1)
        ax.set_xticks(range(len(numeric_cols)))
        ax.set_yticks(range(len(numeric_cols)))
        ax.set_xticklabels(numeric_cols, rotation=45, ha="right", fontsize=7)
        ax.set_yticklabels(numeric_cols, fontsize=7)
        ax.set_title("Numeric Feature Correlations")
        fig.colorbar(image, ax=ax, label="Correlation")
        fig.tight_layout()
        st.pyplot(fig, use_container_width=False)
        plt.close(fig)
    else:
        st.info("At least two numeric columns are needed for correlation.")


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
        f"What is the correlation between {numeric_cols[0]} and {numeric_cols[1]}?"
    )

suggestions = list(dict.fromkeys(suggestions))[:6]

for start in range(0, len(suggestions), 2):
    button_cols = st.columns(2)
    for offset, suggestion in enumerate(suggestions[start:start + 2]):
        if button_cols[offset].button(
            suggestion, key=f"suggestion_{start}_{offset}",
            use_container_width=True,
        ):
            st.session_state.chosen_question = suggestion

question = st.text_input(
    "Your question",
    key="chosen_question",
    placeholder="e.g. What is the highest Age?",
)
analyze = st.button("Analyze", type="primary")


if analyze and question.strip():
    q = question.lower().replace("_", " ").strip()
    answer_text = None
    info_text = None
    result_data = None

    if any(term in q for term in ["missing", "null values", "nulls"]):
        result_data = missing_value_analysis(data)
        answer_text = (
            f"Found **{int(data.isna().sum().sum()):,} missing cells** "
            "across the dataset."
        )

    elif "duplicate" in q:
        answer_text = (
            f"This dataset contains **{int(data.duplicated().sum()):,} duplicate rows**."
        )

    # Check filters before other question types.
    elif re.search(r">=|<=|>|<", question) or any(
        term in q for term in [" equals ", " equal to ", " is ", " contains "]
    ):
        col, operator, value = parse_filter_question(question, data)

        if col is None or operator is None:
            info_text = (
                "I couldn't identify the column and condition. Try "
                "`Age > 18`, `Region equals West`, or `Product contains phone`."
            )
        else:
            result_data = filter_data(data, col, operator, value)

            if result_data is None:
                info_text = "Unable to apply this filter. Check the column and value."
            else:
                answer_text = (
                    f"Found **{len(result_data):,} rows** where "
                    f"**{col} {operator} {value}**."
                )

    elif "correlation" in q:
        col_a = find_column(numeric_cols, q)
        col_b = find_column(
            [column for column in numeric_cols if column != col_a], q
        ) if col_a is not None else None

        if col_a is not None and col_b is not None:
            correlation = correlation_analysis(data, col_a, col_b)

            if correlation is None or pd.isna(correlation):
                info_text = "Correlation could not be calculated for these columns."
            else:
                answer_text = (
                    f"Correlation between **{col_a}** and **{col_b}** "
                    f"is **{correlation:.3f}**."
                )
                info_text = "Correlation does not prove causation."
        else:
            result_data = correlation_analysis(data)
            if result_data is not None:
                info_text = "Here is the full numeric correlation matrix."
            else:
                info_text = "Mention two numeric column names."

    elif " by " in q and any(
        word in q for word in ["average", "mean", "total", "sum", "count"]
    ):
        num_col = find_column(numeric_cols, q)
        cat_col = find_column(categorical_cols, q)

        if num_col is not None and cat_col is not None:
            operation = (
                "mean" if any(word in q for word in ["average", "mean"])
                else "sum" if any(word in q for word in ["total", "sum"])
                else "count"
            )
            result_data = groupby_analysis(data, cat_col, num_col, operation)
            answer_text = f"{operation.title()} of **{num_col}** grouped by **{cat_col}**."
        else:
            info_text = "Include a numeric column and a category column, e.g. Average Age by Department."

    elif any(
        word in q for word in [
            "statistics", "describe", "average", "mean", "median",
            "minimum", "maximum", "highest", "lowest", "total", "sum",
        ]
    ):
        col = find_column(numeric_cols, q)

        if col is None:
            info_text = (
                "I couldn't identify a numeric column. Available numeric columns: "
                + (", ".join(map(str, numeric_cols)) if numeric_cols else "none")
            )
        else:
            values = pd.to_numeric(data[col], errors="coerce").dropna()

            if values.empty:
                info_text = f"No valid numeric values were found in **{col}**."
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
                result_data = descriptive_statistics(data, col)
                answer_text = f"Descriptive statistics for **{col}**."

    elif any(word in q for word in ["most common", "common values", "frequency"]):
        col = find_column(data.columns.tolist(), q)

        if col is not None:
            result_data = (
                data[col].value_counts(dropna=False).head(10)
                .rename_axis("Value").reset_index(name="Count")
            )
            answer_text = f"Most common values in **{col}**."
        else:
            info_text = "Please include a column name."

    elif any(word in q for word in ["rows", "columns", "overview", "dataset size"]):
        answer_text = (
            f"This dataset contains **{len(data):,} rows** "
            f"and **{len(data.columns)} columns**."
        )

    else:
        info_text = (
            "Try asking for the highest Age, statistics, missing values, "
            "correlation, average by category, common values, duplicates, "
            "or a filter such as `Age > 18`."
        )

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
        data=data.to_json(orient="records", indent=2).encode("utf-8"),
        file_name="analyzed_data.json",
        mime="application/json",
        use_container_width=True,
    )


with st.expander("About the five analysis tools"):
    st.markdown("""
    **1. Descriptive statistics**
    - Count, mean, standard deviation, quartiles, minimum, maximum.

    **2. Missing-value analysis**
    - Missing cell count and percentage by column.

    **3. Correlation analysis**
    - Pairwise correlation or a numeric correlation matrix.

    **4. Group-by analysis**
    - Calculate average, total, or count by category.

    **5. Data filtering**
    - Numeric comparisons: `>`, `>=`, `<`, `<=`.
    - Text filters: `equals` and `contains`.
    """)

st.divider()
st.caption("Data Analysis Agent · Python, Pandas, Matplotlib, and Streamlit")
