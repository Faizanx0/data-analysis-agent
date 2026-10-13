
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(
    page_title="CSV Data Analysis Agent",
    page_icon="📊",
    layout="wide"
)

st.title("📊 CSV Data Analysis Agent")
st.write(
    "Upload a CSV file, explore your data, "
    "and ask questions about it."
)

uploaded_file = st.file_uploader(
    "Upload your CSV file",
    type=["csv"]
)

if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)

        st.success("CSV uploaded successfully!")

        st.subheader("Dataset Overview")
        col1, col2, col3 = st.columns(3)

        col1.metric("Rows", data.shape[0])
        col2.metric("Columns", data.shape[1])
        col3.metric("Missing Values", int(data.isna().sum().sum()))

        st.subheader("Preview")
        st.dataframe(data.head(20), use_container_width=True)

        st.subheader("Column Names")
        st.write(data.columns.tolist())

        st.subheader("Numeric Summary")
        numeric_cols = data.select_dtypes(
            include="number"
        ).columns.tolist()

        if numeric_cols:
            st.dataframe(
                data[numeric_cols].describe().round(2),
                use_container_width=True
            )
        else:
            st.info("No numeric columns were detected.")

        st.subheader("Explore a Numeric Column")

        if numeric_cols:
            selected_column = st.selectbox(
                "Choose a column",
                numeric_cols
            )

            fig, ax = plt.subplots()
            ax.hist(
                data[selected_column].dropna(),
                bins=20
            )
            ax.set_title(f"Distribution of {selected_column}")
            ax.set_xlabel(selected_column)
            ax.set_ylabel("Count")

            st.pyplot(fig)
            plt.close(fig)

        st.subheader("Suggested Questions")

        questions = [
            "How many rows and columns are in the dataset?",
            "Are there any missing values?",
        ]

        for col in numeric_cols[:3]:
            questions.append(
                f"What are the statistics for {col}?"
            )

        for col in data.select_dtypes(
            include=["object", "category", "bool"]
        ).columns[:2]:
            questions.append(
                f"What are the most common values in {col}?"
            )

        for question in questions:
            st.write("•", question)

        st.subheader("Ask Your Dataset")

        user_question = st.text_input(
            "Enter your question",
            placeholder="Example: What are the statistics for Age?"
        )

        if st.button("Analyze") and user_question:
            q = user_question.lower()

            if "missing" in q:
                report = pd.DataFrame({
                    "Missing Count": data.isna().sum(),
                    "Missing Percentage": (
                        data.isna().mean() * 100
                    ).round(2)
                })

                st.write("Missing Value Report")
                st.dataframe(report, use_container_width=True)

            elif any(word in q for word in [
                "statistics", "average", "mean", "median",
                "describe", "minimum", "maximum"
            ]):
                matched_column = next(
                    (
                        c for c in numeric_cols
                        if c.lower() in q
                    ),
                    None
                )

                if matched_column:
                    st.write(f"Statistics for {matched_column}")
                    st.write(
                        data[matched_column].describe().round(2)
                    )
                elif numeric_cols:
                    st.info(
                        "Please mention a numeric column name. "
                        "Available columns: "
                        + ", ".join(numeric_cols)
                    )
                else:
                    st.info("No numeric columns are available.")

            elif "row" in q or "column" in q or "overview" in q:
                st.write({
                    "Rows": data.shape[0],
                    "Columns": data.shape[1],
                    "Column names": data.columns.tolist()
                })

            elif "most common" in q or "common values" in q:
                categorical_cols = data.select_dtypes(
                    include=["object", "category", "bool"]
                ).columns.tolist()

                matched_column = next(
                    (
                        c for c in categorical_cols
                        if c.lower() in q
                    ),
                    None
                )

                if matched_column:
                    st.write(
                        data[matched_column]
                        .value_counts(dropna=False)
                        .head(10)
                    )
                elif categorical_cols:
                    st.info(
                        "Please mention a categorical column name: "
                        + ", ".join(categorical_cols)
                    )
                else:
                    st.info("No categorical columns are available.")

            else:
                st.info(
                    "I don't recognize that question yet. "
                    "Try asking about missing values, statistics, "
                    "common values, or the dataset overview."
                )

        st.subheader("Download Data")

        csv_data = data.to_csv(index=False).encode("utf-8")

        st.download_button(
            "Download CSV",
            data=csv_data,
            file_name="analyzed_data.csv",
            mime="text/csv"
        )

        json_data = data.to_json(
            orient="records",
            indent=2
        )

        st.download_button(
            "Download JSON",
            data=json_data,
            file_name="analyzed_data.json",
            mime="application/json"
        )

    except Exception as error:
        st.error(f"Could not read this CSV: {error}")

else:
    st.info("Upload a CSV file to begin your analysis.")
