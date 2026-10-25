
# Data Analysis Agent

A Python-based tool-using agent for analyzing tabular datasets and CSV files using natural-language questions.

**Live Demo:**
https://data-analysis-agent-fotc9jjmaromuzh5dbznjz.streamlit.app/

## Features

The agent provides five analysis tools:

1. **Data Profiling** — summarizes dataset dimensions, column types, missing values, and duplicate rows.
2. **Statistical Analysis** — calculates count, mean, median, minimum, maximum, and standard deviation for numeric columns.
3. **Row Filtering** — filters rows using numeric comparison operators.
4. **Group Analysis** — aggregates numeric values by a categorical column.
5. **Correlation Analysis** — calculates Pearson correlation between numeric columns.

## How It Works

The agent uses rule-based question routing to select an appropriate analysis tool. It uses pandas and NumPy to process data and supports CSV uploads through a Streamlit interface.

This is a basic tool-using agent, not an LLM-powered system. Tool selection depends on supported question patterns and column names.

## Tech Stack

- Python
- pandas
- NumPy
- Streamlit

## Run Locally

1. Clone or download this repository.
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Launch the app:

   ```bash
   streamlit run app.py
   ```

4. Upload a CSV file, enter a question, and click **Analyse**.

## Sample Dataset

The repository includes `sample_data.csv` for trying the tools.

## Limitations

- Question routing is rule-based rather than powered by a language model.
- Filtering currently focuses on numeric comparisons.
- Results depend on the available columns and supported question patterns.
- Correlation does not imply causation.

## Future Improvements

- Add more robust question interpretation.
- Support additional filtering and aggregation operations.
- Add clearer validation and error handling for different datasets.
- Add automated tests.
