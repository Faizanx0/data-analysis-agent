import re
import numpy as np
import pandas as pd


def profile_data(data):
    return {
        "rows": len(data),
        "columns": list(data.columns),
        "data_types": data.dtypes.astype(str).to_dict(),
        "missing_values": data.isna().sum().to_dict(),
        "duplicate_rows": int(data.duplicated().sum()),
    }


def statistical_analysis(data, column):
    if column not in data.columns:
        return {"error": f"Unknown column: {column}"}
    if not pd.api.types.is_numeric_dtype(data[column]):
        return {"error": f"{column} is not numeric."}
    s = data[column].dropna()
    if s.empty:
        return {"error": "No numeric values available."}
    return {
        "count": int(s.count()),
        "mean": float(s.mean()),
        "median": float(s.median()),
        "min": float(s.min()),
        "max": float(s.max()),
        "std": float(s.std()) if len(s) > 1 else 0.0,
    }


def filter_rows(data, column, operator, value):
    if column not in data.columns:
        return {"error": f"Unknown column: {column}"}
    ops = {
        ">": lambda s: s > value,
        "<": lambda s: s < value,
        ">=": lambda s: s >= value,
        "<=": lambda s: s <= value,
    }
    if operator not in ops:
        return {"error": "Unsupported operator."}
    try:
        s = pd.to_numeric(data[column], errors="raise")
        result = data.loc[ops[operator](s)]
        return {
            "matching_rows": len(result),
            "rows": result.head(20).to_dict(orient="records"),
            "truncated": len(result) > 20,
        }
    except (ValueError, TypeError):
        return {"error": "Filtering requires a numeric column and value."}


def group_analysis(data, group_column, value_column, operation="sum"):
    if group_column not in data.columns or value_column not in data.columns:
        return {"error": "Unknown column."}
    if not pd.api.types.is_numeric_dtype(data[value_column]):
        return {"error": f"{value_column} must be numeric."}
    if operation not in {"sum", "mean", "count", "min", "max", "median"}:
        return {"error": "Unsupported aggregation."}
    result = data.groupby(group_column, dropna=False)[value_column].agg(operation)
    return result.sort_values(ascending=False).to_dict()


def correlation_analysis(data, column_a, column_b):
    for col in (column_a, column_b):
        if col not in data.columns:
            return {"error": f"Unknown column: {col}"}
        if not pd.api.types.is_numeric_dtype(data[col]):
            return {"error": f"{col} must be numeric."}
    valid = data[[column_a, column_b]].dropna()
    if len(valid) < 2:
        return {"error": "At least two complete rows are required."}
    return {
        "columns": [column_a, column_b],
        "pearson_correlation": float(valid[column_a].corr(valid[column_b])),
        "rows_used": len(valid),
        "note": "Correlation does not imply causation.",
    }


TOOLS = {
    "Dataset Profiler": profile_data,
    "Statistical Analysis": statistical_analysis,
    "Filter Rows": filter_rows,
    "Group-by Analysis": group_analysis,
    "Correlation Analysis": correlation_analysis,
}


def ask_agent(question, data):
    """Rule-based planner that selects a real analysis tool."""
    q = question.lower()
    numeric = data.select_dtypes(include=np.number).columns.tolist()
    categories = [
        c for c in data.columns
        if not pd.api.types.is_numeric_dtype(data[c])
    ]

    def mentioned(columns):
        return next((c for c in columns if c.lower() in q), None)

    if any(w in q for w in ("correlation", "correlate", "relationship")):
        a = mentioned(numeric)
        b = next((c for c in numeric if c != a and c.lower() in q), None)
        if a is None:
            a = numeric[0] if numeric else None
        if b is None:
            b = next((c for c in numeric if c != a), None)
        if a and b:
            return "Correlation Analysis", correlation_analysis(data, a, b)
        return "Correlation Analysis", {"error": "Need two numeric columns."}

    if any(w in q for w in ("group", "breakdown", "by category",
                            "by region", "by product", "compare categories",
                            "highest by", "total by")):
        g = mentioned(categories) or (categories[0] if categories else None)
        v = mentioned(numeric) or (numeric[0] if numeric else None)
        op = "mean" if any(w in q for w in ("average", "mean")) else "sum"
        if g and v:
            return "Group-by Analysis", group_analysis(data, g, v, op)
        return "Group-by Analysis", {"error": "Need category and numeric columns."}

    if any(w in q for w in ("filter", "above", "below", "greater than",
                            "less than", "more than")):
        col = mentioned(numeric)
        match = re.search(
            r"(above|below|greater than|less than|more than|>=|<=|>|<)"
            r"\s*(-?\d+(?:\.\d+)?)", q
        )
        if col and match:
            phrase, val = match.groups()
            op = "<" if phrase in ("below", "less than", "<") else ">"
            if phrase in (">=", "<="):
                op = phrase
            return "Filter Rows", filter_rows(data, col, op, float(val))
        return "Filter Rows", {
            "error": "Include a numeric column and threshold, e.g. filter revenue above 100."
        }

    if any(w in q for w in ("average", "mean", "median", "statistics",
                            "stats", "minimum", "maximum", "standard deviation",
                            "standard deviation", "describe")):
        col = mentioned(numeric) or (numeric[0] if numeric else None)
        if col:
            return "Statistical Analysis", statistical_analysis(data, col)
        return "Statistical Analysis", {"error": "No numeric columns found."}

    return "Dataset Profiler", profile_data(data)
