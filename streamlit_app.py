import io

import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="DataPilot AI",
    page_icon="",
    layout="wide",
)

st.title("DataPilot AI")
st.caption("Autonomous Data Science Assistant — Step 1: Dataset Upload & Analysis")

uploaded_file = st.file_uploader(
    "Upload a CSV file",
    type=["csv"],
    help="Upload a CSV dataset to inspect its structure and data quality.",
)

if uploaded_file is None:
    st.info("Upload a CSV file to begin.")
    st.stop()

try:
    df = pd.read_csv(uploaded_file)
except UnicodeDecodeError:
    try:
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file, encoding="latin-1")
    except Exception as exc:
        st.error(f"Could not read the CSV file: {exc}")
        st.stop()
except pd.errors.EmptyDataError:
    st.error("The uploaded CSV file is empty.")
    st.stop()
except pd.errors.ParserError as exc:
    st.error(f"The CSV file could not be parsed: {exc}")
    st.stop()
except Exception as exc:
    st.error(f"Unexpected error while reading the file: {exc}")
    st.stop()

if df.empty:
    st.warning("The dataset was loaded, but it contains no rows.")
    st.stop()

st.success("Dataset loaded successfully.")

rows, columns = df.shape
missing_total = int(df.isna().sum().sum())
duplicate_rows = int(df.duplicated().sum())

metric_1, metric_2, metric_3, metric_4 = st.columns(4)
metric_1.metric("Rows", f"{rows:,}")
metric_2.metric("Columns", f"{columns:,}")
metric_3.metric("Missing Values", f"{missing_total:,}")
metric_4.metric("Duplicate Rows", f"{duplicate_rows:,}")

st.divider()

st.subheader("Dataset Preview")
preview_rows = st.slider(
    "Number of rows to preview",
    min_value=5,
    max_value=min(50, rows),
    value=min(10, rows),
)
st.dataframe(df.head(preview_rows), use_container_width=True)

st.divider()

st.subheader("Column Information")
column_info = pd.DataFrame(
    {
        "Column": df.columns,
        "Data Type": [str(dtype) for dtype in df.dtypes],
        "Non-Null Count": [int(df[col].notna().sum()) for col in df.columns],
        "Missing Count": [int(df[col].isna().sum()) for col in df.columns],
        "Missing %": [
            round(float(df[col].isna().mean() * 100), 2) for col in df.columns
        ],
        "Unique Values": [int(df[col].nunique(dropna=True)) for col in df.columns],
    }
)
st.dataframe(column_info, use_container_width=True)

st.divider()

st.subheader("Missing Values")
missing_summary = (
    df.isna()
    .sum()
    .rename("Missing Count")
    .to_frame()
)
missing_summary["Missing %"] = (
    missing_summary["Missing Count"] / len(df) * 100
).round(2)
missing_summary = missing_summary[missing_summary["Missing Count"] > 0]

if missing_summary.empty:
    st.success("No missing values were found.")
else:
    st.dataframe(missing_summary, use_container_width=True)

st.divider()

st.subheader("Duplicate Rows")
if duplicate_rows == 0:
    st.success("No duplicate rows were found.")
else:
    st.warning(f"{duplicate_rows} duplicate row(s) were found.")
    if st.checkbox("Show duplicate rows"):
        st.dataframe(
            df[df.duplicated(keep=False)],
            use_container_width=True,
        )

st.divider()

st.subheader("Basic Statistics")

numeric_columns = df.select_dtypes(include="number").columns.tolist()
categorical_columns = df.select_dtypes(exclude="number").columns.tolist()

tab_numeric, tab_categorical = st.tabs(
    ["Numerical Columns", "Categorical Columns"]
)

with tab_numeric:
    if not numeric_columns:
        st.info("No numerical columns were detected.")
    else:
        numeric_stats = df[numeric_columns].describe().T
        st.dataframe(numeric_stats, use_container_width=True)

with tab_categorical:
    if not categorical_columns:
        st.info("No categorical/text columns were detected.")
    else:
        categorical_stats = df[categorical_columns].describe().T
        st.dataframe(categorical_stats, use_container_width=True)

st.divider()

st.subheader("Dataset Summary")
summary = {
    "Rows": rows,
    "Columns": columns,
    "Numerical Columns": len(numeric_columns),
    "Categorical Columns": len(categorical_columns),
    "Missing Values": missing_total,
    "Duplicate Rows": duplicate_rows,
    "Memory Usage (MB)": round(
        df.memory_usage(deep=True).sum() / (1024 ** 2),
        3,
    ),
}

summary_df = pd.DataFrame(
    {"Metric": summary.keys(), "Value": summary.values()}
)
st.dataframe(summary_df, hide_index=True, use_container_width=True)

st.divider()

buffer = io.StringIO()
df.info(buf=buffer)
with st.expander("Show pandas DataFrame info"):
    st.code(buffer.getvalue())

st.success("Step 1 complete: dataset upload and basic analysis are working.")
