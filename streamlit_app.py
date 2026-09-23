import os

import pandas as pd
import streamlit as st

from services.api_client import (
    DataPilotAPIError,
    analyze_dataset,
    detect_problem,
    health_check,
    train_models,
)


st.set_page_config(
    page_title="DataPilot AI",
    page_icon="📊",
    layout="wide",
)

API_URL = os.getenv(
    "DATAPILOT_API_URL",
    "http://127.0.0.1:8000",
)

st.title("DataPilot AI")
st.caption(
    "Autonomous data science assistant for dataset analysis and machine-learning model comparison."
)

with st.sidebar:
    st.subheader("API Status")

    st.code(API_URL)

    if st.button("Check API Connection"):
        try:
            health = health_check(API_URL)
            st.success(
                f"Connected: {health['status']}"
            )
        except DataPilotAPIError as exc:
            st.error(str(exc))

uploaded_file = st.file_uploader(
    "Upload a CSV file",
    type=["csv"],
)

if uploaded_file is None:
    st.info(
        "Upload a CSV file to begin. DataPilot will analyze it through the FastAPI backend."
    )
    st.stop()

file_bytes = uploaded_file.getvalue()
file_name = uploaded_file.name

st.divider()
st.subheader("1. Analyze Dataset")

try:
    analysis = analyze_dataset(
        file_name=file_name,
        file_bytes=file_bytes,
        api_url=API_URL,
    )
except DataPilotAPIError as exc:
    st.error(str(exc))
    st.stop()

st.success(
    "Dataset analyzed successfully."
)

metric_1, metric_2, metric_3, metric_4 = st.columns(4)

metric_1.metric(
    "Rows",
    analysis["rows"],
)

metric_2.metric(
    "Columns",
    analysis["columns"],
)

metric_3.metric(
    "Missing Values",
    analysis["missing_values"],
)

metric_4.metric(
    "Duplicate Rows",
    analysis["duplicate_rows"],
)

st.write("Dataset Preview")

preview_df = pd.DataFrame(
    analysis["preview"]
)

st.dataframe(
    preview_df,
    use_container_width=True,
)

st.write("Column Information")

column_df = pd.DataFrame(
    analysis["column_details"]
)

st.dataframe(
    column_df,
    hide_index=True,
    use_container_width=True,
)

column_names = [
    item["name"]
    for item in analysis["column_details"]
]

st.divider()
st.subheader("2. Choose Prediction Target")

target_column = st.selectbox(
    "Choose the column you want to predict",
    options=column_names,
    index=None,
    placeholder="Select target column",
)

if target_column is None:
    st.info(
        "Select a target column to continue."
    )
    st.stop()

try:
    detection = detect_problem(
        file_name=file_name,
        file_bytes=file_bytes,
        target_column=target_column,
        api_url=API_URL,
    )
except DataPilotAPIError as exc:
    st.error(str(exc))
    st.stop()

detect_1, detect_2, detect_3 = st.columns(3)

detect_1.metric(
    "Problem Type",
    detection["problem_type"],
)

detect_2.metric(
    "Unique Target Values",
    detection["unique_values"],
)

detect_3.metric(
    "Missing Target Values",
    detection["missing_values"],
)

st.divider()
st.subheader("3. Train & Compare Models")

test_size = st.slider(
    "Test set size",
    min_value=0.10,
    max_value=0.40,
    value=0.20,
    step=0.05,
)

if st.button(
    "Train & Compare Models",
    type="primary",
):
    with st.spinner(
        "Training and comparing machine-learning models..."
    ):
        try:
            training = train_models(
                file_name=file_name,
                file_bytes=file_bytes,
                target_column=target_column,
                test_size=test_size,
                api_url=API_URL,
            )
        except DataPilotAPIError as exc:
            st.error(str(exc))
            st.stop()

    st.success(
        "Training completed successfully."
    )

    result_1, result_2, result_3 = st.columns(3)

    result_1.metric(
        "Best Model",
        training["best_model"],
    )

    result_2.metric(
        "Training Rows",
        training["training_rows"],
    )

    result_3.metric(
        "Testing Rows",
        training["testing_rows"],
    )

    st.metric(
        "Processed Features",
        training["processed_features"],
    )

    st.write("Models Trained")

    st.dataframe(
        pd.DataFrame(
            {
                "Model": training[
                    "models_trained"
                ]
            }
        ),
        hide_index=True,
        use_container_width=True,
    )

    training_errors = training.get(
        "training_errors",
        {},
    )

    if training_errors:
        st.warning(
            "Some models could not be trained."
        )

        for model_name, error_message in training_errors.items():
            st.error(
                f"{model_name}: {error_message}"
            )

    st.write("Model Comparison")

    results_df = pd.DataFrame(
        training["results"]
    )

    st.dataframe(
        results_df,
        hide_index=True,
        use_container_width=True,
    )

    csv_bytes = (
        results_df
        .to_csv(index=False)
        .encode("utf-8")
    )

    st.download_button(
        "Download API Training Results",
        data=csv_bytes,
        file_name=(
            "datapilot_api_training_results.csv"
        ),
        mime="text/csv",
    )

    st.success(
        "Model comparison complete. Download the results or try another target column."
    )

st.divider()

with st.expander(
    "System Architecture"
):
    st.code(
        """
Browser
   |
   v
Streamlit frontend
   |
   | HTTPS requests
   v
FastAPI backend
   |
   v
Preprocessing + ML models
   |
   v
JSON response
   |
   v
Streamlit results
        """
    )
