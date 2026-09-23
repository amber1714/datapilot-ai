
# DataPilot AI - FastAPI backend

=======
import io

import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from ml.evaluation import (
    evaluate_classification_models,
    evaluate_regression_models,
)
from ml.preprocessing import prepare_dataset
from ml.problem_detection import get_problem_details
from ml.training import get_models, train_models


app = FastAPI(
    title="DataPilot AI API",
    description=(
        "Backend API for automated dataset analysis, "
        "preprocessing, model training, and evaluation."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def read_csv_upload(file_bytes: bytes) -> pd.DataFrame:
    try:
        return pd.read_csv(io.BytesIO(file_bytes))
    except UnicodeDecodeError:
        return pd.read_csv(
            io.BytesIO(file_bytes),
            encoding="latin-1",
        )
    except pd.errors.EmptyDataError as exc:
        raise HTTPException(
            status_code=400,
            detail="The uploaded CSV file is empty.",
        ) from exc
    except pd.errors.ParserError as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Could not parse CSV: {exc}",
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Could not read CSV: {exc}",
        ) from exc


@app.get("/")
def root():
    return {
        "name": "DataPilot AI API",
        "status": "running",
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "datapilot-ai-api",
    }


@app.post("/analyze")
async def analyze_dataset(
    file: UploadFile = File(...),
):
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail="Only CSV files are supported.",
        )

    df = read_csv_upload(
        await file.read()
    )

    if df.empty:
        raise HTTPException(
            status_code=400,
            detail="The dataset contains no rows.",
        )

    column_details = []

    for column in df.columns:
        column_details.append(
            {
                "name": column,
                "dtype": str(
                    df[column].dtype
                ),
                "missing_count": int(
                    df[column]
                    .isna()
                    .sum()
                ),
                "missing_percent": round(
                    float(
                        df[column]
                        .isna()
                        .mean()
                        * 100
                    ),
                    2,
                ),
                "unique_values": int(
                    df[column]
                    .nunique(
                        dropna=True
                    )
                ),
            }
        )

    return {
        "filename": file.filename,
        "rows": int(
            df.shape[0]
        ),
        "columns": int(
            df.shape[1]
        ),
        "missing_values": int(
            df.isna()
            .sum()
            .sum()
        ),
        "duplicate_rows": int(
            df.duplicated()
            .sum()
        ),
        "column_details": (
            column_details
        ),
        "preview": (
            df.head(10)
            .fillna("")
            .to_dict(
                orient="records"
            )
        ),
    }


@app.post("/detect-problem")
async def detect_problem(
    target_column: str = Form(...),
    file: UploadFile = File(...),
):
    df = read_csv_upload(
        await file.read()
    )

    if target_column not in df.columns:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Target column "
                f"'{target_column}' "
                f"was not found."
            ),
        )

    return {
        "target_column": (
            target_column
        ),
        **get_problem_details(
            df[target_column]
        ),
    }


@app.post("/train")
async def train_models_endpoint(
    target_column: str = Form(...),
    test_size: float = Form(0.2),
    file: UploadFile = File(...),
):
    if not 0.10 <= test_size <= 0.40:
        raise HTTPException(
            status_code=400,
            detail=(
                "test_size must be "
                "between 0.10 and 0.40."
            ),
        )

    df = read_csv_upload(
        await file.read()
    )

    if target_column not in df.columns:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Target column "
                f"'{target_column}' "
                f"was not found."
            ),
        )

    problem_type = (
        get_problem_details(
            df[target_column]
        )["problem_type"]
    )

    if problem_type not in {
        "Classification",
        "Regression",
    }:
        raise HTTPException(
            status_code=400,
            detail=(
                "Could not determine "
                "a supported problem type."
            ),
        )

    try:
        prepared = prepare_dataset(
            df=df,
            target_column=target_column,
            problem_type=problem_type,
            test_size=test_size,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Preprocessing failed: "
                f"{exc}"
            ),
        ) from exc

    models = get_models(
        problem_type
    )

    trained_models, training_errors = (
        train_models(
            models=models,
            X_train=prepared[
                "X_train_processed"
            ],
            y_train=prepared[
                "y_train"
            ],
        )
    )

    if not trained_models:
        raise HTTPException(
            status_code=500,
            detail={
                "message": (
                    "No model could be trained."
                ),
                "training_errors": (
                    training_errors
                ),
            },
        )

    if problem_type == "Classification":
        results = (
            evaluate_classification_models(
                trained_models,
                prepared[
                    "X_test_processed"
                ],
                prepared[
                    "y_test"
                ],
            )
        )
        primary_metric = "F1 Score"
    else:
        results = (
            evaluate_regression_models(
                trained_models,
                prepared[
                    "X_test_processed"
                ],
                prepared[
                    "y_test"
                ],
            )
        )
        primary_metric = "R²"

    return {
        "filename": file.filename,
        "target_column": target_column,
        "problem_type": problem_type,
        "test_size": test_size,
        "training_rows": int(
            len(
                prepared[
                    "X_train"
                ]
            )
        ),
        "testing_rows": int(
            len(
                prepared[
                    "X_test"
                ]
            )
        ),
        "processed_features": int(
            prepared[
                "X_train_processed"
            ].shape[1]
        ),
        "models_trained": list(
            trained_models.keys()
        ),
        "training_errors": (
            training_errors
        ),
        "best_model": str(
            results.iloc[0][
                "Model"
            ]
        ),
        "primary_metric": (
            primary_metric
        ),
        "results": (
            results.where(
                pd.notnull(results),
                None,
            )
            .to_dict(
                orient="records"
            )
        ),
    }

