import json
import sqlite3
from datetime import datetime
from pathlib import Path

import pandas as pd


DEFAULT_DB_PATH = Path("datapilot.db")


def get_connection(db_path=DEFAULT_DB_PATH):
    connection = sqlite3.connect(
        db_path,
        check_same_thread=False,
    )
    connection.row_factory = sqlite3.Row
    return connection


def initialize_database(db_path=DEFAULT_DB_PATH):
    with get_connection(db_path) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS experiments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                dataset_name TEXT NOT NULL,
                target_column TEXT NOT NULL,
                problem_type TEXT NOT NULL,
                test_size REAL NOT NULL,
                training_rows INTEGER NOT NULL,
                testing_rows INTEGER NOT NULL,
                processed_features INTEGER NOT NULL,
                models_trained TEXT NOT NULL,
                best_model TEXT NOT NULL,
                primary_metric_name TEXT NOT NULL,
                primary_metric_value REAL,
                metrics_json TEXT NOT NULL
            )
            """
        )
        connection.commit()


def save_experiment(
    dataset_name,
    target_column,
    problem_type,
    test_size,
    training_rows,
    testing_rows,
    processed_features,
    models_trained,
    best_model,
    primary_metric_name,
    primary_metric_value,
    results_df,
    db_path=DEFAULT_DB_PATH,
):
    initialize_database(db_path)

    metrics_json = results_df.to_json(
        orient="records"
    )

    with get_connection(db_path) as connection:
        cursor = connection.execute(
            """
            INSERT INTO experiments (
                created_at,
                dataset_name,
                target_column,
                problem_type,
                test_size,
                training_rows,
                testing_rows,
                processed_features,
                models_trained,
                best_model,
                primary_metric_name,
                primary_metric_value,
                metrics_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.now().isoformat(
                    timespec="seconds"
                ),
                dataset_name,
                target_column,
                problem_type,
                float(test_size),
                int(training_rows),
                int(testing_rows),
                int(processed_features),
                json.dumps(models_trained),
                best_model,
                primary_metric_name,
                (
                    float(primary_metric_value)
                    if primary_metric_value is not None
                    else None
                ),
                metrics_json,
            ),
        )
        connection.commit()
        return cursor.lastrowid


def get_experiments(db_path=DEFAULT_DB_PATH):
    initialize_database(db_path)

    with get_connection(db_path) as connection:
        rows = connection.execute(
            """
            SELECT
                id,
                created_at,
                dataset_name,
                target_column,
                problem_type,
                test_size,
                training_rows,
                testing_rows,
                processed_features,
                models_trained,
                best_model,
                primary_metric_name,
                primary_metric_value,
                metrics_json
            FROM experiments
            ORDER BY id DESC
            """
        ).fetchall()

    return [dict(row) for row in rows]


def experiments_dataframe(db_path=DEFAULT_DB_PATH):
    rows = get_experiments(db_path)

    if not rows:
        return pd.DataFrame(
            columns=[
                "Experiment",
                "Date",
                "Dataset",
                "Target",
                "Problem",
                "Best Model",
                "Metric",
                "Score",
                "Test Size",
                "Features",
            ]
        )

    records = []

    for row in rows:
        records.append(
            {
                "Experiment": f"EXP-{row['id']:03d}",
                "Date": row["created_at"],
                "Dataset": row["dataset_name"],
                "Target": row["target_column"],
                "Problem": row["problem_type"],
                "Best Model": row["best_model"],
                "Metric": row["primary_metric_name"],
                "Score": (
                    round(
                        row["primary_metric_value"],
                        4,
                    )
                    if row["primary_metric_value"]
                    is not None
                    else None
                ),
                "Test Size": row["test_size"],
                "Features": row[
                    "processed_features"
                ],
            }
        )

    return pd.DataFrame(records)


def get_experiment_details(
    experiment_id,
    db_path=DEFAULT_DB_PATH,
):
    initialize_database(db_path)

    with get_connection(db_path) as connection:
        row = connection.execute(
            """
            SELECT *
            FROM experiments
            WHERE id = ?
            """,
            (int(experiment_id),),
        ).fetchone()

    if row is None:
        return None

    result = dict(row)
    result["models_trained"] = json.loads(
        result["models_trained"]
    )
    result["metrics"] = json.loads(
        result["metrics_json"]
    )
    return result
