import joblib


def save_pipeline_artifact(
    file_path,
    preprocessor,
    model,
    target_encoder,
    target_column,
    problem_type,
    raw_feature_columns,
):
    artifact = {
        "preprocessor": preprocessor,
        "model": model,
        "target_encoder": target_encoder,
        "target_column": target_column,
        "problem_type": problem_type,
        "raw_feature_columns": raw_feature_columns,
    }

    joblib.dump(
        artifact,
        file_path,
    )


def load_pipeline_artifact(file_path):
    return joblib.load(file_path)
