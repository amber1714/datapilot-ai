import pandas as pd


def detect_problem_type(target: pd.Series) -> str:
    clean_target = target.dropna()

    if clean_target.empty:
        return "Unknown"

    unique_count = clean_target.nunique()

    if (
        pd.api.types.is_object_dtype(clean_target)
        or pd.api.types.is_bool_dtype(clean_target)
        or isinstance(clean_target.dtype, pd.CategoricalDtype)
    ):
        return "Classification"

    if pd.api.types.is_integer_dtype(clean_target):
        threshold = max(10, int(len(clean_target) * 0.05))
        if unique_count <= threshold:
            return "Classification"

    if pd.api.types.is_numeric_dtype(clean_target):
        return "Regression"

    return "Classification"


def get_problem_details(target: pd.Series) -> dict:
    clean_target = target.dropna()

    return {
        "problem_type": detect_problem_type(target),
        "target_dtype": str(target.dtype),
        "unique_values": int(clean_target.nunique()),
        "missing_values": int(target.isna().sum()),
        "total_values": int(len(target)),
    }
