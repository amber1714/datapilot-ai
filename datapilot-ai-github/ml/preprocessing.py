<<<<<<< HEAD
# Data preprocessing utilities

=======
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler


def prepare_dataset(
    df: pd.DataFrame,
    target_column: str,
    problem_type: str,
    test_size: float = 0.2,
    random_state: int = 42,
):
    working_df = df.copy()
    working_df = working_df.dropna(subset=[target_column])

    X = working_df.drop(columns=[target_column])
    y_original = working_df[target_column]

    target_encoder = None

    if problem_type == "Classification":
        target_encoder = LabelEncoder()
        y = pd.Series(
            target_encoder.fit_transform(y_original.astype(str)),
            index=y_original.index,
            name=target_column,
        )
    else:
        y = y_original

    numerical_columns = X.select_dtypes(include="number").columns.tolist()
    categorical_columns = X.select_dtypes(exclude="number").columns.tolist()

    numerical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    transformers = []

    if numerical_columns:
        transformers.append(
            ("numerical", numerical_pipeline, numerical_columns)
        )

    if categorical_columns:
        transformers.append(
            ("categorical", categorical_pipeline, categorical_columns)
        )

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )

    stratify_value = None

    if problem_type == "Classification":
        class_counts = y.value_counts()
        if len(class_counts) > 1 and class_counts.min() >= 2:
            stratify_value = y

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_value,
    )

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    try:
        feature_names = preprocessor.get_feature_names_out().tolist()
    except Exception:
        feature_names = [
            f"feature_{index}"
            for index in range(X_train_processed.shape[1])
        ]

    return {
        "X": X,
        "y": y,
        "y_original": y_original,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "X_train_processed": X_train_processed,
        "X_test_processed": X_test_processed,
        "numerical_columns": numerical_columns,
        "categorical_columns": categorical_columns,
        "feature_names": feature_names,
        "preprocessor": preprocessor,
        "target_encoder": target_encoder,
    }
>>>>>>> 85d9c89 (Update DataPilot AI through Step 12)
