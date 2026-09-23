
# Model evaluation utilities

=======
import math

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def evaluate_classification_models(
    trained_models,
    X_test,
    y_test,
):
    rows = []
    class_count = int(pd.Series(y_test).nunique())

    for name, model in trained_models.items():
        predictions = model.predict(X_test)

        average_type = "binary" if class_count == 2 else "weighted"

        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(
            y_test,
            predictions,
            average=average_type,
            zero_division=0,
        )
        recall = recall_score(
            y_test,
            predictions,
            average=average_type,
            zero_division=0,
        )
        f1 = f1_score(
            y_test,
            predictions,
            average=average_type,
            zero_division=0,
        )

        roc_auc = None

        try:
            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(X_test)

                if class_count == 2:
                    roc_auc = roc_auc_score(
                        y_test,
                        probabilities[:, 1],
                    )
                else:
                    roc_auc = roc_auc_score(
                        y_test,
                        probabilities,
                        multi_class="ovr",
                        average="weighted",
                    )
        except Exception:
            roc_auc = None

        rows.append(
            {
                "Model": name,
                "Accuracy": round(accuracy, 4),
                "Precision": round(precision, 4),
                "Recall": round(recall, 4),
                "F1 Score": round(f1, 4),
                "ROC-AUC": (
                    round(roc_auc, 4)
                    if roc_auc is not None
                    else None
                ),
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values("F1 Score", ascending=False)
        .reset_index(drop=True)
    )


def evaluate_regression_models(
    trained_models,
    X_test,
    y_test,
):
    rows = []

    for name, model in trained_models.items():
        predictions = model.predict(X_test)

        mae = mean_absolute_error(y_test, predictions)
        mse = mean_squared_error(y_test, predictions)
        rmse = math.sqrt(mse)
        r2 = r2_score(y_test, predictions)

        rows.append(
            {
                "Model": name,
                "MAE": round(mae, 4),
                "MSE": round(mse, 4),
                "RMSE": round(rmse, 4),
                "R²": round(r2, 4),
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values("R²", ascending=False)
        .reset_index(drop=True)
    )


def get_confusion_matrix(model, X_test, y_test):
    predictions = model.predict(X_test)
    return confusion_matrix(y_test, predictions)


def get_binary_roc_data(model, X_test, y_test):
    if not hasattr(model, "predict_proba"):
        return None

    probabilities = model.predict_proba(X_test)

    if probabilities.shape[1] != 2:
        return None

    fpr, tpr, thresholds = roc_curve(
        y_test,
        probabilities[:, 1],
    )

    auc_value = roc_auc_score(
        y_test,
        probabilities[:, 1],
    )

    return {
        "fpr": fpr,
        "tpr": tpr,
        "thresholds": thresholds,
        "auc": auc_value,
    }


def get_feature_importance(model, feature_names):
    values = None

    if hasattr(model, "feature_importances_"):
        values = model.feature_importances_

    elif hasattr(model, "coef_"):
        coefficients = np.asarray(model.coef_)

        if coefficients.ndim == 2:
            values = np.mean(
                np.abs(coefficients),
                axis=0,
            )
        else:
            values = np.abs(coefficients)

    if values is None:
        return None

    count = min(
        len(values),
        len(feature_names),
    )

    importance = pd.DataFrame(
        {
            "Feature": feature_names[:count],
            "Importance": values[:count],
        }
    )

    return (
        importance
        .sort_values(
            "Importance",
            ascending=False,
        )
        .reset_index(drop=True)
    )


def get_regression_predictions(model, X_test, y_test):
    predictions = model.predict(X_test)

    result = pd.DataFrame(
        {
            "Actual": np.asarray(y_test),
            "Predicted": predictions,
        }
    )

    result["Residual"] = (
        result["Actual"] - result["Predicted"]
    )

    return result

