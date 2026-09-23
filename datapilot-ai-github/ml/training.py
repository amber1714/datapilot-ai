<<<<<<< HEAD
# Model training utilities

=======
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression

try:
    from xgboost import XGBClassifier, XGBRegressor
    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False


def get_models(problem_type: str):
    models = {}

    if problem_type == "Classification":
        models["Logistic Regression"] = LogisticRegression(
            max_iter=1000,
            random_state=42,
        )

        models["Random Forest"] = RandomForestClassifier(
            n_estimators=200,
            random_state=42,
        )

        if XGBOOST_AVAILABLE:
            models["XGBoost"] = XGBClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=4,
                random_state=42,
                eval_metric="logloss",
                n_jobs=1,
            )

    elif problem_type == "Regression":
        models["Linear Regression"] = LinearRegression()

        models["Random Forest Regressor"] = RandomForestRegressor(
            n_estimators=200,
            random_state=42,
        )

        if XGBOOST_AVAILABLE:
            models["XGBoost Regressor"] = XGBRegressor(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=4,
                random_state=42,
                objective="reg:squarederror",
                n_jobs=1,
            )

    return models


def train_models(models, X_train, y_train):
    trained_models = {}
    training_errors = {}

    for name, model in models.items():
        try:
            model.fit(X_train, y_train)
            trained_models[name] = model
        except Exception as exc:
            training_errors[name] = str(exc)

    return trained_models, training_errors
>>>>>>> 85d9c89 (Update DataPilot AI through Step 12)
