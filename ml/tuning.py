import optuna

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import f1_score, r2_score
from xgboost import XGBClassifier, XGBRegressor


def tune_classification_model(
    model_name,
    X_train,
    y_train,
    X_valid,
    y_valid,
    n_trials=15,
):
    if model_name == "Random Forest":
        def objective(trial):
            model = RandomForestClassifier(
                n_estimators=trial.suggest_int("n_estimators", 100, 400),
                max_depth=trial.suggest_int("max_depth", 2, 12),
                min_samples_split=trial.suggest_int("min_samples_split", 2, 10),
                min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 5),
                random_state=42,
                n_jobs=-1,
            )
            model.fit(X_train, y_train)
            predictions = model.predict(X_valid)
            average = "binary" if len(set(y_valid)) == 2 else "weighted"
            return f1_score(
                y_valid,
                predictions,
                average=average,
                zero_division=0,
            )

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        best_model = RandomForestClassifier(
            **study.best_params,
            random_state=42,
            n_jobs=-1,
        )
        best_model.fit(X_train, y_train)

        return best_model, study.best_params, study.best_value

    if model_name == "XGBoost":
        def objective(trial):
            model = XGBClassifier(
                n_estimators=trial.suggest_int("n_estimators", 100, 400),
                learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                max_depth=trial.suggest_int("max_depth", 2, 8),
                subsample=trial.suggest_float("subsample", 0.6, 1.0),
                colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
                random_state=42,
                eval_metric="logloss",
                n_jobs=1,
            )
            model.fit(X_train, y_train)
            predictions = model.predict(X_valid)
            average = "binary" if len(set(y_valid)) == 2 else "weighted"
            return f1_score(
                y_valid,
                predictions,
                average=average,
                zero_division=0,
            )

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        best_model = XGBClassifier(
            **study.best_params,
            random_state=42,
            eval_metric="logloss",
            n_jobs=1,
        )
        best_model.fit(X_train, y_train)

        return best_model, study.best_params, study.best_value

    raise ValueError(f"Tuning is not supported for {model_name}.")


def tune_regression_model(
    model_name,
    X_train,
    y_train,
    X_valid,
    y_valid,
    n_trials=15,
):
    if model_name == "Random Forest Regressor":
        def objective(trial):
            model = RandomForestRegressor(
                n_estimators=trial.suggest_int("n_estimators", 100, 400),
                max_depth=trial.suggest_int("max_depth", 2, 12),
                min_samples_split=trial.suggest_int("min_samples_split", 2, 10),
                min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 5),
                random_state=42,
                n_jobs=-1,
            )
            model.fit(X_train, y_train)
            predictions = model.predict(X_valid)
            return r2_score(y_valid, predictions)

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        best_model = RandomForestRegressor(
            **study.best_params,
            random_state=42,
            n_jobs=-1,
        )
        best_model.fit(X_train, y_train)

        return best_model, study.best_params, study.best_value

    if model_name == "XGBoost Regressor":
        def objective(trial):
            model = XGBRegressor(
                n_estimators=trial.suggest_int("n_estimators", 100, 400),
                learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                max_depth=trial.suggest_int("max_depth", 2, 8),
                subsample=trial.suggest_float("subsample", 0.6, 1.0),
                colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
                random_state=42,
                objective="reg:squarederror",
                n_jobs=1,
            )
            model.fit(X_train, y_train)
            predictions = model.predict(X_valid)
            return r2_score(y_valid, predictions)

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        best_model = XGBRegressor(
            **study.best_params,
            random_state=42,
            objective="reg:squarederror",
            n_jobs=1,
        )
        best_model.fit(X_train, y_train)

        return best_model, study.best_params, study.best_value

    raise ValueError(f"Tuning is not supported for {model_name}.")
