import pandas as pd
from ml.preprocessing import prepare_dataset
from ml.problem_detection import detect_problem_type
from ml.training import get_models, train_models

def test_basic_classification_training():
    df = pd.DataFrame({
        "Age":[21,22,23,24,25,26,27,28,29,30,31,32],
        "Hours":[2,3,4,5,1,6,4,3,5,2,6,3],
        "Department":["AI","CSE","AI","ECE","CSE","AI","ECE","CSE","AI","ECE","AI","CSE"],
        "Passed":["Yes","Yes","Yes","Yes","No","Yes","Yes","No","Yes","No","Yes","Yes"],
    })
    problem_type = detect_problem_type(df["Passed"])
    prepared = prepare_dataset(
        df=df,
        target_column="Passed",
        problem_type=problem_type,
        test_size=0.25,
        random_state=42,
    )
    models = get_models(problem_type)
    models = {"Logistic Regression": models["Logistic Regression"]}
    trained, errors = train_models(
        models=models,
        X_train=prepared["X_train_processed"],
        y_train=prepared["y_train"],
    )
    assert "Logistic Regression" in trained
    assert errors == {}
