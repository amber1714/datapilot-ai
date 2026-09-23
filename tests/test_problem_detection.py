import pandas as pd
from ml.problem_detection import detect_problem_type

def test_detects_classification():
    target = pd.Series(["Yes", "No", "Yes", "No"])
    assert detect_problem_type(target) == "Classification"

def test_detects_regression():
    target = pd.Series([10.5,11.2,12.8,13.7,14.1,15.9,16.4,17.6,18.2,19.5,20.1,21.3])
    assert detect_problem_type(target) == "Regression"
