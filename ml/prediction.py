import pandas as pd


def build_prediction_frame(
    input_values,
    raw_feature_columns,
):
    row = {
        column: input_values.get(column)
        for column in raw_feature_columns
    }

    return pd.DataFrame(
        [row],
        columns=raw_feature_columns,
    )


def make_prediction(
    artifact,
    input_frame,
):
    preprocessor = artifact["preprocessor"]
    model = artifact["model"]
    target_encoder = artifact["target_encoder"]
    problem_type = artifact["problem_type"]

    processed = preprocessor.transform(
        input_frame
    )

    prediction = model.predict(
        processed
    )[0]

    if (
        problem_type == "Classification"
        and target_encoder is not None
    ):
        label = target_encoder.inverse_transform(
            [int(prediction)]
        )[0]

        probability = None

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(
                processed
            )[0]

            probability = float(
                max(probabilities)
            )

        return {
            "prediction": label,
            "probability": probability,
        }

    return {
        "prediction": float(prediction),
        "probability": None,
    }
