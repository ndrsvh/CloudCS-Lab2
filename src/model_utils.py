# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.pipeline import Pipeline
from pickle import load


def make_inference(in_model: Pipeline, in_data: dict) -> dict[str, str]:
    """Return the result of predictions for in_data using in_model."""
    predict = int(in_model.predict(pd.DataFrame(in_data, index=[0]))[0])
    if predict == 0:
        return {"species": "setosa"}
    if predict == 1:
        return {"species": "versicolor"}
    if predict == 2:
        return {"species": "virginica"}


def load_model(path: str) -> Pipeline:
    """Return the model being read which stored on the path."""
    with open(path, "rb") as file:
        model: Pipeline = load(file)

    return model
