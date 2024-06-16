# -*- coding: utf-8 -*-
import pytest
import pandas as pd
from model_utils import make_inference, load_model
from sklearn.pipeline import Pipeline
from pickle import dumps


@pytest.fixture
def create_data() -> dict[str, float]:
    return {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4,
            "petal_width": 0.2}


def test_make_inference(monkeypatch, create_data):
    def mock_get_predictions(_, data: pd.DataFrame) -> list[int]:
        assert create_data == {
            key: value[0] for key, value in data.to_dict("list").items()
        }
        return [0]  # Возвращаем список целых чисел

    in_model = Pipeline([])
    monkeypatch.setattr(Pipeline, "predict", mock_get_predictions)

    result = make_inference(in_model, create_data)
    assert result == {"species": "setosa"}


@pytest.fixture()
def filepath_and_data(tmpdir):
    p = tmpdir.mkdir("datadir").join("fakedmodel.pkl")
    example: Pipeline = Pipeline([])
    p.write_binary(dumps(example))
    return str(p), example


def test_load_model(filepath_and_data):
    assert isinstance(load_model(filepath_and_data[0]), Pipeline)
