# -*- coding: utf-8 -*-
import pytest
from fastapi.testclient import TestClient
from keycloak.uma_permissions import AuthStatus
from typing import Tuple, Any
import requests_mock


URL = "https://keycloak:8443/realms/inference/protocol/openid-connect/token"


@pytest.fixture
def init_test_client(monkeypatch) -> TestClient:
    def mock_make_inference(*args, **kwargs) -> dict[str, float]:
        return {"species": "setosa"}

    def mock_load_model(*args, **kwargs) -> None:
        return None

    def mock_keycloak_openid(*args, **kwargs) -> Any:
        class FakedKeycloakOpenID:
            @staticmethod
            def well_known(*args, **kwargs):
                return {"token_endpoint": "fakedendpoint"}

            @staticmethod
            def has_uma_access(token: str, *args, **kwargs) -> AuthStatus:
                if token == "Ok":
                    return AuthStatus(True, True, set())
                elif token == "Not_logged":
                    return AuthStatus(False, False, set())
                elif token == "Not_authorized":
                    return AuthStatus(True, False, set())
                else:
                    return AuthStatus(False, False, set())
        return FakedKeycloakOpenID

    monkeypatch.setenv("MODEL_PATH", "faked/model.pkl")
    monkeypatch.setenv("KEYCLOAK_URL", "fakeurl")
    monkeypatch.setenv("CLIENT_ID", "fakeid")
    monkeypatch.setenv("CLIENT_SECRET", "fakesecret")
    monkeypatch.setattr("model_utils.make_inference", mock_make_inference)
    monkeypatch.setattr("model_utils.load_model", mock_load_model)
    monkeypatch.setattr("keycloak.KeycloakOpenID", mock_keycloak_openid)

    from main import app
    return TestClient(app)


def test_healthcheck(init_test_client) -> None:
    response = init_test_client.get("/healthcheck")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predictions_success(init_test_client, monkeypatch):
    with requests_mock.Mocker() as m:
        m.post(URL, json={'access_token': 'Ok'}, status_code=200)
        response = init_test_client.post(
            "/predictions",
            json={
                "instance": {
                    "sepal_length": 5.5,
                    "sepal_width": 3.6,
                    "petal_length": 1.4,
                    "petal_width": 0.2
                },
                "client": {
                    "client_id": "test_client_id",
                    "client_secret": "test_client_secret"
                }
            }
        )
    assert response.status_code == 200
    assert "species" in response.json()


def test_not_logged(init_test_client, monkeypatch):
    with requests_mock.Mocker() as m:
        m.post(URL, json={'access_token': 'Not_logged'}, status_code=200)
        response = init_test_client.post(
            "/predictions",
            json={
                "instance": {
                    "sepal_length": 5.5,
                    "sepal_width": 3.6,
                    "petal_length": 1.4,
                    "petal_width": 0.2
                },
                "client": {
                    "client_id": "test_client_id",
                    "client_secret": "test_client_secret"
                }
            }
        )
    assert response.status_code == 401
    assert response.json() == {
        "detail": "Invalid authentication credentials"
    }


def test_not_authorized(init_test_client, monkeypatch):
    with requests_mock.Mocker() as m:
        m.post(URL, json={'access_token': 'Not_authorized'}, status_code=200)
        response = init_test_client.post(
            "/predictions",
            json={
                "instance": {
                    "sepal_length": 5.5,
                    "sepal_width": 3.6,
                    "petal_length": 1.4,
                    "petal_width": 0.2
                },
                "client": {
                    "client_id": "test_client_id",
                    "client_secret": "test_client_secret"
                }
            }
        )
    assert response.status_code == 403
    assert response.json() == {
        "detail": "Access denied"
    }


def test_inference(init_test_client, monkeypatch):
    with requests_mock.Mocker() as m:
        m.post(URL, json={'access_token': 'Ok'}, status_code=200)
        response = init_test_client.post(
            "/predictions",
            json={
                "instance": {
                    "sepal_length": 5.5,
                    "sepal_width": 3.6,
                    "petal_length": 1.4,
                    "petal_width": 0.2
                },
                "client": {
                    "client_id": "test_client_id",
                    "client_secret": "test_client_secret"
                }
            }
        )
    assert response.status_code == 200
    assert response.json()["species"] == "setosa"
