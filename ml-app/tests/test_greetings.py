"""Tests for the greetings endpoints — the minimal worked example in this project.

Testing strategy
----------------
These tests cover the simplest endpoints in the API:
  GET  /api/v1/greetings/hello
  GET  /api/v1/greetings/greet
  POST /api/v1/greetings/greet-and-return

The greetings endpoints have no dependency on the ML pipeline, so both the
`client` fixture (model loaded) and `client_no_model` fixture would work here.
We use `client_no_model` for most tests to show that these endpoints are
independent of the model state — an important property for the "docs and
getting-started" tier of the API.

What these tests verify:
  - Status codes are correct.
  - Response shape matches the declared response_model (GreetingResponse /
    GreetAndReturnResponse) — if the service accidentally changes the field
    name from "message" to "msg", the test catches it.
  - Query parameters are correctly bound.
  - Required query parameters return 422 when missing.
  - The POST body is echoed back correctly.

These tests are intentionally simple — they are the reference point for
students learning how to write FastAPI endpoint tests. Compare the structure
here to test_predict.py and test_drift.py to see how complexity scales.
"""

from __future__ import annotations

from fastapi.testclient import TestClient


class TestHello:
    """GET /api/v1/greetings/hello — generic greeting, no parameters."""

    def test_returns_200(self, client_no_model: TestClient) -> None:
        """The endpoint is always available — no model required."""
        response = client_no_model.get("/api/v1/greetings/hello")
        assert response.status_code == 200

    def test_response_has_message_field(self, client_no_model: TestClient) -> None:
        """Response must match the GreetingResponse schema: {"message": "..."}."""
        data = client_no_model.get("/api/v1/greetings/hello").json()
        assert "message" in data
        assert isinstance(data["message"], str)

    def test_message_is_non_empty(self, client_no_model: TestClient) -> None:
        """A greeting with an empty string is not useful."""
        data = client_no_model.get("/api/v1/greetings/hello").json()
        assert len(data["message"]) > 0


class TestGreet:
    """GET /api/v1/greetings/greet — personalised greeting via query parameter."""

    def test_returns_200(self, client_no_model: TestClient) -> None:
        response = client_no_model.get("/api/v1/greetings/greet", params={"name": "Alice"})
        assert response.status_code == 200

    def test_response_contains_name(self, client_no_model: TestClient) -> None:
        """The greeting message must include the name provided in the query parameter.

        This tests the contract between the query parameter binding and the
        service layer: `name` must flow all the way from the URL to the response.
        """
        data = client_no_model.get("/api/v1/greetings/greet", params={"name": "Alice"}).json()
        assert "Alice" in data["message"]

    def test_response_has_message_field(self, client_no_model: TestClient) -> None:
        """Response must match the GreetingResponse schema."""
        data = client_no_model.get("/api/v1/greetings/greet", params={"name": "Bob"}).json()
        assert "message" in data
        assert isinstance(data["message"], str)

    def test_missing_name_returns_422(self, client_no_model: TestClient) -> None:
        """The `name` query parameter is required — omitting it must return 422.

        FastAPI validates required parameters before calling the endpoint
        handler. 422 (Unprocessable Entity) tells the client to fix their
        request, not to retry or escalate.
        """
        response = client_no_model.get("/api/v1/greetings/greet")
        assert response.status_code == 422

    def test_different_names_produce_different_messages(self, client_no_model: TestClient) -> None:
        """Two different names must produce two different greetings."""
        msg_alice = client_no_model.get("/api/v1/greetings/greet", params={"name": "Alice"}).json()["message"]
        msg_bob = client_no_model.get("/api/v1/greetings/greet", params={"name": "Bob"}).json()["message"]
        assert msg_alice != msg_bob


class TestGreetAndReturn:
    """POST /api/v1/greetings/greet-and-return — greeting + body echo.

    This endpoint demonstrates combining a query parameter with a JSON request
    body. It is the most complex of the greeting endpoints, but still simpler
    than any of the churn endpoints.
    """

    def test_returns_200(self, client_no_model: TestClient) -> None:
        response = client_no_model.post(
            "/api/v1/greetings/greet-and-return",
            params={"name": "Alice"},
            json={"plan": "premium"},
        )
        assert response.status_code == 200

    def test_response_has_message_and_data(self, client_no_model: TestClient) -> None:
        """Response must match GreetAndReturnResponse: {"message": "...", "data": {...}}."""
        data = client_no_model.post(
            "/api/v1/greetings/greet-and-return",
            params={"name": "Alice"},
            json={"plan": "premium"},
        ).json()
        assert "message" in data
        assert "data" in data

    def test_message_includes_name(self, client_no_model: TestClient) -> None:
        """The name from the query parameter must appear in the response message."""
        data = client_no_model.post(
            "/api/v1/greetings/greet-and-return",
            params={"name": "Alice"},
            json={"plan": "premium"},
        ).json()
        assert "Alice" in data["message"]

    def test_data_is_echoed_correctly(self, client_no_model: TestClient) -> None:
        """The JSON body must be echoed back in the `data` field unchanged.

        This is the key contract of greet-and-return: any JSON object sent
        as the body is returned as-is under the `data` key.
        """
        payload = {"plan": "premium", "active": True, "score": 99}
        response_data = client_no_model.post(
            "/api/v1/greetings/greet-and-return",
            params={"name": "Alice"},
            json=payload,
        ).json()
        assert response_data["data"] == payload

    def test_missing_name_returns_422(self, client_no_model: TestClient) -> None:
        """The `name` query parameter is required — omitting it must return 422."""
        response = client_no_model.post(
            "/api/v1/greetings/greet-and-return",
            json={"plan": "premium"},
        )
        assert response.status_code == 422

    def test_empty_body_is_valid(self, client_no_model: TestClient) -> None:
        """An empty JSON object is a valid body — the echo should return {}."""
        data = client_no_model.post(
            "/api/v1/greetings/greet-and-return",
            params={"name": "Alice"},
            json={},
        ).json()
        assert data["data"] == {}
