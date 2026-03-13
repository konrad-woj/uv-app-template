"""Unit tests for app/core/dependencies.py.

Testing strategy
----------------
`get_pipeline` is the gateway between every model-dependent endpoint and
app.state. Testing it in isolation from endpoints means we're testing the
dependency contract, not the endpoint behaviour — a subtle but important
distinction when onboarding students to FastAPI's DI system.

We construct a minimal Request/Application pair directly, which is simpler
and faster than spinning up a TestClient for this single concern.

`get_settings` is a trivial wrapper around the singleton and is not tested
here — its testability is demonstrated by the pattern of overriding it in
endpoint tests (see test_predict.py).
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from churn_lib import ChurnPipeline
from fastapi import HTTPException
from starlette.applications import Starlette
from starlette.requests import Request

from app.core.dependencies import get_pipeline


def _make_request(pipeline_value) -> Request:
    """Build a minimal Request whose app.state.pipeline is set to pipeline_value.

    This avoids spinning up the full FastAPI app with its lifespan hooks.
    """
    app = Starlette()
    app.state.pipeline = pipeline_value
    # Starlette's Request needs a minimal ASGI scope.
    scope = {"type": "http", "app": app}
    return Request(scope)


class TestGetPipeline:
    """get_pipeline() dependency — unit tests."""

    def test_returns_pipeline_when_loaded(self) -> None:
        """When app.state.pipeline is set, get_pipeline() must return it."""
        mock_pipeline = MagicMock(spec=ChurnPipeline)
        request = _make_request(mock_pipeline)

        result = get_pipeline(request)

        assert result is mock_pipeline

    def test_raises_503_when_pipeline_is_none(self) -> None:
        """When app.state.pipeline is None, get_pipeline() must raise HTTP 503.

        503 (Service Unavailable) is correct here because the situation is
        temporary — the model will be loaded eventually. The endpoint does not
        need to handle this; FastAPI catches the HTTPException and returns
        the 503 response before calling the endpoint function.
        """
        request = _make_request(None)

        with pytest.raises(HTTPException) as exc_info:
            get_pipeline(request)

        assert exc_info.value.status_code == 503

    def test_503_detail_is_actionable(self) -> None:
        """The 503 detail must tell operators how to fix the missing-model condition."""
        request = _make_request(None)

        with pytest.raises(HTTPException) as exc_info:
            get_pipeline(request)

        detail = exc_info.value.detail.lower()
        assert "model" in detail

    def test_raises_503_when_pipeline_attribute_missing(self) -> None:
        """If app.state has no pipeline attribute at all, get_pipeline() must still raise 503.

        `getattr(request.app.state, 'pipeline', None)` handles a freshly
        created app.state that has never had pipeline set — defensive coding
        that prevents AttributeError from becoming a 500.
        """
        app = Starlette()
        # Deliberately do NOT set app.state.pipeline
        scope = {"type": "http", "app": app}
        request = Request(scope)

        with pytest.raises(HTTPException) as exc_info:
            get_pipeline(request)

        assert exc_info.value.status_code == 503
