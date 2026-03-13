"""FastAPI dependency functions shared across all routers.

A dependency is a callable that FastAPI resolves once per request and injects
as a parameter. Keeping shared dependencies here — rather than inline in each
endpoint — means they can be reused, overridden in tests, and inspected in
one place.

FastAPI's DI system gives you two benefits over direct imports:
  1. Testability — override Depends(get_pipeline) with a fake in tests without
     patching globals or restarting the app.
  2. Explicit contracts — endpoint signatures declare their requirements clearly,
     making it easy to understand what each handler needs at a glance.

Usage in any endpoint:

    from fastapi import Depends
    from app.core.dependencies import get_pipeline, get_settings

    @router.post("/predict")
    async def predict(
        body: PredictRequest,
        pipeline: ChurnPipeline = Depends(get_pipeline),
        settings: Settings = Depends(get_settings),
    ):
        ...

FastAPI resolves Depends(get_pipeline) before calling the endpoint function,
so by the time the handler runs, `pipeline` is guaranteed to be a loaded
ChurnPipeline or the request has already been rejected with a 503.
"""

from churn_lib import ChurnPipeline
from fastapi import HTTPException, Request, status

from app.core.config import Settings
from app.core.config import settings as _settings


def get_pipeline(request: Request) -> ChurnPipeline:
    """Return the in-memory ChurnPipeline, or raise 503 if no model is loaded.

    The pipeline is stored on app.state at two moments:
      1. Startup — if CHURN_MODEL_PATH points to an existing pipeline.joblib.
      2. After a successful POST /api/v1/churn/train — the new model is
         hot-reloaded into app.state so predictions use it immediately.

    Raises:
        HTTPException(503): When app.state.pipeline is None.
    """
    pipeline: ChurnPipeline | None = getattr(request.app.state, "pipeline", None)
    if pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "No model is loaded. "
                "Either set CHURN_MODEL_PATH to a trained pipeline.joblib and restart, "
                "or POST /api/v1/churn/train to train one on demand."
            ),
        )
    return pipeline


def get_settings() -> Settings:
    """Return the application settings singleton.

    Using Depends(get_settings) instead of importing `settings` directly makes
    endpoints testable — you can override this dependency in tests to inject
    any Settings object without touching environment variables or globals.

    Example override in a test:

        def override_settings():
            return Settings(default_threshold=0.3)

        app.dependency_overrides[get_settings] = override_settings

    Returns:
        The module-level Settings singleton loaded from env vars / .env file.
    """
    return _settings
