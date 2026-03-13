"""API v1 router — aggregates all endpoint routers under /api/v1.

To add a new ML service (e.g. "customer segmentation") to this API:
  1. Create app/api/v1/endpoints/segmentation/ with an __init__.py and router.
  2. Add a service layer at app/services/segmentation/.
  3. Add schemas at app/schemas/segmentation.py.
  4. Import the router here and include it with a prefix and tag.

This "big-API" layout scales to dozens of services while keeping each one
independently testable and easy to locate.
"""

from fastapi import APIRouter

from app.api.v1.endpoints.churn import drift, predict
from app.api.v1.endpoints.greetings import hello

router = APIRouter(prefix="/api/v1")

# Greetings — a minimal example for students getting started.
router.include_router(hello.router, prefix="/greetings", tags=["greetings"])

# Churn — inference endpoints (always available; only churn-lib[predict] required).
router.include_router(predict.router, prefix="/churn", tags=["churn — predict"])
router.include_router(drift.router, prefix="/churn", tags=["churn — drift"])

# Training endpoints — only registered when churn-lib[train] is installed.
#
# The try/except gates the import: if mlflow or matplotlib are missing,
# ImportError is raised here and the training routes simply don't exist.
# The inference process is unaffected — it never imports the heavy deps.
#
# To convert to Option B (two separate services):
#   1. Move app/api/v1/endpoints/churn/train.py into a new FastAPI project.
#   2. Deploy it as a separate Docker container on an internal network.
#   3. Remove this block — the inference API stays exactly as-is.
try:
    from app.api.v1.endpoints.churn import train

    router.include_router(train.router, prefix="/churn", tags=["churn — train"])
except ImportError:
    pass  # churn-lib[train] not installed; training endpoints disabled.
