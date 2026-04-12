"""Greeting endpoints — a minimal worked example for students.

This module demonstrates the simplest possible FastAPI pattern:
  - An APIRouter with a tag for Swagger grouping.
  - Thin endpoint functions that delegate all logic to a service module.
  - Query parameters declared as plain function arguments (no Pydantic needed
    for simple scalar inputs).
  - response_model declared on every route so FastAPI validates the response
    shape and generates accurate OpenAPI docs.

There are intentionally no Pydantic *request* models for the simple text
parameters (name). Once request bodies become structured objects, add a schema
in app/schemas/ and use it as a function parameter with a type annotation —
FastAPI will parse and validate it automatically. See app/schemas/churn.py and
the churn endpoints for the full pattern.

See app/services/greeter.py for the business logic layer.
"""

from fastapi import APIRouter

from app.schemas.greetings import GreetAndReturnResponse, GreetingResponse
from app.services.greeter import greet, say_hello

router = APIRouter()


@router.get("/hello", response_model=GreetingResponse, summary="Return a generic greeting")
async def hello_endpoint() -> GreetingResponse:
    """Return a generic greeting from the Greeter service.

    This is the simplest possible endpoint:
      - No path parameters, no query parameters, no request body.
      - Delegates all logic to the service layer.
      - Returns a typed response validated against GreetingResponse.

    Start here when learning the codebase, then follow the pattern into
    the churn endpoints for a real-world ML service example.
    """
    return say_hello()


@router.get("/greet", response_model=GreetingResponse, summary="Return a personalised greeting")
async def greet_endpoint(name: str) -> GreetingResponse:
    """Return a personalised greeting.

    Demonstrates query parameters: FastAPI maps simple function arguments
    (str, int, float, bool) to query parameters automatically — no decorator
    or annotation needed.

    Args:
        name: The name to greet — passed as a query parameter (?name=Alice).

    Example:
        GET /api/v1/greetings/greet?name=Alice
        → {"message": "Welcome, Alice! This is a greeting from the Greeter service."}
    """
    return greet(name)


@router.post(
    "/greet-and-return",
    response_model=GreetAndReturnResponse,
    summary="Return a greeting and echo the request body",
)
async def greet_with_data_endpoint(name: str, data: dict) -> GreetAndReturnResponse:
    """Greet by name and echo back any JSON body.

    Demonstrates combining a query parameter with a request body in one endpoint:
      - `name` (str) → query parameter (?name=Alice).
      - `data` (dict) → request body (any JSON object).

    FastAPI infers the source of each parameter from its type:
      - Simple scalars (str, int, float, bool) → query / path parameters.
      - Complex types (dict, Pydantic models) → request body.

    In a production service you would replace `data: dict` with a typed Pydantic
    model (e.g. `body: MyRequest`) so OpenAPI can document the expected structure
    and Pydantic can validate it. See app/schemas/churn.py for the full pattern.

    Example:
        POST /api/v1/greetings/greet-and-return?name=Alice
        Body: {"plan": "premium", "active": true}
        → {"message": "Welcome, Alice! ... Thanks for sending me this data.",
           "data": {"plan": "premium", "active": true}}
    """
    greeting = greet(name)
    return GreetAndReturnResponse(
        message=greeting["message"] + " Thanks for sending me this data.",
        data=data,
    )
