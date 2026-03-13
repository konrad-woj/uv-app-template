"""Greeting endpoints — a minimal worked example for students.

This module demonstrates the simplest possible FastAPI pattern:
  - An APIRouter with a tag for Swagger grouping.
  - Thin endpoint functions that delegate all logic to a service module.
  - Query parameters declared as plain function arguments (no Pydantic needed
    for simple scalar inputs).

There are intentionally no Pydantic request models here because the inputs are
simple strings. Once inputs become structured objects, add a schema in
app/schemas/ and use it as a function parameter with type annotation — FastAPI
will parse and validate it automatically.

See app/services/greeter.py for the business logic layer.
"""

from fastapi import APIRouter

from app.services.greeter import greet, say_hello

router = APIRouter()


@router.get("/hello")
async def hello_endpoint():
    """Return a generic greeting from the Greeter service."""
    return say_hello()


@router.get("/greet")
async def greet_endpoint(name: str):
    """Return a personalised greeting.

    Args:
        name: The name to greet — passed as a query parameter (?name=Alice).
    """
    return greet(name)


@router.post("/greet-and-return")
async def greet_with_data_endpoint(name: str, data: dict):
    """Greet by name and echo back any JSON body.

    Demonstrates combining a query parameter with a request body in one endpoint.
    FastAPI parses the JSON body into `data` automatically.
    """
    greeting = greet(name)
    return {
        "message": greeting["message"] + " Thanks for sending me this data.",
        "data": data,
    }
