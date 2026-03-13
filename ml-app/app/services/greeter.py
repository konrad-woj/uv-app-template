"""Greeter service — a minimal example of the service layer pattern.

The service layer sits between the HTTP router and any external dependency
(database, ML model, third-party API). Its job is pure business logic with no
HTTP concepts — no Request, no Response, no status codes.

Benefits:
  - Endpoint handlers stay thin (they only translate HTTP ↔ service calls).
  - Service functions can be unit-tested without starting a FastAPI app.
  - Swapping the transport layer (HTTP → gRPC, CLI → API) only requires
    changing the router, not this file.
"""


def say_hello() -> dict:
    """Return a generic greeting from the Greeter service."""
    return {"message": "Hello! This is a greeting from the Greeter service."}


def greet(name: str) -> dict:
    """Return a personalised greeting for the given name."""
    return {"message": f"Welcome, {name}! This is a greeting from the Greeter service."}
