"""Pydantic schemas for greetings endpoints.

These are deliberately simple — one field each — to demonstrate the pattern
without burying it in complexity. Real domains (churn, segmentation) use the
same pattern with richer fields; see app/schemas/churn.py.

Why bother with a schema for a single string field?
  - FastAPI auto-generates OpenAPI docs from the schema.
  - The schema enforces the response contract: endpoint code that accidentally
    returns {"msg": ...} instead of {"message": ...} fails at the Pydantic layer
    before it reaches the client.
  - Tests can import the schema and validate shape without hardcoding field names.
"""

from pydantic import BaseModel, Field


class GreetingResponse(BaseModel):
    """A generic text greeting."""

    message: str = Field(description="Human-readable greeting string.")


class GreetAndReturnResponse(BaseModel):
    """A personalised greeting combined with an echo of the request body.

    Demonstrates how to return heterogeneous data (text + arbitrary JSON)
    from a single endpoint. In a real service you would replace `data: dict`
    with a typed Pydantic model so OpenAPI can document the expected shape.
    """

    message: str = Field(description="Personalised greeting string.")
    data: dict = Field(description="Echo of the JSON body sent with the request.")
