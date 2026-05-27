from pydantic import BaseModel


class GuardVerdict(BaseModel):
    verdict: str
    reason: str
