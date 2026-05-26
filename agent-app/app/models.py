from pydantic import BaseModel, field_validator


class ChatRequest(BaseModel):
    thread_id: str
    message: str

    @field_validator("message")
    @classmethod
    def message_must_not_be_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("message must not be blank")
        return v

    @field_validator("thread_id")
    @classmethod
    def thread_id_must_not_be_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("thread_id must not be blank")
        return v


class ChatResponse(BaseModel):
    thread_id: str
    status: str
    is_interrupted: bool = False
    interrupt_value: dict | None = None
    final_answer: str | None = None
    guard_reason: str | None = None


class CheckpointInfo(BaseModel):
    checkpoint_id: str
    step: int
    source: str
    next: list[str]
    status: str | None
    messages_count: int


class ReplayRequest(BaseModel):
    checkpoint_id: str
