from pydantic import BaseModel, Field, field_validator

MAX_REQUEST_MESSAGE_LEN = 4096


class ChatRequest(BaseModel):
    thread_id: str = Field(max_length=128)
    message: str = Field(max_length=MAX_REQUEST_MESSAGE_LEN)
    approve: bool | None = None  # None = new turn; True/False = resume interrupt

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
    dead_letter: dict | None = None


class CheckpointInfo(BaseModel):
    checkpoint_id: str
    step: int
    source: str
    next: list[str]
    status: str | None
    messages_count: int


class ReplayRequest(BaseModel):
    checkpoint_id: str

    @field_validator("checkpoint_id")
    @classmethod
    def checkpoint_id_must_not_be_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("checkpoint_id must not be blank")
        return v
