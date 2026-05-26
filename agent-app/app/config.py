"""Application-wide settings loaded from environment variables or a .env file.

Pydantic-settings reads env vars with the AGENT_ prefix automatically.

Example .env:
    AGENT_DB_URI=postgresql://postgres:postgres@localhost:5433/langgraph
    AGENT_LLM_MODEL=openai/unsloth/Qwen3.6-35B-A3B-UD-MLX-4bit
    AGENT_LLM_BASE_URL=http://127.0.0.1:8888/v1
    AGENT_LLM_API_KEY=your-api-key-here
    AGENT_LLM_THINKING=false
    AGENT_LLM_TIMEOUT_SECONDS=60
    AGENT_LLM_MAX_RETRIES=3
    AGENT_MCP_SERVER_URL=http://localhost:8001
    AGENT_MAX_REFLECTION_ATTEMPTS=5
    AGENT_MAX_REACT_STEPS=10
    LOG_LEVEL=INFO
    LOG_ENV=production
"""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="AGENT_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    db_uri: str = Field(
        default="postgresql://postgres:postgres@localhost:5433/langgraph",
        description="PostgreSQL connection string for LangGraph checkpointer.",
    )
    llm_model: str = Field(
        default="openai/unsloth/Qwen3.6-35B-A3B-UD-MLX-4bit",
        description="LiteLLM model identifier (e.g. 'openai/unsloth/Qwen3.6-35B-A3B-UD-MLX-4bit', 'gpt-5-mini').",
    )
    llm_base_url: str = Field(
        default="http://127.0.0.1:8888/v1",
        description="Base URL for the LLM provider (Unsloth Studio OpenAI-compatible endpoint).",
    )
    llm_api_key: str | None = Field(
        default=None,
        description="API key passed to the LLM provider. Works for any backend: Unsloth Studio, Ollama, Gemini, OpenAI, etc.",
    )
    llm_thinking: bool = Field(
        default=False,
        description="Enable extended thinking mode. False by default; set to true to activate chain-of-thought.",
    )
    llm_timeout_seconds: float = Field(
        default=60.0,
        description="Per-call timeout in seconds for LLM invocations. asyncio.TimeoutError is translated to LLMServiceUnavailableError.",
    )
    llm_max_retries: int = Field(
        default=3,
        description="Maximum retries for transient LLM errors (rate limit, 5xx). Uses exponential backoff capped at 30s.",
    )
    mcp_server_url: str = Field(
        default="http://localhost:8001",
        description="URL of the fastmcp tool server.",
    )
    max_reflection_attempts: int = Field(
        default=5,
        description="Hard ceiling on reflection critic/refiner iterations. Loop exits on quality pass or when this limit is reached.",
    )
    max_react_steps: int = Field(
        default=10,
        description="Hard ceiling on ReAct tool-call iterations. Routes to writer when reached even if model still emits tool_calls.",
    )
    max_pipeline_steps: int = Field(
        default=50,
        description=(
            "LangGraph recursion_limit: maximum supersteps across the entire pipeline per invocation. "
            "Worst-case for this graph is ~26 (react=10 + reflection=10 + ~6 other nodes); "
            "50 is generous for normal runs while still bounding runaway execution."
        ),
    )


settings = Settings()
