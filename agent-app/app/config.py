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
    AGENT_MCP_SERVER_URL=http://localhost:8001/mcp
    AGENT_MAX_REFLECTION_ATTEMPTS=5
    AGENT_MAX_REACT_STEPS=10
    AGENT_GUARD_MODEL=fastino/gliguard-LLMGuardrails-300M
    AGENT_GUARD_DEVICE=cpu
    AGENT_APP_HOST=0.0.0.0
    AGENT_APP_PORT=8000
    AGENT_MCP_HOST=0.0.0.0
    AGENT_MCP_PORT=8001
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
        default="http://localhost:8001/mcp",
        description="URL of the fastmcp tool server (streamable-http endpoint).",
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
    web_search_max_results: int = Field(
        default=10,
        description="Server-side cap on max_results for web_search and fact_check MCP tools.",
    )
    guard_model: str = Field(
        default="fastino/gliguard-LLMGuardrails-300M",
        description="HuggingFace model name for the GLiGuard guardrail (GLiNER2-based).",
    )
    guard_device: str = Field(
        default="cpu",
        description="Device for GLiGuard inference: 'cpu', 'cuda', or 'mps'.",
    )
    app_host: str = Field(default="0.0.0.0", description="Bind host for the FastAPI app.")
    app_port: int = Field(default=8000, description="Bind port for the FastAPI app.")
    mcp_host: str = Field(default="0.0.0.0", description="Bind host for the MCP tool server.")
    mcp_port: int = Field(default=8001, description="Bind port for the MCP tool server.")


settings = Settings()
