# uv-app-template

A collection of best practices and patterns for Python AI/ML projects — designed to learn from or use as a starting point.

The repo is a monorepo with four packages:

| Package | Description |
|---------|-------------|
| [`churn-lib/`](churn-lib/README.md) | ML library — training, inference, validation, and drift detection for churn prediction |
| [`ml-app/`](ml-app/README.md) | FastAPI microservice — production-ready HTTP API built on top of churn-lib |
| [`agent-lib/`](agent-lib/README.md) | LangGraph agent library — reusable graph primitives and Postgres checkpointing |
| [`agent-app/`](agent-app/README.md) | LangGraph research-assistant reference implementation — FastAPI + Postgres checkpointing |

## Highlights

### ML microservice (`churn-lib` / `ml-app`)

- **Layered architecture** — endpoints → services → library; no HTTP leaking into business logic
- **Optional extras** — inference stays lightweight; heavy training deps (`mlflow`, `matplotlib`) are opt-in
- **Dependency injection** — `get_pipeline()` FastAPI dependency; overridable in tests
- **Structured JSON logging** with request ID correlation
- **Async + sync training** endpoints with in-memory job tracking
- **MLflow experiment tracking** — every training run logged with params, metrics, and artifacts; local file store or Docker-based server
- **Drift detection** via Population Stability Index (PSI)
- **Threshold calibration** — find the F1/recall/precision operating point that fits your business constraint
- **Docker** multi-stage build from repo root; non-root user

### LangGraph agent (`agent-lib` / `agent-app`)

- **LangGraph patterns** — subgraph, fan-out/fan-in (`Send` API), ReAct, human-in-the-loop (`interrupt`), reflection, MCP, guardrails, dead letter, time-travel, token streaming
- **Postgres checkpointing** — every state snapshot persisted; full time-travel and replay via API
- **Three-layer guardrails** — regex → GLiGuard → LLM topic check on input; GLiGuard PII redaction → LLM grounding check on output
- **MCP integration** — `fastmcp` server exposes tools; `langchain-mcp-adapters` binds them to the ReAct researcher node
- **Reliability safeguards** — configurable ceilings for reflection loops, ReAct steps, LLM timeouts, retries, and total pipeline supersteps
- **SSE streaming** — writer-node tokens streamed as Server-Sent Events; keepalive ping prevents proxy idle-timeout
- **Langfuse observability** — optional tracing; configure via `LANGFUSE_*` env vars
