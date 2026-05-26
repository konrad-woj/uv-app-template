# agent-app

LangGraph research-assistant reference implementation using FastAPI + Postgres checkpointing.

Each node in the graph is a vehicle for exactly one LangGraph or agentic pattern — the business logic is deliberately simple so the mechanics are easy to follow.

## Patterns demonstrated

| Pattern | Where | Mechanism |
|---|---|---|
| **Subgraph** | `search_subgraph`, `reflection_subgraph` | Compiled `StateGraph` added as a single node via wrapper |
| **Fan-out / Fan-in** | Inside `search_subgraph` | `Send` API spawns parallel searchers; `operator.add` reducer collects results |
| **ReAct** | `react_researcher` | Model ↔ `ToolNode` loop; exits when model emits no `tool_calls` |
| **Human-in-the-loop** | `planner` | `interrupt()` pauses graph; resumed with `Command(resume=True/False)` |
| **Reflection** | `reflection_subgraph` | Critic → Refiner loop until quality criteria are met |
| **MCP** | `react_researcher` (consumer) + `app/mcp/server.py` (server) | `fastmcp` exposes tools; `langchain-mcp-adapters` binds them to LangGraph |
| **Guardrails** | `input_guard`, `output_guard` | LLM-based safety/relevance check at graph entry and exit |
| **Dead letter** | `dead_letter` terminal node | Any unhandled node exception writes `DeadLetterInfo` to state and routes here instead of crashing |
| **Time-travel** | `GET /v1/threads/{id}/history`, `POST /v1/threads/{id}/replay` | Postgres checkpointer stores every state snapshot; replay re-invokes from any checkpoint |
| **SSE streaming** | `POST /v1/chat/stream` | `astream_events(version="v2")` filtered to `on_chat_model_stream` |

## Reliability safeguards

All limits are configurable via `AGENT_` env vars.

| Safeguard | Env var | Default | What it prevents |
|---|---|---|---|
| Reflection ceiling | `AGENT_MAX_REFLECTION_ATTEMPTS` | `5` | Critic/refiner loop running indefinitely when quality bar is never met |
| ReAct ceiling | `AGENT_MAX_REACT_STEPS` | `10` | Model emitting tool calls forever without self-terminating |
| LLM timeout | `AGENT_LLM_TIMEOUT_SECONDS` | `60` | Single LLM call blocking the event loop indefinitely |
| Retry with backoff | `AGENT_LLM_MAX_RETRIES` | `3` | Transient rate-limit / 5xx errors surfacing as failures immediately |
| Global pipeline cap | `AGENT_MAX_PIPELINE_STEPS` | `50` | Routing bugs or unexpected loops consuming unbounded supersteps (`GraphRecursionError` on breach) |

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/)
- Python 3.13
- [uv](https://docs.astral.sh/uv/getting-started/installation/)
- LLM inference server (Phase 2+) — see [LLM setup](#2-llm-setup)

## Local setup

### 1. Postgres

```bash
docker run --name langgraph-db \
  -e POSTGRES_PASSWORD=postgres \
  -e POSTGRES_DB=langgraph \
  -e POSTGRES_USER=postgres \
  -p 5433:5432 \
  -d postgres:17
```

### 2. LLM setup

**Option A — Unsloth Studio** (recommended; GUI, easier model management):

```bash
pip install unsloth-studio
unsloth studio -H 127.0.0.1 -p 8888
# Open http://127.0.0.1:8888, download unsloth/Qwen3.6-35B-A3B-MTP-GGUF (UD-Q4_K_XL quant), click Start.
# Set AGENT_LLM_BASE_URL=http://127.0.0.1:<port>/v1 to the port shown in Studio.
```

**Option B — llama.cpp server** (headless):

```bash
hf download unsloth/Qwen3.6-35B-A3B-MTP-GGUF --include "*UD-Q4_K_XL*"
./llama.cpp/llama-server \
  --model unsloth/Qwen3.6-35B-A3B-MTP-GGUF/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf \
  --alias "unsloth/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf" \
  --ctx-size 16384 --port 8002
# Set AGENT_LLM_BASE_URL=http://127.0.0.1:8002/v1
# Port 8001 is reserved for the MCP server (AGENT_MCP_SERVER_URL default).
```

### 3. MCP server (Phase 2+, separate terminal)

```bash
cd agent-app
uv run python -m app.mcp.server   # serves on http://localhost:8001
```

### 4. App

```bash
cd agent-app
cp .env.example .env   # fill LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY for Phase 4
uv sync
uv run python -m app
# → http://localhost:8000/docs
```

### 5. Langfuse (Phase 4, optional)

```bash
git clone https://github.com/langfuse/langfuse.git
cd langfuse
docker compose up -d
# UI at http://localhost:3000 — default login: admin@langfuse.com / password
# Create a project and copy the keys to .env
```

### 6. LangGraph Studio

Download the free Mac desktop app from https://studio.langchain.com.
Open the `agent-app/` directory — Studio reads `langgraph.json` and starts a dev server automatically.

## Environment variables

All variables use the `AGENT_` prefix. Defaults work for local development.

| Variable | Default | Description |
|---|---|---|
| `AGENT_DB_URI` | `postgresql://postgres:postgres@localhost:5433/langgraph` | Postgres connection string |
| `AGENT_LLM_MODEL` | `openai/unsloth/Qwen3.6-35B-A3B-UD-MLX-4bit` | LiteLLM model identifier |
| `AGENT_LLM_BASE_URL` | `http://127.0.0.1:8888/v1` | LLM provider base URL (Unsloth Studio) |
| `AGENT_LLM_THINKING` | `false` | Enable Qwen3 chain-of-thought mode |
| `AGENT_LLM_TIMEOUT_SECONDS` | `60` | Per-call LLM timeout in seconds |
| `AGENT_LLM_MAX_RETRIES` | `3` | Retries for transient LLM errors (exponential backoff) |
| `AGENT_MCP_SERVER_URL` | `http://localhost:8001` | MCP tool server URL |
| `AGENT_MAX_REFLECTION_ATTEMPTS` | `5` | Hard ceiling on reflection critic/refiner iterations |
| `AGENT_MAX_REACT_STEPS` | `10` | Hard ceiling on ReAct tool-call iterations |
| `AGENT_MAX_PIPELINE_STEPS` | `50` | LangGraph `recursion_limit`: total supersteps across the whole pipeline per invocation |
| `AGENT_LOG_LEVEL` | `INFO` | Logging verbosity: DEBUG, INFO, WARNING, ERROR |

## API

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Liveness probe |
| `POST` | `/v1/chat` | Invoke the agent (first turn or interrupt resume) |
| `GET` | `/v1/threads/{id}/history` | Full checkpoint list for a thread (time-travel) |
| `POST` | `/v1/threads/{id}/replay` | Re-invoke from a named checkpoint |
| `POST` | `/v1/chat/stream` | SSE token stream (Phase 3) |

### Example

```bash
# Start a research session
curl -s -X POST http://localhost:8000/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"thread_id": "session-1", "message": "Research recent advances in LLM agents"}' | jq

# Resume after human-in-the-loop interrupt (approve the plan)
curl -s -X POST http://localhost:8000/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"thread_id": "session-1", "message": "", "resume": true}' | jq

# Retrieve checkpoint history
curl -s http://localhost:8000/v1/threads/session-1/history | jq

# Replay from a specific checkpoint
curl -s -X POST http://localhost:8000/v1/threads/session-1/replay \
  -H "Content-Type: application/json" \
  -d '{"checkpoint_id": "<id_from_history>"}' | jq

# Stream tokens (Phase 3)
curl -N -X POST http://localhost:8000/v1/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"thread_id": "session-2", "message": "What is RAG?"}'
```

## Tests

```bash
# Full suite (requires Postgres on localhost:5433)
uv run pytest

# Models only — no Postgres needed
uv run pytest tests/test_models.py

# Graph checkpointing and time-travel
uv run pytest tests/graph/
```

## Evals (Phase 4)

```bash
uv run task create-dataset   # upload sample.yaml to Langfuse
uv run task experiment       # run eval suite, write results to evals/.runs/
```

Results appear in the Langfuse UI at http://localhost:3000.
