"""Structured JSON logging for churn-lib CLI entrypoints.

Imported inside main() only — never at module level — so library consumers
are free to configure their own logging without interference.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime

# Fields that are part of every LogRecord but should not appear as extra keys.
_STDLIB_RECORD_KEYS = frozenset(logging.LogRecord("", 0, "", 0, "", (), None).__dict__)


class JsonFormatter(logging.Formatter):
    """Emit each log record as a single-line JSON object.

    Always-present fields: timestamp (ISO-8601 UTC), level, logger, message.
    Any values passed via ``extra=`` are merged in as additional top-level keys:

        logger.info("Pipeline saved", extra={"path": str(path), "size_kb": 42})
    """

    def format(self, record: logging.LogRecord) -> str:
        doc: dict[str, object] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Merge any extra fields the caller attached
        for key, value in record.__dict__.items():
            if key not in _STDLIB_RECORD_KEYS and not key.startswith("_"):
                doc[key] = value

        if record.exc_info:
            doc["exc_info"] = self.formatException(record.exc_info)

        return json.dumps(doc, default=str)


def configure_cli_logging(level: str = "INFO") -> None:
    """Attach a JSON stderr handler to the root logger. Call once inside main()."""
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    root.handlers.clear()
    root.addHandler(handler)
