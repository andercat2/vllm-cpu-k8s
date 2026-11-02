"""Logging filter that keeps user content out of stdout/stderr.

It is attached to the handlers in ``docker/vllm/logging.json``. vLLM loads that
file through ``VLLM_LOGGING_CONFIG_PATH`` in every process it spawns, so the
filter also covers the API server and engine processes. Nothing in site-packages
has to be patched.

Keep this module free of vLLM imports: it is imported while ``vllm.logger`` is
still initialising.
"""

import logging
import re
import traceback
from typing import Any

REDACTED = "<redacted>"

REQUEST_LOGGER = "vllm.entrypoints.serve.utils.request_logger"

# Modules that see prompts, chat messages or generated text. String arguments of
# their records are redacted, because an arbitrary string or exception message
# here can quote the user's input.
CONTENT_LOGGERS = (
    "vllm.entrypoints.",
    "vllm.renderers.",
    "vllm.parser.",
    "vllm.tool_parsers.",
    "vllm.reasoning.",
    "vllm.v1.engine.async_llm",
    "vllm.v1.engine.input_processor",
    "vllm.v1.engine.output_processor",
    "vllm.v1.engine.detokenizer",
    "vllm.v1.structured_output.",
)

# Startup code inside vllm.entrypoints. It never sees request content, and its
# routes, addresses and versions are worth keeping.
STARTUP_LOGGERS = (
    "vllm.entrypoints.launchers.",
    "vllm.entrypoints.cli.",
)

# Exception tracebacks from these loggers lose their message text.
EXCEPTION_LOGGERS = CONTENT_LOGGERS + ("uvicorn.error", "fastapi", "starlette")

# Fired on engine failures, with the token ids of every scheduled request.
DROPPED_MESSAGE_PREFIXES = ("Dumping scheduler output",)

# chatcmpl-1f3a..., cmpl-1f3a...-0-a1b2c3 and so on.
REQUEST_ID = re.compile(r"^(chatcmpl|cmpl|resp|embd|pool)-[0-9a-f]{8,}(-[0-9a-f]+)*$")


class PrivacyFilter(logging.Filter):
    """Drops or rewrites any record that could carry user content."""

    def filter(self, record: logging.LogRecord) -> bool:
        # DEBUG is where vLLM prints full prompts and request bodies, and the
        # uvicorn access line contains the query string. AuditLogMiddleware
        # replaces the access log.
        if record.levelno < logging.INFO or record.name == "uvicorn.access":
            return False

        message = str(record.msg)
        if message.startswith(DROPPED_MESSAGE_PREFIXES):
            return False

        if record.name == REQUEST_LOGGER:
            return _rewrite_request_log(record, message)

        if _matches(record.name, CONTENT_LOGGERS) and not _matches(
            record.name, STARTUP_LOGGERS
        ):
            _redact_arguments(record)

        if record.exc_info and _matches(record.name, EXCEPTION_LOGGERS):
            _redact_exception(record)

        return True


def _matches(name: str, prefixes: tuple[str, ...]) -> bool:
    return any(name == p.rstrip(".") or name.startswith(p) for p in prefixes)


def _rewrite_request_log(record: logging.LogRecord, message: str) -> bool:
    """Reduces RequestLogger records to metadata, dropping anything unknown."""
    args = record.args
    if not isinstance(args, tuple):
        return False

    if message.startswith("Received request"):
        request_id, params, _lora_request = args
        record.msg = (
            "Received request %s: max_tokens=%s temperature=%s top_p=%s "
            "(content redacted)"
        )
        record.args = (
            request_id,
            getattr(params, "max_tokens", None),
            getattr(params, "temperature", None),
            getattr(params, "top_p", None),
        )
        return True

    if message.startswith("Generated response"):
        request_id, stream_info, _output, finish_reason = args
        # One record per streamed token, with nothing left to say once redacted.
        if "delta" in stream_info:
            return False
        record.msg = "Generated response %s%s: finish_reason=%s (content redacted)"
        record.args = (request_id, stream_info, finish_reason)
        return True

    return False


def _safe_argument(value: Any) -> Any:
    if value is None or isinstance(value, bool | int | float):
        return value
    if isinstance(value, str) and REQUEST_ID.match(value):
        return value
    if isinstance(value, BaseException):
        return f"<{type(value).__name__}>"
    return REDACTED


def _redact_arguments(record: logging.LogRecord) -> None:
    if isinstance(record.args, dict):
        record.args = {k: _safe_argument(v) for k, v in record.args.items()}
    elif record.args:
        record.args = tuple(_safe_argument(v) for v in record.args)


def _redact_exception(record: logging.LogRecord) -> None:
    """Keeps exception types and stack frames, removes exception messages."""
    exc_type, exc, tb = record.exc_info
    lines = ["Traceback (most recent call last):\n"]
    lines += traceback.format_tb(tb)
    lines.append(f"{exc_type.__name__ if exc_type else 'Exception'}: {REDACTED}")

    # An f-string log call has already interpolated the exception message.
    if exc is not None and str(exc):
        record.msg = str(record.msg).replace(str(exc), REDACTED)

    record.exc_info = None
    record.exc_text = "".join(lines)
