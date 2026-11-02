"""Unit tests for PrivacyFilter. Needs only pytest: python -m pytest tests"""

import logging
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vllm_privacy.logging_filter import (  # noqa: E402
    REDACTED,
    REQUEST_LOGGER,
    PrivacyFilter,
)

SECRET = "my card number is 4111"


def make_record(name, level, msg, args=(), exc_info=None):
    return logging.LogRecord(name, level, __file__, 1, msg, args, exc_info)


def rendered(record):
    formatter = logging.Formatter("%(message)s")
    return formatter.format(record)


def keep(record):
    return PrivacyFilter().filter(record)


def exc_info_with(message):
    try:
        raise ValueError(message)
    except ValueError:
        return sys.exc_info()


def test_debug_records_are_dropped_for_any_logger():
    assert not keep(make_record("vllm.anything", logging.DEBUG, SECRET))
    assert not keep(make_record("root", logging.DEBUG, SECRET))


def test_debug_body_of_the_request_logger_is_dropped():
    record = make_record(
        REQUEST_LOGGER, logging.INFO, "Request %s JSON body: %s", ("id", SECRET)
    )
    assert not keep(record)


def test_received_request_keeps_only_metadata():
    params = SimpleNamespace(
        max_tokens=16, temperature=0.5, top_p=1.0, stop=[SECRET], logit_bias=SECRET
    )
    record = make_record(
        REQUEST_LOGGER,
        logging.INFO,
        "Received request %s: params: %s, lora_request: %s.",
        ("chatcmpl-abc12345", params, None),
    )
    assert keep(record)
    text = rendered(record)
    assert "chatcmpl-abc12345" in text
    assert "max_tokens=16" in text
    assert SECRET not in text


def test_generated_response_loses_its_output():
    record = make_record(
        REQUEST_LOGGER,
        logging.INFO,
        "Generated response %s%s: output: %r, finish_reason: %s",
        ("chatcmpl-abc12345", "", SECRET, "stop"),
    )
    assert keep(record)
    text = rendered(record)
    assert "finish_reason=stop" in text
    assert SECRET not in text


def test_streaming_deltas_are_dropped():
    record = make_record(
        REQUEST_LOGGER,
        logging.INFO,
        "Generated response %s%s: output: %r, finish_reason: %s",
        ("chatcmpl-abc12345", " (streaming delta)", SECRET, None),
    )
    assert not keep(record)


def test_model_name_from_the_request_is_redacted():
    # vLLM logs "Error with model %s" with the client-supplied model field.
    record = make_record(
        "vllm.entrypoints.openai.chat_completion.serving",
        logging.ERROR,
        "Error with model %s",
        (SECRET,),
    )
    assert keep(record)
    assert SECRET not in rendered(record)


def test_exception_argument_in_a_content_logger_is_reduced_to_its_type():
    record = make_record(
        "vllm.v1.engine.async_llm",
        logging.INFO,
        "Request %s failed (bad request): %s.",
        ("chatcmpl-abc12345", ValueError(SECRET)),
    )
    assert keep(record)
    text = rendered(record)
    assert "chatcmpl-abc12345" in text
    assert "<ValueError>" in text
    assert SECRET not in text


def test_string_argument_in_a_content_logger_is_redacted():
    record = make_record(
        "vllm.renderers.hf", logging.WARNING, "Template rejected: %s", (SECRET,)
    )
    assert keep(record)
    assert rendered(record) == f"Template rejected: {REDACTED}"


def test_numbers_survive_in_a_content_logger():
    record = make_record(
        "vllm.entrypoints.openai.x", logging.WARNING, "n=%d ratio=%.1f", (3, 0.5)
    )
    assert keep(record)
    assert rendered(record) == "n=3 ratio=0.5"


def test_mapping_arguments_are_redacted():
    record = make_record("vllm.renderers.hf", logging.WARNING, "x=%(x)s", ())
    record.args = {"x": SECRET}
    assert keep(record)
    assert SECRET not in rendered(record)


def test_exception_text_is_removed_from_tracebacks():
    record = make_record(
        "uvicorn.error",
        logging.ERROR,
        "Exception in ASGI application",
        exc_info=exc_info_with(SECRET),
    )
    assert keep(record)
    text = rendered(record)
    assert "ValueError" in text
    assert "exc_info_with" in text
    assert SECRET not in text


def test_exception_interpolated_by_an_f_string_is_removed():
    record = make_record(
        "vllm.tool_parsers.xlam_tool_parser",
        logging.ERROR,
        f"Error in streaming tool calls: {SECRET}",
        exc_info=exc_info_with(SECRET),
    )
    assert keep(record)
    assert SECRET not in rendered(record)


def test_scheduler_dump_is_dropped():
    record = make_record(
        "vllm.logging_utils.dump_input",
        logging.ERROR,
        "Dumping scheduler output for model execution: %s",
        (SECRET,),
    )
    assert not keep(record)


def test_uvicorn_access_log_is_dropped():
    record = make_record(
        "uvicorn.access",
        logging.INFO,
        '%s - "%s %s HTTP/%s" %d',
        ("10.0.0.1:1", "GET", f"/v1/models?q={SECRET}", "1.1", 200),
    )
    assert not keep(record)


def test_startup_logs_stay_readable():
    record = make_record(
        "vllm.entrypoints.launchers.api_server.entry",
        logging.INFO,
        "Starting server on %s",
        ("http://0.0.0.0:8000",),
    )
    assert keep(record)
    assert "http://0.0.0.0:8000" in rendered(record)


def test_engine_startup_logs_stay_readable():
    record = make_record(
        "vllm.v1.engine.core", logging.INFO, "Loaded model %s", ("/models/qwen",)
    )
    assert keep(record)
    assert "/models/qwen" in rendered(record)
