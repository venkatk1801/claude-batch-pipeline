"""Thin wrapper around Anthropic's Messages Batches API.

Responsibilities:
- build batch request payloads (JSONL-ready) from (custom_id, messages) pairs,
- submit a batch and poll with exponential backoff until it ends,
- download results and parse them into (custom_id, parsed_output, usage) triples,
  surfacing per-request failures (errored / canceled / expired) individually so
  one bad request never sinks the whole batch.

A ``client`` is passed into every public function (instead of being created
here) so callers can inject a configured ``anthropic.Anthropic`` instance —
or a fake in tests.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field

from batchpipe.structured import MissingToolCallError, tool_input_from_message

log = logging.getLogger(__name__)


class BatchError(Exception):
    """Base class for batch pipeline errors."""


class BatchSubmitError(BatchError):
    """Raised when the batch request cannot be submitted."""


class BatchTimeoutError(BatchError):
    """Raised when the batch does not finish within the allotted timeout."""


@dataclass
class RequestResult:
    """Internal parse of one MessageBatchIndividualResponse."""

    custom_id: str
    succeeded: bool
    output: object | None
    usage: dict
    result_type: str = "succeeded"
    error: str | None = None


@dataclass
class SucceededResult:
    """One successful request: (custom_id, parsed_output, usage)."""

    custom_id: str
    output: object
    usage: dict


@dataclass
class FailedResult:
    """One failed request — partial failures never raise on their own."""

    custom_id: str
    result_type: str  # "errored" | "canceled" | "expired"
    error: str


@dataclass
class BatchResults:
    succeeded: list[SucceededResult] = field(default_factory=list)
    failed: list[FailedResult] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.succeeded) + len(self.failed)


def build_request(
    custom_id: str,
    messages: list[dict],
    model: str,
    max_tokens: int = 4096,
    tools: list[dict] | None = None,
    tool_choice: dict | None = None,
) -> dict:
    """Build one batch request entry (the dict serialized as one JSONL line).

    ``messages`` are standard Messages API message dicts. Pass ``tools`` +
    ``tool_choice`` to force structured output inside the batch — the Batch
    API supports the full Messages API surface including tools.
    """
    params: dict = {"model": model, "max_tokens": max_tokens, "messages": messages}
    if tools:
        params["tools"] = tools
    if tool_choice:
        params["tool_choice"] = tool_choice
    return {"custom_id": custom_id, "params": params}


def build_jsonl(requests: list[dict]) -> str:
    """Serialize batch requests to the JSONL body used by the Batch API."""
    if not requests:
        raise ValueError("Cannot build a JSONL body from an empty request list")
    return "\n".join(json.dumps(r) for r in requests)


def submit_batch(client, requests: list[dict]) -> str:
    """Submit a batch; return the batch id. Raises BatchSubmitError on failure."""
    if not requests:
        raise ValueError("Cannot submit an empty batch")
    try:
        batch = client.messages.batches.create(requests=requests)
    except Exception as exc:  # e.g. anthropic.APIError — keep it import-light
        raise BatchSubmitError(f"Failed to submit batch: {exc}") from exc
    log.info(
        "Submitted batch %s (%d requests, status=%s)",
        batch.id,
        len(requests),
        batch.processing_status,
    )
    return batch.id


def poll_batch(
    client,
    batch_id: str,
    poll_interval: float = 30.0,
    backoff: float = 1.5,
    max_interval: float = 600.0,
    timeout: float = 24 * 3600.0,
):
    """Poll until the batch reaches ``processing_status == "ended"``.

    Uses exponential backoff between polls (capped at ``max_interval``).
    Raises BatchTimeoutError if the batch hasn't ended within ``timeout``.
    """
    start = time.monotonic()
    wait = poll_interval
    last_status: str | None = None
    while True:
        batch = client.messages.batches.retrieve(batch_id)
        status = batch.processing_status
        if status != last_status:
            counts = getattr(batch, "request_counts", None)
            log.info("Batch %s status=%s counts=%s", batch_id, status, counts)
            last_status = status
        if status == "ended":
            return batch
        elapsed = time.monotonic() - start
        if elapsed >= timeout:
            raise BatchTimeoutError(
                f"Batch {batch_id} did not finish within {timeout:.0f}s "
                f"(last status: {status})"
            )
        time.sleep(wait)
        wait = min(wait * backoff, max_interval)


def _usage_dict(usage) -> dict:
    """Normalize an SDK Usage object (or dict/None) to a plain dict."""
    if usage is None:
        return {"input_tokens": 0, "output_tokens": 0}
    if isinstance(usage, dict):
        return {
            "input_tokens": int(usage.get("input_tokens", 0)),
            "output_tokens": int(usage.get("output_tokens", 0)),
        }
    return {
        "input_tokens": int(getattr(usage, "input_tokens", 0) or 0),
        "output_tokens": int(getattr(usage, "output_tokens", 0) or 0),
    }


def parse_batch_results(client, batch_id: str, tool_name: str | None = None) -> BatchResults:
    """Download results and split into succeeded triples + failed records.

    When ``tool_name`` is given, each successful message is parsed as a forced
    tool call and ``output`` is the tool's input payload. Requests whose tool
    call is missing are treated as failures, not successes.
    """
    results = BatchResults()
    for response in client.messages.batches.results(batch_id):
        custom_id = response.custom_id
        result = response.result
        result_type = getattr(result, "type", "unknown")

        if result_type != "succeeded":
            results.failed.append(
                FailedResult(
                    custom_id=custom_id,
                    result_type=result_type,
                    error=_format_failure(result, result_type),
                )
            )
            continue

        message = result.message
        try:
            if tool_name:
                output = tool_input_from_message(message, tool_name)
            else:
                output = _text_content(message)
        except MissingToolCallError as exc:
            results.failed.append(
                FailedResult(custom_id=custom_id, result_type="no_tool_call", error=str(exc))
            )
            continue
        results.succeeded.append(
            SucceededResult(
                custom_id=custom_id,
                output=output,
                usage=_usage_dict(getattr(message, "usage", None)),
            )
        )
    return results


def _text_content(message) -> str:
    parts = []
    for block in getattr(message, "content", []) or []:
        if getattr(block, "type", "") == "text":
            parts.append(getattr(block, "text", ""))
    return "\n".join(parts)


def _format_failure(result, result_type: str) -> str:
    if result_type == "errored":
        error = getattr(result, "error", None)
        if error is not None:
            etype = getattr(error, "type", "error")
            emsg = getattr(error, "message", str(error))
            return f"{etype}: {emsg}"
        return "request errored (no error detail)"
    if result_type == "canceled":
        return "request canceled"
    if result_type == "expired":
        return "request expired before processing"
    return f"request finished with unrecognized result type: {result_type}"


def run_batch(
    client,
    requests: list[dict],
    tool_name: str | None = None,
    poll_interval: float = 30.0,
    timeout: float = 24 * 3600.0,
) -> tuple[str, BatchResults]:
    """Submit, wait for, and parse a batch. Returns (batch_id, BatchResults)."""
    batch_id = submit_batch(client, requests)
    poll_batch(client, batch_id, poll_interval=poll_interval, timeout=timeout)
    return batch_id, parse_batch_results(client, batch_id, tool_name=tool_name)
