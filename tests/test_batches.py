"""Tests for batchpipe.batches — full batch lifecycle with a mocked client."""

import json

import pytest

from batchpipe import batches
from batchpipe.batches import (
    BatchSubmitError,
    BatchTimeoutError,
    MissingToolCallError,
    build_jsonl,
    build_request,
    parse_batch_results,
    poll_batch,
    run_batch,
    submit_batch,
)
from conftest import (
    FakeBatch,
    FakeClient,
    FakeIndividualResponse,
    FakeResult,
    FakeText,
    FakeMessage,
    errored_response,
    succeeded_response,
)

SCHEMA = {"type": "object", "properties": {"symbol": {"type": "string"}}, "required": ["symbol"]}


def make_requests():
    return [
        build_request(
            "doc-a",
            [{"role": "user", "content": "hello"}],
            model="claude-sonnet-4-6",
            tools=[{"name": "extract_data", "description": "x", "input_schema": SCHEMA}],
            tool_choice={"type": "tool", "name": "extract_data"},
        ),
        build_request("doc-b", [{"role": "user", "content": "world"}], model="claude-sonnet-4-6"),
    ]


def test_build_jsonl_assembles_valid_lines():
    requests = make_requests()
    body = build_jsonl(requests)
    lines = body.strip().split("\n")
    assert len(lines) == 2
    first = json.loads(lines[0])
    assert first["custom_id"] == "doc-a"
    assert first["params"]["model"] == "claude-sonnet-4-6"
    assert first["params"]["tool_choice"] == {"type": "tool", "name": "extract_data"}
    assert first["params"]["tools"][0]["input_schema"] == SCHEMA


def test_build_jsonl_rejects_empty():
    with pytest.raises(ValueError):
        build_jsonl([])


def test_submit_batch_returns_id_and_passes_requests(client):
    requests = make_requests()
    batch_id = submit_batch(client, requests)
    assert batch_id.startswith("msgbatch_test_")
    assert client.messages.batches.created_requests == requests


def test_submit_batch_wraps_errors():
    class Boom:
        @property
        def messages(self):
            raise RuntimeError("nope")

    with pytest.raises(BatchSubmitError):
        submit_batch(Boom(), make_requests())


def test_poll_batch_backs_off_until_ended(client):
    client.messages.batches.retrieve_statuses = ["in_progress", "in_progress", "ended"]
    batch = poll_batch(client, "msgbatch_test_x", poll_interval=0.001, backoff=2.0)
    assert batch.processing_status == "ended"
    assert client.messages.batches.retrieve_calls == 3


def test_poll_batch_timeout(client):
    client.messages.batches.retrieve_statuses = ["in_progress"]
    with pytest.raises(BatchTimeoutError):
        poll_batch(client, "msgbatch_test_x", poll_interval=0.001, timeout=0.01)


def test_run_batch_lifecycle_mixed_results(client):
    api = client.messages.batches
    api.retrieve_statuses = ["in_progress", "ended"]
    api.individual_responses = [
        succeeded_response("doc-a", {"symbol": "SOFI"}, input_tokens=120, output_tokens=40),
        errored_response("doc-b"),
        FakeIndividualResponse("doc-c", FakeResult("canceled")),
        FakeIndividualResponse("doc-d", FakeResult("expired")),
    ]

    batch_id, results = run_batch(client, make_requests(), tool_name="extract_data",
                                  poll_interval=0.001)

    assert batch_id.startswith("msgbatch_test_")
    assert results.total == 4

    ok = results.succeeded[0]
    assert ok.custom_id == "doc-a"
    assert ok.output == {"symbol": "SOFI"}
    assert ok.usage == {"input_tokens": 120, "output_tokens": 40}

    failed = {f.custom_id: f for f in results.failed}
    assert set(failed) == {"doc-b", "doc-c", "doc-d"}
    assert failed["doc-b"].result_type == "errored"
    assert "invalid_request_error" in failed["doc-b"].error
    assert failed["doc-c"].result_type == "canceled"
    assert failed["doc-d"].result_type == "expired"


def test_parse_results_missing_tool_call_becomes_failure(client):
    api = client.messages.batches
    api.individual_responses = [
        FakeIndividualResponse(
            "doc-a",
            FakeResult("succeeded", message=FakeMessage([FakeText("no tool here")])),
        ),
    ]
    results = parse_batch_results(client, "whatever", tool_name="extract_data")
    assert not results.succeeded
    assert len(results.failed) == 1
    assert results.failed[0].result_type == "no_tool_call"
    assert "extract_data" in results.failed[0].error


def test_parse_results_raw_text_without_tool(client):
    api = client.messages.batches
    api.individual_responses = [
        FakeIndividualResponse(
            "doc-a",
            FakeResult("succeeded", message=FakeMessage([FakeText("hello"), FakeText("world")])),
        ),
    ]
    results = parse_batch_results(client, "whatever")  # no tool_name
    assert results.succeeded[0].output == "hello\nworld"


def test_extract_tool_input_missing_raises():
    from batchpipe.structured import tool_input_from_message
    msg = FakeMessage([FakeText("nope")])
    with pytest.raises(MissingToolCallError):
        tool_input_from_message(msg, "extract_data")
