"""Tests for batchpipe.structured."""

import pytest

from batchpipe.structured import (
    NoToolCallError,
    SchemaValidationError,
    extract,
    extraction_tool,
    validate_output,
)
from conftest import FakeClient, FakeMessage, FakeText, FakeToolUse

SCHEMA = {
    "type": "object",
    "properties": {
        "symbol": {"type": "string"},
        "pnl": {"type": "number"},
    },
    "required": ["symbol", "pnl"],
    "additionalProperties": False,
}


def test_validate_output_success():
    data = {"symbol": "SOFI", "pnl": -48.0}
    assert validate_output(data, SCHEMA) is data


def test_validate_output_failure_names_path_and_value():
    with pytest.raises(SchemaValidationError) as excinfo:
        validate_output({"symbol": "SOFI", "pnl": "oops"}, SCHEMA)
    msg = str(excinfo.value)
    assert "/pnl" in msg
    assert "oops" in msg


def test_validate_output_missing_required():
    with pytest.raises(SchemaValidationError) as excinfo:
        validate_output({"symbol": "SOFI"}, SCHEMA)
    assert "pnl" in str(excinfo.value)


def test_extraction_tool_shape():
    tool = extraction_tool(SCHEMA, name="extract_data")
    assert tool["name"] == "extract_data"
    assert tool["input_schema"] == SCHEMA


def test_extract_forces_tool_and_validates():
    payload = {"symbol": "AVGO", "pnl": 120.5}
    client = FakeClient(create_response=FakeMessage([FakeToolUse("extract_data", payload)]))
    out = extract(
        client,
        messages=[{"role": "user", "content": "doc"}],
        json_schema=SCHEMA,
        model="claude-sonnet-4-6",
    )
    assert out == payload
    call = client.messages.create_calls[0]
    assert call["tool_choice"] == {"type": "tool", "name": "extract_data"}
    assert call["tools"][0]["input_schema"] == SCHEMA


def test_extract_raises_when_no_tool_call():
    client = FakeClient(create_response=FakeMessage([FakeText("just text")]))
    with pytest.raises(NoToolCallError):
        extract(client, [{"role": "user", "content": "doc"}], SCHEMA)


def test_extract_raises_on_schema_violation():
    client = FakeClient(
        create_response=FakeMessage([FakeToolUse("extract_data", {"symbol": "SOFI"})])
    )
    with pytest.raises(SchemaValidationError):
        extract(client, [{"role": "user", "content": "doc"}], SCHEMA)
