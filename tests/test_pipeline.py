"""Tests for pricing math and the end-to-end pipeline (mocked client)."""

import csv
import json

import pytest

from batchpipe.pipeline import (
    build_pipeline_requests,
    estimate_folder_cost,
    read_documents,
    run_folder,
)
from batchpipe.pricing import cost_breakdown, estimate_cost, known_models
from conftest import FakeClient, errored_response, succeeded_response

SCHEMA = {
    "type": "object",
    "properties": {
        "symbol": {"type": "string"},
        "sentiment": {"type": "string", "enum": ["bullish", "bearish", "neutral"]},
    },
    "required": ["symbol", "sentiment"],
    "additionalProperties": False,
}


def write_docs(target_dir, files):
    target_dir.mkdir(parents=True, exist_ok=True)
    for name, text in files.items():
        (target_dir / name).write_text(text, encoding="utf-8")
    return target_dir


# ---- pricing ----

def test_estimate_cost_batch_is_half_realtime():
    realtime = estimate_cost(1_000_000, 1_000_000, "claude-sonnet-4-6", batch=False)
    batch = estimate_cost(1_000_000, 1_000_000, "claude-sonnet-4-6", batch=True)
    assert realtime == pytest.approx(3.0 + 15.0)
    assert batch == pytest.approx(1.5 + 7.5)
    assert batch == pytest.approx(realtime / 2)


def test_cost_breakdown_fields():
    bd = cost_breakdown(500_000, 100_000, "claude-haiku-4-5")
    assert bd["realtime_usd"] == pytest.approx(0.5 * 1.0 + 0.1 * 5.0)
    assert bd["batch_usd"] == pytest.approx(bd["realtime_usd"] / 2)
    assert bd["savings_usd"] == pytest.approx(bd["realtime_usd"] - bd["batch_usd"])


def test_unknown_model_raises():
    with pytest.raises(ValueError, match="Unknown model"):
        estimate_cost(100, 100, "claude-fictional-9")


def test_known_models_lists_current_lineup():
    assert "claude-sonnet-4-6" in known_models()


# ---- pipeline assembly ----

def test_build_pipeline_requests_jsonl(tmp_path):
    docs = read_documents(write_docs(tmp_path, {"a.md": "x" * 100, "b.txt": "y" * 200}))
    requests = build_pipeline_requests(docs, SCHEMA, "Read:\n{document}", "claude-sonnet-4-6")
    assert [r["custom_id"] for r in requests] == ["a.md", "b.txt"]
    params = requests[0]["params"]
    assert "x" * 100 in params["messages"][0]["content"]
    assert params["tool_choice"] == {"type": "tool", "name": "extract_data"}
    assert params["tools"][0]["input_schema"] == SCHEMA


def test_build_pipeline_requests_appends_document_when_no_placeholder(tmp_path):
    docs = read_documents(write_docs(tmp_path, {"a.md": "hello doc"}))
    requests = build_pipeline_requests(docs, SCHEMA, "Extract fields.", "claude-sonnet-4-6")
    assert requests[0]["params"]["messages"][0]["content"].endswith("hello doc")


def test_read_documents_rejects_non_directory(tmp_path):
    with pytest.raises(ValueError, match="Not a directory"):
        read_documents(tmp_path / "missing")


# ---- end to end ----

def test_run_folder_end_to_end(tmp_path, capsys):
    input_dir = write_docs(
        tmp_path / "in",
        {"trade1.md": "Bought SOFI, feeling bullish.", "trade2.md": "Shorted INTC, bearish."},
    )
    out_dir = tmp_path / "out"

    client = FakeClient()
    client.messages.batches.retrieve_statuses = ["ended"]
    client.messages.batches.individual_responses = [
        succeeded_response("trade1.md", {"symbol": "SOFI", "sentiment": "bullish"},
                           input_tokens=100, output_tokens=25),
        succeeded_response("trade2.md", {"symbol": "INTC", "sentiment": "unknown"},  # invalid enum
                           input_tokens=100, output_tokens=25),
    ]

    summary = run_folder(
        input_dir,
        SCHEMA,
        prompt_template="Extract:\n{document}",
        model="claude-sonnet-4-6",
        output_dir=out_dir,
        client=client,
        poll_interval=0.001,
    )

    assert summary["documents"] == 2
    assert summary["succeeded"] == 1
    assert summary["failed"] == 1  # validation_failed
    assert summary["batch_id"].startswith("msgbatch_test_")

    results = json.loads((out_dir / "results.json").read_text(encoding="utf-8"))
    assert len(results["results"]) == 2
    by_file = {r["file"]: r for r in results["results"]}
    assert by_file["trade1.md"]["status"] == "succeeded"
    assert by_file["trade1.md"]["output"] == {"symbol": "SOFI", "sentiment": "bullish"}
    assert by_file["trade2.md"]["status"] == "validation_failed"
    assert "sentiment" in by_file["trade2.md"]["error"]

    cost = results["cost"]
    assert cost["batch_usd"] == pytest.approx(cost["realtime_usd"] / 2)

    with (out_dir / "summary.csv").open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 2
    assert rows[0]["file"] == "trade1.md"
    assert rows[0]["status"] == "succeeded"
    assert rows[1]["status"] == "validation_failed"


def test_run_folder_surfaces_request_errors(tmp_path, capsys):
    input_dir = write_docs(tmp_path / "in", {"bad.md": "unprocessable"})
    out_dir = tmp_path / "out"

    client = FakeClient()
    client.messages.batches.retrieve_statuses = ["ended"]
    client.messages.batches.individual_responses = [errored_response("bad.md")]

    summary = run_folder(input_dir, SCHEMA, model="claude-sonnet-4-6",
                         output_dir=out_dir, client=client, poll_interval=0.001)
    assert summary["succeeded"] == 0
    assert summary["failed"] == 1
    results = json.loads((out_dir / "results.json").read_text(encoding="utf-8"))
    assert results["results"][0]["status"] == "errored"


def test_estimate_folder_cost(tmp_path):
    input_dir = write_docs(tmp_path, {"a.md": "x" * 4000, "b.md": "y" * 4000})
    est = estimate_folder_cost(input_dir, model="claude-sonnet-4-6", output_tokens_per_doc=500)
    assert est["documents"] == 2
    assert est["estimated_input_tokens"] == 2000
    assert est["estimated_output_tokens"] == 1000
    assert est["batch_usd"] == pytest.approx(est["realtime_usd"] / 2)
