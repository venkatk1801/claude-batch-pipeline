"""End-to-end pipeline: folder of documents -> Batch API -> validated results.

``run_folder`` reads .txt/.md files, builds one batch request per document
with forced structured tool use, submits the batch, waits, parses + validates
each result, and writes ``results.json`` and ``summary.csv``. It also prints
(and returns) a cost comparison of batch vs realtime pricing.
"""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from batchpipe.batches import (
    BatchResults,
    build_request,
    run_batch,
)
from batchpipe.pricing import cost_breakdown
from batchpipe.structured import DEFAULT_TOOL_NAME, SchemaValidationError, validate_output

log = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-sonnet-4-6"
SUPPORTED_SUFFIXES = {".txt", ".md"}

DEFAULT_PROMPT_TEMPLATE = (
    "Extract the fields defined by the extraction schema from the document below. "
    "Call the extract_data tool with your answer.\n\n"
    "Document:\n{document}"
)


@dataclass
class Document:
    path: Path
    text: str


def read_documents(input_dir: str | Path) -> list[Document]:
    """Read all .txt/.md files in a directory, sorted by filename."""
    input_dir = Path(input_dir)
    if not input_dir.is_dir():
        raise ValueError(f"Not a directory: {input_dir}")
    docs: list[Document] = []
    for path in sorted(input_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES:
            text = path.read_text(encoding="utf-8")
            if not text.strip():
                log.warning("Skipping empty file: %s", path.name)
                continue
            docs.append(Document(path=path, text=text))
    if not docs:
        raise ValueError(f"No .txt/.md files found in {input_dir}")
    return docs


def estimate_tokens(text: str) -> int:
    """Rough token estimate (~4 chars/token). For cost planning only, not billing."""
    return max(1, len(text) // 4)


def estimate_folder_cost(
    input_dir: str | Path,
    model: str = DEFAULT_MODEL,
    output_tokens_per_doc: int = 500,
) -> dict:
    """Pre-flight cost estimate for a folder, batch vs realtime."""
    docs = read_documents(input_dir)
    input_tokens = sum(estimate_tokens(d.text) for d in docs)
    output_tokens = output_tokens_per_doc * len(docs)
    breakdown = cost_breakdown(input_tokens, output_tokens, model)
    return {
        "documents": len(docs),
        "model": model,
        "estimated_input_tokens": input_tokens,
        "estimated_output_tokens": output_tokens,
        **breakdown,
    }


def build_pipeline_requests(
    docs: list[Document],
    json_schema: dict,
    prompt_template: str,
    model: str = DEFAULT_MODEL,
    max_tokens: int = 4096,
    tool_name: str = DEFAULT_TOOL_NAME,
) -> list[dict]:
    """Build one batch request per document with forced structured tool use."""
    if "{document}" not in prompt_template:
        prompt_template = prompt_template.rstrip() + "\n\n{document}"
    requests = []
    for doc in docs:
        prompt = prompt_template.replace("{document}", doc.text)
        requests.append(
            build_request(
                custom_id=doc.path.name,
                messages=[{"role": "user", "content": prompt}],
                model=model,
                max_tokens=max_tokens,
                tools=[
                    {
                        "name": tool_name,
                        "description": "Extract structured data matching the provided schema.",
                        "input_schema": json_schema,
                    }
                ],
                tool_choice={"type": "tool", "name": tool_name},
            )
        )
    return requests


def run_folder(
    input_dir: str | Path,
    schema: dict,
    prompt_template: str | None = None,
    model: str = DEFAULT_MODEL,
    output_dir: str | Path | None = None,
    client=None,
    max_tokens: int = 4096,
    poll_interval: float = 30.0,
    timeout: float = 24 * 3600.0,
) -> dict:
    """Run the full pipeline over a folder of documents.

    Returns a summary dict with counts, cost comparison, and output paths.
    """
    if client is None:
        raise ValueError("A configured client is required (pass anthropic.Anthropic())")
    template = prompt_template or DEFAULT_PROMPT_TEMPLATE
    out_dir = Path(output_dir) if output_dir else Path.cwd() / "batchpipe-output"
    out_dir.mkdir(parents=True, exist_ok=True)

    docs = read_documents(input_dir)
    requests = build_pipeline_requests(docs, schema, template, model, max_tokens)
    log.info("Submitting %d documents as one batch (%s)", len(docs), model)

    batch_id, results = run_batch(
        client,
        requests,
        tool_name=DEFAULT_TOOL_NAME,
        poll_interval=poll_interval,
        timeout=timeout,
    )

    records = _validate_results(results, schema)
    cost = _total_cost(records, model)

    results_path = out_dir / "results.json"
    csv_path = out_dir / "summary.csv"
    results_path.write_text(
        json.dumps(
            {
                "batch_id": batch_id,
                "model": model,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "cost": cost,
                "results": records,
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    _write_csv(csv_path, records)

    summary = {
        "batch_id": batch_id,
        "documents": len(docs),
        "succeeded": sum(1 for r in records if r["status"] == "succeeded"),
        "failed": sum(1 for r in records if r["status"] != "succeeded"),
        "cost": cost,
        "results_json": str(results_path),
        "summary_csv": str(csv_path),
    }
    _print_summary(summary, records)
    return summary


def _validate_results(results: BatchResults, schema: dict) -> list[dict]:
    """Validate each succeeded result against the schema; collect per-file records."""
    records: list[dict] = []
    for s in results.succeeded:
        record = {
            "file": s.custom_id,
            "custom_id": s.custom_id,
            "status": "succeeded",
            "output": s.output,
            "error": None,
            "usage": s.usage,
        }
        try:
            validate_output(s.output, schema)
        except SchemaValidationError as exc:
            record["status"] = "validation_failed"
            record["error"] = str(exc)
        records.append(record)
    for f in results.failed:
        records.append(
            {
                "file": f.custom_id,
                "custom_id": f.custom_id,
                "status": f.result_type,  # errored | canceled | expired | no_tool_call
                "output": None,
                "error": f.error,
                "usage": {"input_tokens": 0, "output_tokens": 0},
            }
        )
    return records


def _total_cost(records: list[dict], model: str) -> dict:
    input_tokens = sum(r["usage"]["input_tokens"] for r in records)
    output_tokens = sum(r["usage"]["output_tokens"] for r in records)
    return {"model": model, **cost_breakdown(input_tokens, output_tokens, model)}


def _write_csv(csv_path: Path, records: list[dict]) -> None:
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "file",
                "custom_id",
                "status",
                "input_tokens",
                "output_tokens",
                "error",
            ],
        )
        writer.writeheader()
        for r in records:
            writer.writerow(
                {
                    "file": r["file"],
                    "custom_id": r["custom_id"],
                    "status": r["status"],
                    "input_tokens": r["usage"]["input_tokens"],
                    "output_tokens": r["usage"]["output_tokens"],
                    "error": r["error"] or "",
                }
            )


def _print_summary(summary: dict, records: list[dict]) -> None:
    cost = summary["cost"]
    print(f"\nBatch {summary['batch_id']} complete: "
          f"{summary['succeeded']} succeeded, {summary['failed']} failed "
          f"of {summary['documents']} documents")
    print(f"Cost: ${cost['batch_usd']:.4f} (batch) vs ${cost['realtime_usd']:.4f} "
          f"(realtime) — saved ${cost['savings_usd']:.4f}")
    print(f"Results: {summary['results_json']}")
    print(f"Summary: {summary['summary_csv']}")
    for r in records:
        if r["status"] != "succeeded":
            print(f"  FAILED {r['file']}: [{r['status']}] {r['error']}")
