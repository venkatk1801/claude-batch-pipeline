"""Command-line interface for batchpipe."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import typer
from dotenv import load_dotenv

from batchpipe.pipeline import (
    DEFAULT_MODEL,
    estimate_folder_cost,
    read_documents,
    run_folder,
)
from batchpipe.pricing import PRICING_VERIFIED, known_models
from batchpipe.structured import extract

app = typer.Typer(help="Bulk document processing with Anthropic's Batch API.", no_args_is_help=True)


def _get_client():
    load_dotenv()
    try:
        from anthropic import Anthropic
    except ImportError:
        typer.echo("The 'anthropic' package is required. Install with: pip install anthropic", err=True)
        raise typer.Exit(1)
    return Anthropic()  # reads ANTHROPIC_API_KEY from env


def _load_schema(schema_path: str) -> dict:
    path = Path(schema_path)
    if not path.is_file():
        typer.echo(f"Schema file not found: {schema_path}", err=True)
        raise typer.Exit(1)
    try:
        schema = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        typer.echo(f"Invalid JSON in schema file: {exc}", err=True)
        raise typer.Exit(1)
    if not isinstance(schema, dict):
        typer.echo("Schema must be a JSON object.", err=True)
        raise typer.Exit(1)
    return schema


@app.command()
def estimate(
    input_dir: str = typer.Argument(..., help="Folder of .txt/.md documents."),
    model: str = typer.Option(DEFAULT_MODEL, help=f"Model alias. Known: {', '.join(known_models())}"),
    output_tokens: int = typer.Option(500, help="Assumed output tokens per document."),
):
    """Estimate batch vs realtime cost for a folder (no API calls)."""
    try:
        est = estimate_folder_cost(input_dir, model=model, output_tokens_per_doc=output_tokens)
    except (ValueError, OSError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1)
    typer.echo(f"Documents:            {est['documents']}")
    typer.echo(f"Model:                {est['model']}")
    typer.echo(f"~Input tokens:        {est['estimated_input_tokens']:,}")
    typer.echo(f"~Output tokens:       {est['estimated_output_tokens']:,}")
    typer.echo(f"Realtime cost:        ${est['realtime_usd']:.2f}")
    typer.echo(f"Batch cost (50% off): ${est['batch_usd']:.2f}")
    typer.echo(f"You save:             ${est['savings_usd']:.2f}")
    typer.echo(f"(Pricing verified {PRICING_VERIFIED}; token counts are ~4 chars/token estimates.)")


@app.command()
def run(
    input_dir: str = typer.Argument(..., help="Folder of .txt/.md documents."),
    schema: str = typer.Option(..., "--schema", "-s", help="Path to JSON schema file."),
    model: str = typer.Option(DEFAULT_MODEL, help="Model alias."),
    prompt: str = typer.Option(None, "--prompt", help="Prompt template with a {document} placeholder."),
    prompt_file: str = typer.Option(None, "--prompt-file", help="File containing the prompt template."),
    output_dir: str = typer.Option(None, "--output-dir", "-o", help="Where to write results.json and summary.csv."),
    timeout: float = typer.Option(24 * 3600, help="Max seconds to wait for the batch."),
):
    """Run the full pipeline: batch-submit a folder, wait, validate, write results."""
    json_schema = _load_schema(schema)
    if prompt_file and prompt:
        typer.echo("Pass either --prompt or --prompt-file, not both.", err=True)
        raise typer.Exit(1)
    template = prompt
    if prompt_file:
        pf = Path(prompt_file)
        if not pf.is_file():
            typer.echo(f"Prompt file not found: {prompt_file}", err=True)
            raise typer.Exit(1)
        template = pf.read_text(encoding="utf-8")
    client = _get_client()
    try:
        summary = run_folder(
            input_dir,
            json_schema,
            prompt_template=template,
            model=model,
            output_dir=output_dir,
            client=client,
            timeout=timeout,
        )
    except Exception as exc:
        typer.echo(f"Pipeline failed: {exc}", err=True)
        raise typer.Exit(1)
    if summary["failed"]:
        raise typer.Exit(2)


@app.command("extract-one")
def extract_one(
    file: str = typer.Argument(..., help="Single .txt/.md document."),
    schema: str = typer.Option(..., "--schema", "-s", help="Path to JSON schema file."),
    model: str = typer.Option(DEFAULT_MODEL, help="Model alias."),
):
    """Realtime structured extraction of a single document (for testing a schema)."""
    json_schema = _load_schema(schema)
    path = Path(file)
    if not path.is_file():
        typer.echo(f"File not found: {file}", err=True)
        raise typer.Exit(1)
    text = path.read_text(encoding="utf-8")
    client = _get_client()
    try:
        data = extract(
            client,
            messages=[{"role": "user", "content": f"Extract the fields defined by the extraction schema from this document:\n\n{text}"}],
            json_schema=json_schema,
            model=model,
        )
    except Exception as exc:
        typer.echo(f"Extraction failed: {exc}", err=True)
        raise typer.Exit(1)
    typer.echo(json.dumps(data, indent=2))
