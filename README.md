# claude-batch-pipeline

Bulk document processing with [Anthropic's Batch API](https://platform.claude.com/docs/en/api/messages-batch) + structured outputs via forced tool use.

Point it at a folder of `.txt` / `.md` files and a JSON schema. It submits every
document as one batch (50% cheaper than realtime), waits for completion, validates
each extracted record against your schema, and writes `results.json` + `summary.csv`
plus a batch-vs-realtime cost report.

```
            .txt / .md files
                  |
                  v
   +----------------------------------+
   | run_folder()                     |
   |  build requests (JSONL)          |
   |  forced tool use -> input_schema |
   +----------------+-----------------+
                    |
                    v
   +----------------------------------+
   |  Messages Batches API            |
   |  async, 50% cheaper, one batch   |
   +----------------+-----------------+
                    |
                    v
   +----------------------------------+
   |  parse + validate vs JSON schema |
   |  per-request errors isolated     |
   +----------------+-----------------+
                    |
                    v
        results.json + summary.csv
        + batch vs realtime cost
```

## Why batch?

The Batch API is a flat 50% discount on *both* input and output tokens. If your
workload is non-interactive — nightly ETL, backfilling a data warehouse,
processing a document corpus — batching is free money. See [When batching is worth it](#when-batching-is-worth-it).

## Pricing (verified 2026-09-23)

Source: [anthropic.com/pricing](https://platform.claude.com/docs/en/about-claude/pricing).
Prices change over time — the estimator warns you it's a snapshot.

| Model | Input / 1M | Output / 1M | Batch input / 1M | Batch output / 1M |
|---|---|---|---|---|
| Claude Haiku 4.5 | $1.00 | $5.00 | $0.50 | $2.50 |
| Claude Sonnet 5 | $2.00 | $10.00 | $1.00 | $5.00 |
| Claude Sonnet 4.6 | $3.00 | $15.00 | $1.50 | $7.50 |
| Claude Opus 5 | $5.00 | $25.00 | $2.50 | $12.50 |

### Example: 10,000 documents, ~2,000 input tokens each, ~300 output tokens each, Sonnet 4.6

| | Realtime | Batch (50% off) |
|---|---|---|
| Input (20M tokens) | $60.00 | $30.00 |
| Output (3M tokens) | $45.00 | $22.50 |
| **Total** | **$105.00** | **$52.50** |

## Install

```bash
pip install claude-batch-pipeline   # or: pip install -e .
```

Requires Python ≥ 3.10 and an `ANTHROPIC_API_KEY` (via env or a `.env` file).

## Quickstart

1. Estimate the cost of a folder first — no API calls:

```bash
batchpipe estimate ./docs --model claude-sonnet-4-6 --output-tokens 500
```

2. Run the pipeline against a JSON schema:

```bash
batchpipe run ./docs --schema examples/trade_schema.json --model claude-sonnet-4-6
```

This submits one batch, waits for it, validates every record, and writes
`batchpipe-output/results.json` and `batchpipe-output/summary.csv`.

3. Test a schema on a single document in realtime before burning a batch:

```bash
batchpipe extract-one ./docs/trade-01.md --schema examples/trade_schema.json
```

Try the bundled example end-to-end:

```bash
batchpipe estimate ./examples
batchpipe run ./examples --schema examples/trade_schema.json -o ./demo-output
```

## API

```python
from anthropic import Anthropic
from batchpipe import extract, run_folder, estimate_cost

client = Anthropic()

# Single realtime extraction with forced tool use + schema validation
record = extract(client, [{"role": "user", "content": text}], schema)

# Full pipeline over a folder
summary = run_folder("./docs", schema, client=client, model="claude-sonnet-4-6")
print(summary["cost"])  # {"realtime_usd": ..., "batch_usd": ..., "savings_usd": ...}
```

## When batching is worth it

**Batch when:**
- the work is async by nature (nightly jobs, corpus backfills, analytics)
- you process hundreds+ of requests with the same schema
- you can tolerate results arriving minutes-to-hours later (batches typically
  finish well inside the 24h processing window)

**Don't batch when:**
- a human is waiting on the answer (interactive apps, agents in a loop)
- you need < ~1 minute latency
- request count is tiny — the absolute savings won't beat the orchestration overhead

## Repo layout

```
src/batchpipe/
  batches.py     # Batch API wrapper: build JSONL, submit, poll w/ backoff, parse
  structured.py  # forced tool-use extraction + jsonschema validation
  pipeline.py    # run_folder(): docs -> batch -> validated results.json + summary.csv
  pricing.py     # per-model pricing + batch/realtime cost comparison
  cli.py         # typer CLI: estimate / run / extract-one
examples/        # trade_schema.json + 3 sample trade-journal docs
tests/           # fully mocked anthropic client — no network, no key needed
```

Run the tests:

```bash
pytest
```

## License

MIT — see [LICENSE](LICENSE).
