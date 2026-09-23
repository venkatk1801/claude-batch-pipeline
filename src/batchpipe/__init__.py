"""claude-batch-pipeline: bulk document processing with Anthropic's Batch API + structured outputs."""

from batchpipe.batches import (
    BatchError,
    BatchSubmitError,
    BatchTimeoutError,
    BatchResults,
    FailedResult,
    MissingToolCallError,
    RequestResult,
    SucceededResult,
    build_jsonl,
    build_request,
    parse_batch_results,
    poll_batch,
    run_batch,
    submit_batch,
)
from batchpipe.structured import (
    NoToolCallError,
    SchemaValidationError,
    StructuredError,
    extract,
    extraction_tool,
    tool_input_from_message,
    validate_output,
)
from batchpipe.pipeline import (
    DEFAULT_MODEL,
    estimate_folder_cost,
    read_documents,
    run_folder,
)
from batchpipe.pricing import BATCH_DISCOUNT, PRICING, cost_breakdown, estimate_cost

__version__ = "0.1.0"

__all__ = [
    "BATCH_DISCOUNT",
    "PRICING",
    "BatchError",
    "BatchResults",
    "BatchSubmitError",
    "BatchTimeoutError",
    "DEFAULT_MODEL",
    "FailedResult",
    "MissingToolCallError",
    "NoToolCallError",
    "RequestResult",
    "SchemaValidationError",
    "StructuredError",
    "SucceededResult",
    "build_jsonl",
    "build_request",
    "cost_breakdown",
    "estimate_cost",
    "estimate_folder_cost",
    "extract",
    "extraction_tool",
    "parse_batch_results",
    "poll_batch",
    "read_documents",
    "run_batch",
    "run_folder",
    "submit_batch",
    "tool_input_from_message",
    "validate_output",
]
