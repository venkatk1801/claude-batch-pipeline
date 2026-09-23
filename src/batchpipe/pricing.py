"""Model pricing and cost estimation.

Pricing is per 1M tokens, verified 2026-09-23 against Anthropic's official
pricing page (via anthropic.com/pricing references). Prices change over time —
treat these as a snapshot and check https://platform.claude.com/docs/en/about-claude/pricing
for the live numbers.

The Batch API costs 50% of realtime pricing for both input and output tokens.
"""

from __future__ import annotations

# (input $/1M tokens, output $/1M tokens), realtime rates.
PRICING: dict[str, tuple[float, float]] = {
    "claude-haiku-4-5": (1.00, 5.00),
    "claude-sonnet-5": (2.00, 10.00),
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-opus-5": (5.00, 25.00),
}

BATCH_DISCOUNT = 0.5
PRICING_VERIFIED = "2026-09-23"


def known_models() -> list[str]:
    """Model aliases this module has pricing for."""
    return sorted(PRICING)


def rates(model: str, batch: bool = True) -> tuple[float, float]:
    """Return (input $/1M, output $/1M) for a model, with or without batch discount."""
    if model not in PRICING:
        raise ValueError(
            f"Unknown model {model!r}. Known models: {', '.join(known_models())}"
        )
    in_rate, out_rate = PRICING[model]
    if batch:
        in_rate *= BATCH_DISCOUNT
        out_rate *= BATCH_DISCOUNT
    return in_rate, out_rate


def estimate_cost(
    input_tokens: int,
    output_tokens: int,
    model: str,
    batch: bool = True,
) -> float:
    """Estimate USD cost for a token volume. ``batch=True`` applies the 50% Batch API discount."""
    if input_tokens < 0 or output_tokens < 0:
        raise ValueError("Token counts must be non-negative")
    in_rate, out_rate = rates(model, batch=batch)
    return (input_tokens / 1_000_000) * in_rate + (output_tokens / 1_000_000) * out_rate


def cost_breakdown(
    input_tokens: int,
    output_tokens: int,
    model: str,
) -> dict[str, float]:
    """Compare realtime vs batch cost for the same token volume."""
    realtime = estimate_cost(input_tokens, output_tokens, model, batch=False)
    batch_cost = estimate_cost(input_tokens, output_tokens, model, batch=True)
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "realtime_usd": round(realtime, 4),
        "batch_usd": round(batch_cost, 4),
        "savings_usd": round(realtime - batch_cost, 4),
    }
