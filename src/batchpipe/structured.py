"""Structured extraction via forced tool use.

Forcing the model to call a tool whose ``input_schema`` is your JSON schema is
the most reliable way to get structured output from Claude: the model *must*
produce a valid tool call, and ``tool_input`` then holds the extracted data.
"""

from __future__ import annotations

import jsonschema

DEFAULT_TOOL_NAME = "extract_data"


class StructuredError(Exception):
    """Base class for structured-extraction errors."""


class NoToolCallError(StructuredError):
    """The model did not call the extraction tool (unexpected with forced tool use)."""


# Alias kept for backwards compatibility with earlier versions of this module.
MissingToolCallError = NoToolCallError


class SchemaValidationError(StructuredError):
    """Extracted data did not conform to the JSON schema."""


def extraction_tool(
    json_schema: dict,
    name: str = DEFAULT_TOOL_NAME,
    description: str = "Extract structured data matching the provided schema.",
) -> dict:
    """Build the tool definition whose input_schema is the extraction schema."""
    return {"name": name, "description": description, "input_schema": json_schema}


def tool_input_from_message(message, tool_name: str = DEFAULT_TOOL_NAME):
    """Pull the tool input payload out of a Message.

    Raises NoToolCallError when the message contains no matching tool_use block.
    """
    for block in getattr(message, "content", []) or []:
        if getattr(block, "type", "") == "tool_use" and getattr(block, "name", "") == tool_name:
            return getattr(block, "input", {})
    # Helpful context for debugging: what did the model actually return?
    preview = []
    for block in getattr(message, "content", []) or []:
        btype = getattr(block, "type", "")
        if btype == "text":
            text = getattr(block, "text", "")
            preview.append(text[:200])
        else:
            preview.append(f"<{btype}>")
    raise NoToolCallError(
        f"Expected a '{tool_name}' tool call but found none. "
        f"Model response preview: {' | '.join(preview) or '(empty)'}"
    )


def validate_output(data, json_schema: dict):
    """Validate ``data`` against ``json_schema``; return ``data`` if valid.

    Raises SchemaValidationError with the JSON path of the failure, the
    validator's message, and the offending value.
    """
    try:
        jsonschema.validate(data, json_schema)
    except jsonschema.ValidationError as exc:
        path = "/" + "/".join(str(p) for p in exc.absolute_path) if exc.absolute_path else "(root)"
        raise SchemaValidationError(
            f"Schema validation failed at {path}: {exc.message}. "
            f"Offending value: {exc.instance!r}"
        ) from exc
    return data


def extract(
    client,
    messages: list[dict],
    json_schema: dict,
    model: str = "claude-sonnet-4-6",
    max_tokens: int = 4096,
    tool_name: str = DEFAULT_TOOL_NAME,
):
    """Single realtime structured extraction via forced tool use.

    Returns the extracted data as a validated dict. Raises NoToolCallError or
    SchemaValidationError on failure.
    """
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        tools=[extraction_tool(json_schema, name=tool_name)],
        tool_choice={"type": "tool", "name": tool_name},
        messages=messages,
    )
    data = tool_input_from_message(response, tool_name)
    return validate_output(data, json_schema)
