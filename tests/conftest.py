"""Fake anthropic client — no network, no API key."""

import pytest


class FakeUsage:
    def __init__(self, input_tokens=0, output_tokens=0):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


class FakeToolUse:
    def __init__(self, name, input):
        self.type = "tool_use"
        self.name = name
        self.input = input


class FakeText:
    def __init__(self, text):
        self.type = "text"
        self.text = text


class FakeMessage:
    def __init__(self, content, input_tokens=0, output_tokens=0):
        self.type = "message"
        self.role = "assistant"
        self.content = content
        self.usage = FakeUsage(input_tokens, output_tokens)
        self.stop_reason = "tool_use"


class FakeResult:
    def __init__(self, type, message=None, error=None):
        self.type = type
        self.message = message
        self.error = error


class FakeErrorDetail:
    def __init__(self, type, message):
        self.type = type
        self.message = message


class FakeIndividualResponse:
    def __init__(self, custom_id, result):
        self.custom_id = custom_id
        self.result = result


class FakeBatch:
    _counter = 0

    def __init__(self, status="in_progress", request_counts=None):
        FakeBatch._counter += 1
        self.id = f"msgbatch_test_{FakeBatch._counter}"
        self.processing_status = status
        self.request_counts = request_counts or {
            "processing": 0,
            "succeeded": 0,
            "errored": 0,
            "canceled": 0,
            "expired": 0,
        }


class FakeBatchesAPI:
    """Mimics client.messages.batches. Configure retrieve_statuses to script polling."""

    def __init__(self):
        self.created_requests = None
        self.retrieve_statuses = ["in_progress", "ended"]
        self.retrieve_calls = 0
        self.individual_responses = []

    def create(self, requests):
        self.created_requests = requests
        return FakeBatch(status=self.retrieve_statuses[0] if self.retrieve_statuses else "ended")

    def retrieve(self, batch_id):
        idx = min(self.retrieve_calls, len(self.retrieve_statuses) - 1)
        self.retrieve_calls += 1
        return FakeBatch(status=self.retrieve_statuses[idx])

    def results(self, batch_id):
        return list(self.individual_responses)


class FakeMessagesAPI:
    def __init__(self, create_response=None):
        self.batches = FakeBatchesAPI()
        self.create_response = create_response
        self.create_calls = []

    def create(self, **kwargs):
        self.create_calls.append(kwargs)
        if isinstance(self.create_response, Exception):
            raise self.create_response
        return self.create_response


class FakeClient:
    def __init__(self, create_response=None):
        self.messages = FakeMessagesAPI(create_response=create_response)


def succeeded_response(custom_id, output, input_tokens=100, output_tokens=50):
    return FakeIndividualResponse(
        custom_id,
        FakeResult(
            "succeeded",
            message=FakeMessage(
                [FakeToolUse("extract_data", output)],
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            ),
        ),
    )


def errored_response(custom_id, message="invalid_request_error: bad params"):
    return FakeIndividualResponse(
        custom_id,
        FakeResult("errored", error=FakeErrorDetail("invalid_request_error", message)),
    )


@pytest.fixture
def client():
    return FakeClient()
