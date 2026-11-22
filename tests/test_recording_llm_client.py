"""RecordingLLMClient keeps every exchange; fakes are injected, no monkeypatching."""

import pytest

from mosaic.llm.client import LLMClient, RecordingLLMClient, ask


class FakeLLMClient(LLMClient):
    def __init__(self, response="<START>advice<END>"):
        self.response = response

    def query(self, prompt: str) -> str:
        return self.response


class ExplodingLLMClient(LLMClient):
    def query(self, prompt: str) -> str:
        raise RuntimeError("boom")


def test_records_prompt_response_and_timing_and_returns_the_response():
    client = RecordingLLMClient(FakeLLMClient("hello"))
    assert client.query("where?") == "hello"
    (exchange,) = client.history
    assert (exchange.prompt, exchange.response, exchange.error) == (
        "where?",
        "hello",
        None,
    )
    assert exchange.ended >= exchange.started
    assert exchange.latency >= 0


def test_records_the_error_and_raises_it_again():
    client = RecordingLLMClient(ExplodingLLMClient())
    with pytest.raises(RuntimeError, match="boom"):
        client.query("where?")
    (exchange,) = client.history
    assert exchange.response is None and exchange.error == "boom"


def test_max_history_keeps_only_the_last_exchanges():
    client = RecordingLLMClient(FakeLLMClient(), max_history=2)
    for i in range(3):
        client.query(f"q{i}")
    assert [e.prompt for e in client.history] == ["q1", "q2"]


def test_is_an_llm_client_so_it_works_with_ask():
    client = RecordingLLMClient(FakeLLMClient())
    ask({}, client=client, prompt_builder=lambda obs: "the prompt")
    assert client.history[0].prompt == "the prompt"


def test_rejects_a_non_client():
    with pytest.raises(TypeError):
        RecordingLLMClient(object())
