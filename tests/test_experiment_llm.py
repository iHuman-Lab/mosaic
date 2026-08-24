"""Tests for experiment.llm: provider selection, model defaults, and the
LlamaIndex-backed LLMClient. No test here calls the real get_llm() (which
would import real provider packages and require credentials) — backend
behavior is exercised entirely through a fake object passed directly into
LlamaIndexLLMClient's constructor. No monkeypatching.
"""

from types import SimpleNamespace

import pytest

from experiment.llm import LlamaIndexLLMClient, build_llm_client
from mosaic.llm.client import DummyLLMClient


class _FakeBackend:
    """Stand-in for a llama_index llm object's .chat() interface."""

    def __init__(self, text="backend reply"):
        self.text = text
        self.received_messages = None

    def chat(self, messages):
        self.received_messages = messages
        return SimpleNamespace(message=SimpleNamespace(content=self.text))


def test_build_llm_client_dummy_returns_dummy_client():
    client = build_llm_client("dummy")
    assert isinstance(client, DummyLLMClient)
    assert client.query("anything") == "Currently, no commands are available."


def test_build_llm_client_unknown_provider_raises():
    with pytest.raises(ValueError, match="bogus"):
        build_llm_client("bogus")


def test_build_llm_client_openai_construction_is_lazy():
    client = build_llm_client("openai")
    assert isinstance(client, LlamaIndexLLMClient)
    assert client.provider == "openai"
    assert client.model == "gpt-4o-mini"
    assert client._llm is None


def test_build_llm_client_google_default_model_is_lazy():
    client = build_llm_client("google")
    assert client.provider == "google"
    assert client.model == "gemini-1.5-flash"
    assert client._llm is None


def test_build_llm_client_explicit_model_overrides_default():
    client = build_llm_client("openai", model="gpt-4-turbo")
    assert client.model == "gpt-4-turbo"


def test_llama_index_client_queries_injected_backend_directly():
    pytest.importorskip("llama_index")
    backend = _FakeBackend(text="<START>go north<END>")
    client = LlamaIndexLLMClient(provider="google", model="gemini-1.5-flash", llm=backend)

    result = client.query("some prompt")

    assert result == "<START>go north<END>"
    [message] = backend.received_messages
    assert message.content == "some prompt"
    # backend was supplied at construction time — never replaced
    assert client._llm is backend


def test_llama_index_client_raises_on_empty_response():
    pytest.importorskip("llama_index")
    client = LlamaIndexLLMClient(
        provider="openai", model="test-model", llm=_FakeBackend(text="")
    )
    with pytest.raises(ValueError):
        client.query("prompt")
