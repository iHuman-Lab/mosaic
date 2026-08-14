"""Tests for the mosaic.llm.client API: LLMClient contract, built-in clients,
build_llm_client factory, and the ask() SAR-prompting pipeline.

All tests use fake/injected clients — no real OpenAI or Google requests.
"""

import subprocess
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from mosaic.llm import client as client_module
from mosaic.llm.client import (
    DummyLLMClient,
    LLMClient,
    LlamaIndexLLMClient,
    ask,
    build_llm_client,
)
from mosaic.llm.parser import clean_response
from mosaic.llm.process_prompts import build_prompt

REPO_ROOT = Path(__file__).resolve().parent.parent


def _fake_obs(**overrides) -> dict:
    """A minimal obs dict with an empty 3x3 grid — enough for build_prompt()
    to run without needing any objects placed on the map."""
    obs = {
        "grid": [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
        "agent_x": 1,
        "agent_y": 1,
        "agent_dir": 0,
        "carrying": None,
        "step_count": 0,
        "max_steps": 100,
        "room_size": 3,
        "num_rows": 1,
        "num_cols": 1,
        "remaining_victims": 0,
        "saved_victims": 0,
        "victim_health": {},
    }
    obs.update(overrides)
    return obs


class FakeClient(LLMClient):
    """LLMClient stub that records the prompt it received and returns a
    canned response."""

    def __init__(self, response="<START>advice<END>", provider="fake", model="fake-model"):
        super().__init__(provider=provider, model=model)
        self.response = response
        self.received_prompt = None

    def query(self, prompt: str) -> str:
        self.received_prompt = prompt
        return self.response


class _FakeBackend:
    """Stand-in for a llama_index llm object's .chat() interface."""

    def __init__(self, text="backend reply"):
        self.text = text
        self.received_messages = None

    def chat(self, messages):
        self.received_messages = messages
        return SimpleNamespace(message=SimpleNamespace(content=self.text))


# --- build_llm_client --------------------------------------------------


def test_build_llm_client_openai_default():
    client = build_llm_client("openai")
    assert isinstance(client, LlamaIndexLLMClient)
    assert client.provider == "openai"
    assert client.model == "gpt-4o-mini"


def test_build_llm_client_google_default_model():
    client = build_llm_client("google")
    assert client.provider == "google"
    assert client.model == "gemini-1.5-flash"


def test_build_llm_client_dummy():
    client = build_llm_client("dummy")
    assert isinstance(client, DummyLLMClient)
    assert client.provider == "dummy"
    assert client.query("anything") == "Currently, no commands are available."


def test_build_llm_client_unknown_provider_raises():
    with pytest.raises(ValueError, match="local"):
        build_llm_client("local")


def test_build_llm_client_construction_is_lazy(monkeypatch):
    """Constructing a client must not import a provider package, touch the
    llm_cache, or otherwise contact a backend."""
    monkeypatch.setitem(sys.modules, "llama_index.llms.openai", None)
    sys.modules.pop("llama_index.llms.openai", None)

    client_module.llm_cache.clear()
    build_llm_client("openai")

    assert "llama_index.llms.openai" not in sys.modules
    assert client_module.llm_cache == {}


# --- LLMClient contract / LlamaIndexLLMClient ---------------------------


def test_llm_client_query_round_trips_text():
    fake = FakeClient(response="hello")
    assert fake.query("prompt text") == "hello"
    assert fake.received_prompt == "prompt text"


def test_llama_index_client_uses_get_llm_with_own_provider_and_model(monkeypatch):
    """build_llm_client("google") must resolve to the Google default model,
    not silently fall back to the OpenAI default — verified via a
    monkeypatched get_llm so no real Google package is imported."""
    calls = []

    def fake_get_llm(model, provider):
        calls.append((model, provider))
        return _FakeBackend(text="<START>go north<END>")

    monkeypatch.setattr(client_module, "get_llm", fake_get_llm)

    client = build_llm_client("google")
    result = client.query("some prompt")

    assert calls == [("gemini-1.5-flash", "google")]
    assert result == "<START>go north<END>"


def test_llama_index_client_raises_on_empty_response(monkeypatch):
    monkeypatch.setattr(
        client_module, "get_llm", lambda model, provider: _FakeBackend(text="")
    )
    client = build_llm_client("openai")
    with pytest.raises(ValueError):
        client.query("prompt")


# --- ask() ---------------------------------------------------------------


def test_ask_rejects_non_llmclient():
    with pytest.raises(TypeError):
        ask(_fake_obs(), client=object())


def test_ask_rejects_invalid_prompt_type():
    with pytest.raises(ValueError, match="prompt_type"):
        ask(_fake_obs(), client=FakeClient(), prompt_type="medium")


def test_ask_rejects_invalid_prompt_type_even_in_dummy_mode():
    with pytest.raises(ValueError, match="prompt_type"):
        ask(None, provider="dummy", prompt_type="medium")


def test_ask_dummy_mode_needs_no_observation_or_network():
    assert ask(None, provider="dummy") == "Currently, no commands are available."
    assert ask("garbage", provider="dummy") == "Currently, no commands are available."


def test_ask_uses_injected_client_and_builds_expected_prompt():
    obs = _fake_obs()
    fake = FakeClient(response="<START>rescue the victim<END>")

    result = ask(obs, client=fake, prompt_type="detailed")

    assert fake.received_prompt == build_prompt(obs, prompt_type="detailed")
    assert result == clean_response(fake.response)


def test_ask_resolves_google_default_model_when_no_model_given(monkeypatch):
    calls = []

    def fake_get_llm(model, provider):
        calls.append((model, provider))
        return _FakeBackend(text="<START>ok<END>")

    monkeypatch.setattr(client_module, "get_llm", fake_get_llm)

    ask(_fake_obs(), provider="google")

    assert calls == [("gemini-1.5-flash", "google")]


def test_ask_existing_call_shape_with_explicit_model_still_works(monkeypatch):
    """The exact call shape used by src/mosaic/gui/user.py today."""
    calls = []

    def fake_get_llm(model, provider):
        calls.append((model, provider))
        return _FakeBackend(text="<START>go east<END>")

    monkeypatch.setattr(client_module, "get_llm", fake_get_llm)

    result = ask(
        _fake_obs(), model="gpt-4o-mini", provider="openai", prompt_type="sparse"
    )

    assert calls == [("gpt-4o-mini", "openai")]
    assert result == "go east"


def test_ask_prompt_builder_override_skips_build_prompt(monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError("build_prompt should not be called")

    monkeypatch.setattr(client_module, "build_prompt", boom)

    fake = FakeClient(response="<START>ok<END>")
    result = ask(
        _fake_obs(),
        client=fake,
        prompt_builder=lambda obs: "CUSTOM PROMPT",
    )

    assert fake.received_prompt == "CUSTOM PROMPT"
    assert result == "ok"


def test_ask_response_processor_override_skips_clean_response(monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError("clean_response should not be called")

    monkeypatch.setattr(client_module, "clean_response", boom)

    fake = FakeClient(response="<START>raw text<END>")
    result = ask(
        _fake_obs(),
        client=fake,
        response_processor=lambda text: text.upper(),
    )

    assert result == "<START>RAW TEXT<END>"


def test_ask_default_path_unaffected_without_overrides():
    obs = _fake_obs()
    fake = FakeClient(response="<START>the red key<END>")

    result = ask(obs, client=fake, prompt_type="sparse")

    assert fake.received_prompt == build_prompt(obs, prompt_type="sparse")
    assert result == clean_response(fake.response)


# --- packaging -------------------------------------------------------------


def test_prompts_yaml_is_included_in_built_wheel(tmp_path):
    """Build an actual wheel and inspect it — proves the package-data entry
    works, not just that the pyproject.toml line and source file exist."""
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            str(REPO_ROOT),
            "--no-deps",
            "--no-build-isolation",
            "--wheel-dir",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr

    wheels = list(tmp_path.glob("mosaic-*.whl"))
    assert wheels, "no mosaic wheel was built"

    with zipfile.ZipFile(wheels[0]) as zf:
        names = zf.namelist()
    assert "mosaic/llm/prompts.yaml" in names
