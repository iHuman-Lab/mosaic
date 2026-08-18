"""Tests for the mosaic.llm.client API: the LLMClient contract, the neutral
DummyLLMClient fallback, and the ask() SAR-prompting pipeline.

mosaic.llm.client knows nothing about providers, model names, or LlamaIndex.
All tests use fake/injected clients — no real OpenAI or Google requests, and
no monkeypatching.
"""

import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from mosaic.llm.client import DummyLLMClient, LLMClient, ask
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


class FakeLLMClient(LLMClient):
    """LLMClient stub that records the prompt it received and returns a
    canned response. Requires no provider or model — the base contract is
    text-in, text-out only."""

    def __init__(self, response="<START>advice<END>"):
        self.response = response
        self.prompts = []

    def query(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.response


class ExplodingLLMClient(LLMClient):
    def query(self, prompt: str) -> str:
        raise RuntimeError("boom")


# --- LLMClient / DummyLLMClient ------------------------------------------


def test_llm_client_is_abstract_and_requires_no_provider_or_model():
    fake = FakeLLMClient(response="hello")
    assert fake.query("prompt text") == "hello"
    assert fake.prompts == ["prompt text"]
    assert not hasattr(fake, "provider")
    assert not hasattr(fake, "model")


def test_dummy_llm_client_returns_fixed_message_with_no_network():
    client = DummyLLMClient()
    assert client.query("anything") == "Currently, no commands are available."
    assert client.query("") == "Currently, no commands are available."


# --- ask() ---------------------------------------------------------------


def test_ask_rejects_non_llmclient():
    with pytest.raises(TypeError):
        ask(_fake_obs(), client=object())


def test_ask_rejects_invalid_prompt_type_with_default_builder():
    with pytest.raises(ValueError, match="prompt_type"):
        ask(_fake_obs(), client=FakeLLMClient(), prompt_type="medium")


def test_ask_skips_prompt_type_validation_when_custom_builder_given():
    fake = FakeLLMClient(response="<START>ok<END>")
    result = ask(
        _fake_obs(),
        client=fake,
        prompt_type="medium",
        prompt_builder=lambda obs: "CUSTOM PROMPT",
    )
    assert fake.prompts == ["CUSTOM PROMPT"]
    assert result == "ok"


def test_ask_dummy_client_needs_no_real_observation():
    assert ask(None, client=DummyLLMClient()) == "Currently, no commands are available."
    assert ask("garbage", client=DummyLLMClient()) == "Currently, no commands are available."


def test_ask_dummy_client_response_still_passes_through_response_processor():
    result = ask(
        None,
        client=DummyLLMClient(),
        response_processor=lambda text: text.upper(),
    )
    assert result == "CURRENTLY, NO COMMANDS ARE AVAILABLE."


def test_ask_uses_injected_client_and_default_prompt_builder():
    obs = _fake_obs()
    fake = FakeLLMClient(response="<START>rescue the victim<END>")

    result = ask(obs, client=fake, prompt_type="detailed")

    assert fake.prompts == [build_prompt(obs, prompt_type="detailed")]
    assert result == clean_response(fake.response)


def test_ask_custom_prompt_builder_bypasses_default_builder():
    """An obs shape the default build_prompt() cannot handle — if ask() ever
    fell back to the default builder instead of the injected one, this test
    fails with a KeyError/AttributeError from build_prompt itself."""
    obs = {"custom": "value"}
    fake = FakeLLMClient(response="<START>ok<END>")

    result = ask(
        obs,
        client=fake,
        prompt_builder=lambda o: f"Custom: {o['custom']}",
    )

    assert fake.prompts == ["Custom: value"]
    assert result == "ok"


def test_ask_custom_response_processor_bypasses_clean_response():
    fake = FakeLLMClient(response="<START>raw text<END>")
    result = ask(
        _fake_obs(),
        client=fake,
        response_processor=lambda text: text.upper(),
    )
    assert result == "<START>RAW TEXT<END>"


def test_ask_default_path_unaffected_without_overrides():
    obs = _fake_obs()
    fake = FakeLLMClient(response="<START>the red key<END>")

    result = ask(obs, client=fake, prompt_type="sparse")

    assert fake.prompts == [build_prompt(obs, prompt_type="sparse")]
    assert result == clean_response(fake.response)


def test_ask_propagates_client_errors():
    with pytest.raises(RuntimeError, match="boom"):
        ask(_fake_obs(), client=ExplodingLLMClient())


# --- import isolation -------------------------------------------------------


def test_mosaic_llm_client_does_not_import_llama_index():
    """Run in a subprocess so this is never order-dependent on some other
    test in the same session having already imported llama_index."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import mosaic.llm.client, sys; "
            "assert not any(m.startswith('llama_index') for m in sys.modules), "
            "'llama_index was imported by mosaic.llm.client'",
        ],
        cwd=str(REPO_ROOT / "src"),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


# --- packaging -------------------------------------------------------------


def test_package_data_is_included_in_built_wheel(tmp_path):
    """Build an actual wheel and inspect it — proves the package-data
    entries work, not just that the pyproject.toml lines and source files
    exist."""
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
    assert "mosaic/gui/theme.json" in names
    assert "mosaic/gui/compass_inv.png" in names
