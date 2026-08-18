from abc import ABC, abstractmethod
from typing import Callable, Optional

from .parser import clean_response
from .process_prompts import build_prompt


class LLMClient(ABC):
    """Text-in, text-out contract for talking to an LLM backend.

    Deliberately knows nothing about SAR observations, prompts, GUI panels,
    game actions, or experiment conditions — and nothing about providers or
    model names, since those aren't universal properties of an LLM client.
    """

    @abstractmethod
    def query(self, prompt: str) -> str:
        """Send a text prompt and return text."""


class DummyLLMClient(LLMClient):
    """No-LLM fallback — returns a neutral message, no network."""

    def query(self, prompt: str) -> str:
        return "Currently, no commands are available."


def ask(
    obs: dict,
    client: LLMClient,
    prompt_type: str = "sparse",
    prompt_builder: Optional[Callable[[dict], str]] = None,
    response_processor: Optional[Callable[[str], str]] = None,
) -> str:
    if not isinstance(client, LLMClient):
        raise TypeError("client must be an LLMClient")
    if prompt_builder is None and prompt_type not in ("sparse", "detailed"):
        raise ValueError(
            f"Unknown prompt_type {prompt_type!r}. Expected 'sparse' or 'detailed'."
        )

    if prompt_builder is None:
        build = lambda observation: build_prompt(observation, prompt_type=prompt_type)
    else:
        build = prompt_builder

    process = response_processor if response_processor is not None else clean_response

    if isinstance(client, DummyLLMClient):
        raw_response = client.query("")
    else:
        prompt = build(obs)
        if not isinstance(prompt, str):
            raise TypeError("prompt_builder must return a string")
        raw_response = client.query(prompt)

    if not isinstance(raw_response, str):
        raise TypeError("LLMClient.query() must return a string")

    response = process(raw_response)
    if not isinstance(response, str):
        raise TypeError("response_processor must return a string")

    return response
