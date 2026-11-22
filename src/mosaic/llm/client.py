import time
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass
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


@dataclass(frozen=True)
class LLMExchange:
    """One call to an LLM: what was sent, what came back, and when."""

    prompt: str
    response: Optional[str]  # None if the call raised
    error: Optional[str]  # str(exception) if the call raised
    started: float  # time.time() when the query was sent
    ended: float  # time.time() when it returned or raised

    @property
    def latency(self) -> float:
        """Seconds between sending the prompt and getting the answer."""
        return self.ended - self.started


class RecordingLLMClient(LLMClient):
    """Wraps any LLMClient and keeps every exchange in ``history``.

    Use it when you want to save the prompts, responses, errors, and timing:

        client = RecordingLLMClient(my_client)
        ...
        for exchange in client.history:
            print(exchange.prompt, exchange.response, exchange.latency)

    ``max_history`` keeps only the last N exchanges (prompts hold the whole
    map, so a long session can grow); the default keeps all of them.
    Exceptions are recorded and then raised again unchanged.
    """

    def __init__(self, client: LLMClient, max_history: Optional[int] = None):
        if not isinstance(client, LLMClient):
            raise TypeError("client must be an LLMClient")
        self.client = client
        self.history: deque[LLMExchange] = deque(maxlen=max_history)

    def query(self, prompt: str) -> str:
        started = time.time()
        try:
            response = self.client.query(prompt)
        except Exception as e:
            self.history.append(
                LLMExchange(prompt, None, str(e), started, time.time())
            )
            raise
        self.history.append(
            LLMExchange(prompt, response, None, started, time.time())
        )
        return response


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
