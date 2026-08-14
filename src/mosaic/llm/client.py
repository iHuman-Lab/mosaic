from abc import ABC, abstractmethod
from typing import Callable, Optional

from .parser import clean_response
from .process_prompts import build_prompt

llm_cache: dict = {}


def get_llm(model: str, provider: str):
    key = (provider, model)
    if key not in llm_cache:
        if provider == "openai":
            from llama_index.llms.openai import OpenAI

            llm_cache[key] = OpenAI(model=model)
        elif provider == "google":
            from llama_index.llms.google_genai import GoogleGenAI

            llm_cache[key] = GoogleGenAI(model=model)
        else:
            raise ValueError(
                f"Unknown provider: {provider!r}. Use 'openai' or 'google'."
            )
    return llm_cache[key]


class LLMClient(ABC):
    """Text-in, text-out contract for talking to an LLM backend.

    Deliberately knows nothing about SAR observations, prompts, GUI panels,
    game actions, or experiment conditions.
    """

    def __init__(self, provider: str, model: str):
        self.provider = provider
        self.model = model

    @abstractmethod
    def query(self, prompt: str) -> str:
        """Send a text prompt and return text."""


class LlamaIndexLLMClient(LLMClient):
    """LLMClient backed by llama_index (OpenAI/Google providers)."""

    def query(self, prompt: str) -> str:
        from llama_index.core.llms import ChatMessage, MessageRole

        llm = get_llm(self.model, self.provider)
        # Gemini requires at least one USER message; SYSTEM-only crashes with
        # pop from empty list. Sending prompt as USER works universally.
        messages = [ChatMessage(role=MessageRole.USER, content=prompt)]
        response = llm.chat(messages)
        content = response.message.content
        if not content:
            raise ValueError("Provider response did not contain text.")
        return content


class DummyLLMClient(LLMClient):
    """No-LLM experiment condition — returns a neutral message, no network."""

    def __init__(self):
        super().__init__(provider="dummy", model="dummy")

    def query(self, prompt: str) -> str:
        return "Currently, no commands are available."


_DEFAULT_MODELS = {
    "openai": "gpt-4o-mini",
    "google": "gemini-1.5-flash",
    "dummy": "dummy",
}


def build_llm_client(provider: str = "openai", model: Optional[str] = None) -> LLMClient:
    """Construct an LLMClient for the given provider.

    Construction never imports a provider package, contacts the network, or
    requires credentials — everything provider-specific is deferred to the
    client's first query() call.
    """
    if provider not in _DEFAULT_MODELS:
        raise ValueError(
            f"Unknown provider {provider!r}. Expected openai, google, or dummy."
        )
    if provider == "dummy":
        return DummyLLMClient()
    return LlamaIndexLLMClient(provider=provider, model=model or _DEFAULT_MODELS[provider])


def ask(
    obs: dict,
    model: Optional[str] = None,
    provider: str = "openai",
    prompt_type: str = "sparse",
    client: Optional[LLMClient] = None,
    prompt_builder: Optional[Callable[[dict], str]] = None,
    response_processor: Optional[Callable[[str], str]] = None,
) -> str:
    if client is not None and not isinstance(client, LLMClient):
        raise TypeError("client must be an LLMClient")
    if prompt_type not in ("sparse", "detailed"):
        raise ValueError(
            f"Unknown prompt_type {prompt_type!r}. Expected 'sparse' or 'detailed'."
        )
    if client is None:
        client = build_llm_client(provider=provider, model=model)
    if isinstance(client, DummyLLMClient):
        return client.query("")

    build = prompt_builder or (lambda o: build_prompt(o, prompt_type=prompt_type))
    process = response_processor or clean_response
    return process(client.query(build(obs)))
