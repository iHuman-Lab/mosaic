from typing import Optional

from mosaic.llm.client import DummyLLMClient, LLMClient

_DEFAULT_MODELS = {
    "openai": "gpt-4o-mini",
    "google": "gemini-1.5-flash",
}

_llm_cache: dict = {}


def get_llm(provider: str, model: str):
    key = (provider, model)
    if key not in _llm_cache:
        if provider == "openai":
            from llama_index.llms.openai import OpenAI

            _llm_cache[key] = OpenAI(model=model)
        elif provider == "google":
            from llama_index.llms.google_genai import GoogleGenAI

            _llm_cache[key] = GoogleGenAI(model=model)
        else:
            raise ValueError(
                f"Unknown provider: {provider!r}. Use 'openai' or 'google'."
            )
    return _llm_cache[key]


class LlamaIndexLLMClient(LLMClient):
    """LLMClient backed by llama_index (OpenAI/Google providers).

    The backend can be supplied directly (e.g. a fake for tests); otherwise
    it is resolved lazily on first query() via get_llm(), so construction
    never imports a provider package, contacts the network, or requires
    credentials.
    """

    def __init__(self, provider: str, model: str, llm=None):
        self.provider = provider
        self.model = model
        self._llm = llm

    def query(self, prompt: str) -> str:
        from llama_index.core.llms import ChatMessage, MessageRole

        if self._llm is None:
            self._llm = get_llm(self.provider, self.model)
        # Gemini requires at least one USER message; SYSTEM-only crashes with
        # pop from empty list. Sending prompt as USER works universally.
        messages = [ChatMessage(role=MessageRole.USER, content=prompt)]
        response = self._llm.chat(messages)
        content = response.message.content
        if not content:
            raise ValueError("Provider response did not contain text.")
        return content


def build_llm_client(provider: str = "openai", model: Optional[str] = None) -> LLMClient:
    """Construct an LLMClient for the given provider.

    Construction never imports a provider package, contacts the network, or
    requires credentials — everything provider-specific is deferred to the
    client's first query() call.
    """
    if provider == "dummy":
        return DummyLLMClient()
    if provider not in _DEFAULT_MODELS:
        raise ValueError(
            f"Unknown provider {provider!r}. Expected openai, google, or dummy."
        )
    resolved_model = model or _DEFAULT_MODELS[provider]
    return LlamaIndexLLMClient(provider=provider, model=resolved_model)
