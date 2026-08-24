# LLM Integration

MOSAIC separates the **contract** (in `mosaic`) from **provider wiring** (in `experiment`):

- `mosaic.llm.client.LLMClient` — abstract base class; any implementation just needs `query(prompt) -> str`.
- `mosaic.llm.client.DummyLLMClient` — a no-op implementation for testing/demoing without API keys.
- `experiment.llm.LlamaIndexLLMClient` — concrete implementation backed by `llama_index`, supporting OpenAI and Google providers.
- `experiment.llm.build_llm_client(provider, model)` — factory that returns a `DummyLLMClient` when no provider is configured, otherwise a configured `LlamaIndexLLMClient`.

```python
from experiment.llm import build_llm_client

client = build_llm_client(provider="openai", model="gpt-4o-mini")
response = client.query(prompt_string)
```

## Supported providers

| Provider | Example models |
| --- | --- |
| OpenAI | `gpt-4o-mini`, `gpt-5-mini-2025-08-07` |
| Google | `gemini-2.5-flash` |
| Dummy | No-op client for testing without API keys |

## How it works

1. Player presses `Alt` (or the auto-nudge interval fires).
2. `mosaic/llm/process_prompts.py` builds a structured prompt from the current `GameObservation`:
   agent position and facing direction, visible/reachable objects with pathfinding distances,
   current mission status, and prompt type (`sparse` = minimal, `detailed` = full grid scan + strategy).
3. The prompt is sent to the configured `LLMClient` asynchronously.
4. `mosaic/llm/parser.py` cleans the response (removes coordinates, reformats object labels).
5. The cleaned suggestion appears in the `ChatPanel`.

## Configuration

```yaml
# configs/experiment.yaml
game:
  prompt_type: sparse          # "sparse" or "detailed"
  openai_model: gpt-5-mini-2025-08-07
  llm_nudge_interval: 50       # auto-prompt every N steps (0 = disabled)
```

Tests exercise `LLMClient` consumers by passing a fake implementation through the constructor
(dependency injection) rather than monkeypatching the client.
