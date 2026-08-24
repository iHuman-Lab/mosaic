# Testing

```bash
pytest tests/
```

MOSAIC's tests favor dependency injection over monkeypatching: seams like `LLMClient` are
exercised with fakes passed directly into constructors, not by patching module attributes.

| Test file | What it validates |
| --- | --- |
| `test_environment.py` | Environment initialization, step, reset |
| `test_actions.py` | `RescueAction`'s SAR event emission (`info["events"]`) |
| `test_lava.py` | Lava placement doesn't block all paths |
| `test_lava_50_percent.py` | High-density lava placement edge cases |
| `test_reachability.py` | All objects (victims, keys) are reachable from spawn |
| `test_strict_reachability.py` | Strict BFS reachability with `unblocking=False` |
| `test_victim_placement.py` | Split between generic victim placement and study calibration |
| `test_pacing.py` | `TunedPickupVictimEnv`'s victim-health decay (the only decay tick in the codebase) |
| `test_llm_api.py` | The `mosaic.llm.client` contract: `LLMClient`, `DummyLLMClient`, and the `ask()` pipeline — no providers, no real requests |
| `test_experiment_llm.py` | `experiment.llm` provider selection, model defaults, and `LlamaIndexLLMClient`, via a fake object — never calls the real `get_llm()` |
| `test_gui_api.py` | The GUI/LLM dependency-injection boundary: `User`, `SAREnvGUI`, `ChatPanel` receive an already-built `LLMClient` |
| `test_gui_feedback.py` | `EdgeVignette` feedback behavior |
| `playground.py` | Manual development / debugging entrypoint |
