# Configuration

## `configs/experiment.yaml` (full experiment)

```yaml
game:
  display: 0              # Display index (0 = primary)
  fullscreen: false
  max_time: 10            # Mission time limit in minutes
  prompt_type: sparse     # LLM prompt verbosity
  openai_model: gpt-5-mini-2025-08-07
  num_rows: 2              # Rooms tall
  num_cols: 2              # Rooms wide
  llm_nudge_interval: 50   # Steps between automatic LLM prompts

vs:                        # Visual Search task config
  ...

mot:                       # Multi-Object Tracking task config
  ...

surveys:                   # NASA-TLX and SART config
  ...
```

## `configs/config.yml` (base / dev)

Simpler config used when running `python -m experiment.main` directly. Override individual parameters here for quick iteration.
