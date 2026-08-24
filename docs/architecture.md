# Architecture

`mosaic/` is the installable package (`pyproject.toml`'s `packages.find`) and ships generic,
neutral defaults — no study-specific tuning. `experiment/` (excluded from packaging) holds this
lab's calibrated parameters, orchestration, and entry points, wired on top of `mosaic/`'s
injection points.

```text
mosaic/
├── src/
│   └── mosaic/
│       ├── core/
│       │   ├── camera.py        # Camera strategy implementations
│       │   ├── placers.py       # Placer ABC
│       │   └── level.py         # SARLevelGen base class
│       ├── sar/
│       │   ├── env.py           # PickupVictimEnv (main environment), build_sar_env() factory
│       │   ├── objects.py       # Victim and FakeVictim classes
│       │   ├── actions.py       # RescueAction + RescueRewards (neutral defaults)
│       │   ├── observations.py  # GameObservation processor
│       │   ├── instructions.py  # PickupAllVictimsInstr mission
│       │   └── placers.py       # VictimPlacer (neutral health), LavaPlacer, LockedRoomPlacer
│       ├── gui/
│       │   ├── main.py          # SAREnvGUI controller (constructor-injectable components)
│       │   ├── user.py          # Keyboard input + LLM threading
│       │   ├── info.py          # InfoPanel (stats, object table)
│       │   ├── chat.py          # ChatPanel (LLM messages)
│       │   ├── feedback.py      # EdgeVignette (configurable edge feedback)
│       │   └── theme.json       # pygame_gui UI theme
│       ├── llm/
│       │   ├── client.py        # LLMClient contract (ABC) + DummyLLMClient
│       │   ├── parser.py        # LLM response cleaning
│       │   ├── process_prompts.py  # Prompt generation from game state
│       │   └── pathfinding.py   # Pathfinding queries for LLM context
│       └── tutorial_env.py      # Single-room tutorial environment
├── experiment/                  # excluded from the installable package
│   ├── main.py                  # Entry point: interactive GUI mode (python -m experiment.main)
│   ├── experiment_main.py       # Entry point: full research experiment
│   ├── llm.py                   # build_llm_client() — provider wiring (OpenAI/Google via llama_index)
│   ├── replay.py                # Replay recorded sessions
│   ├── utils.py                 # Utility classes (ColorPrint, skip_run)
│   ├── placers.py               # LavaRiskVictimPlacer — this study's health/decay tuning
│   ├── pacing.py                # TunedPickupVictimEnv — this study's pacing formula
│   ├── game.py                  # SARGame task with LSL streaming; wires tuned params into mosaic
│   ├── tutorial.py              # SARTutorial task
│   └── instructions.yaml        # Participant-facing instructions
├── configs/
│   ├── experiment.yaml          # Full experiment configuration
│   └── config.yml               # Base configuration
├── tests/
├── docs/
├── mkdocs.yml
├── requirements.txt
└── pyproject.toml
```

## The mosaic / experiment split

`mosaic/` defines contracts and generic defaults; `experiment/` supplies this lab's specific
implementations and wires them in via constructor injection. For example:

- `mosaic.llm.client.LLMClient` is an abstract contract (`query(prompt) -> str`); `mosaic.llm.client.DummyLLMClient`
  is a no-op implementation useful for testing without API keys. The concrete provider-backed
  implementation (`LlamaIndexLLMClient`, wrapping OpenAI/Google via `llama_index`) and its
  `build_llm_client()` factory live in `experiment/llm.py`, not in `mosaic`.
- `mosaic.core.placers.Placer` is an abstract placement contract; `mosaic.sar.placers.VictimPlacer`
  ships neutral defaults, while `experiment.placers.LavaRiskVictimPlacer` carries this study's
  tuned health/decay behavior.
- `mosaic.gui.main.SAREnvGUI` accepts every major component (`llm_client`, `prompt_builder`,
  `response_processor`, `user`, `info_panel`, `chat_panel`, `vignette`) as constructor
  parameters, defaulting to the neutral `mosaic` implementations when omitted — callers (like
  `experiment/`) compose it from already-built instances rather than subclassing or passing
  partial factories.

## Design patterns

- **Strategy Pattern** — Camera system (`core/camera.py`): swap strategies at runtime via `env.switch_camera(...)`.
- **Dependency Injection** — `Placer`, `LLMClient`, and the GUI's panels/vignette are all injected as
  constructed instances rather than looked up or subclassed.
- **Factory function** — `build_sar_env()` constructs a fully configured `PickupVictimEnv` from
  already-built placer instances.
- **Template Method** — `SARLevelGen` defines level generation hooks for subclasses (`PickupVictimEnv`, `TutorialEnv`).

## High-level data flow

```text
Keyboard Input
      │
      ▼
User.handle_key()
      │
      ▼
PickupVictimEnv.step(action)
  ├── Move / rotate agent
  ├── Trigger RescueAction (pickup)
  ├── Deplete victim health
  ├── Check mission state
  └── Return observation dict
      │
      ▼
GameObservation.process()
  └── Encode grid, position, status
      │
      ▼
SAREnvGUI.render()
  ├── Camera.get_crop() → RGB frame
  ├── InfoPanel.update()
  ├── ChatPanel.update()
  └── EdgeVignette.trigger(events)
```
