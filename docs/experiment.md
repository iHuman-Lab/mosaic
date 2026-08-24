# Experiment Framework

**File:** `src/experiment/experiment_main.py`

## Task sequence

```text
Tobii Calibration
       │
       ▼
Visual Search Task  (baseline cognitive measure)
       │
       ▼
Multi-Object Tracking Task  (baseline cognitive measure)
       │
       ▼
Tutorial  (single-room practice)
       │
       ▼
Main SAR Game  (primary task, with LLM assistant)
       │
       ▼
SART Survey  (situation awareness)
       │
       ▼
NASA-TLX Survey  (perceived workload)
```

## Data collection

Every frame during the main game is recorded as a `SARGameTrial` object streamed via LSL (Lab Streaming Layer):

- Full observation dict (grid state, agent position, camera bounds)
- Action taken and resulting reward
- LLM prompt + response (if any) with timing
- Eye gaze coordinates (via Tobii LSL bridge)
- Experiment metadata (participant ID, trial number, task order)

This enables full post-hoc replay and analysis without re-running the experiment.

## Eye-tracking integration

**Package:** `ixp` (internal experiment framework) + `tobii-research`

- `TobiiEyeTracker` is initialized and calibrated before the experiment begins
- Optional recalibration between tasks
- Gaze data is streamed in sync with game state via LSL
- Area of Interest (AOI) analysis can be run post-hoc using recorded gaze + frame data

The game can run without a Tobii device; eye-tracking is automatically skipped if the hardware is not detected.

## Replay system

**File:** `src/experiment/replay.py`

Reads a JSON-encoded LSL recording and replays the session in a Pygame window.

```bash
PYTHONPATH=src python -m experiment.replay --file <recording.json>
```

**Playback controls:**

| Key | Action |
| --- | --- |
| `Space` | Pause / resume |
| `←` | Step back one frame |
| `→` | Step forward one frame |
| `ESC` | Quit |

The replay reconstructs each frame from the encoded `grid` field in the recording — no environment re-execution required.

## This study's tuning

`experiment/` wires study-specific behavior on top of `mosaic`'s neutral defaults:

- `experiment.placers.LavaRiskVictimPlacer` — this study's health/decay tuning for victim placement.
- `experiment.pacing.TunedPickupVictimEnv` — this study's pacing formula, passed as `env_cls` to `build_sar_env()`.
- `experiment.game.SARGame` — the task that streams `SARGameTrial` records via LSL, wiring the tuned placers and pacing into `mosaic`'s injection points.
