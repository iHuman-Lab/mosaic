# MOSAIC Reference Guide

A comprehensive reference for developers, researchers, and contributors working with the MOSAIC codebase.

---

## Table of Contents

1. [Overview](#overview)
2. [Project Structure](#project-structure)
3. [Setup & Installation](#setup--installation)
4. [Running the Game](#running-the-game)
5. [Game Concept & Mechanics](#game-concept--mechanics)
6. [Controls](#controls)
7. [Architecture](#architecture)
8. [Core Systems](#core-systems)
   - [Environment](#environment)
   - [Level Generation](#level-generation)
   - [Victim System](#victim-system)
   - [Hazards & Obstacles](#hazards--obstacles)
   - [Mission System](#mission-system)
   - [Camera System](#camera-system)
   - [Observation Encoding](#observation-encoding)
   - [Reward System](#reward-system)
9. [GUI Components](#gui-components)
10. [LLM Integration](#llm-integration)
11. [Experiment Framework](#experiment-framework)
12. [Eye-Tracking Integration](#eye-tracking-integration)
13. [Configuration](#configuration)
14. [Testing](#testing)
15. [Replay System](#replay-system)
16. [Key Classes & APIs](#key-classes--apis)

---

## Overview

**MOSAIC** is a modular platform for Human–AI collaboration research, built on top of [MiniGrid](https://github.com/Farama-Foundation/Minigrid). Its first testbed is a grid-based Search and Rescue (SAR) simulation environment, designed for two purposes:

1. **Interactive gameplay** — A human player navigates a multi-room building, rescues real victims, avoids decoys, and manages hazards under time pressure.
2. **Research platform** — A fully instrumented environment for studying human cognition, human-AI collaboration, and visual attention, with support for eye-tracking (Tobii), LLM-powered AI assistance, and cognitive surveys.

**Core scenario:** A building is on fire. Victims are trapped inside. You must navigate locked rooms, avoid lava, distinguish real victims from decoys, and rescue everyone before time runs out.

---

## Project Structure

`mosaic/` is the installable package (`pyproject.toml`'s `packages.find`) and ships generic,
neutral defaults — no study-specific tuning. `experiment/` (excluded from packaging) holds this
lab's calibrated parameters, orchestration, and entry points, wired on top of `mosaic/`'s
injection points.

```
mosaic/
├── src/
│   └── mosaic/
│       ├── core/
│       │   ├── camera.py        # Camera strategy implementations
│       │   ├── placers.py       # Placer ABC
│       │   └── level.py         # SARLevelGen base class
│       ├── sar/
│       │   ├── env.py           # PickupVictimEnv (main environment)
│       │   ├── objects.py       # Victim and FakeVictim classes
│       │   ├── actions.py       # RescueAction + RescueRewards (neutral defaults)
│       │   ├── observations.py  # GameObservation processor
│       │   ├── instructions.py  # PickupAllVictimsInstr mission
│       │   └── placers.py       # VictimPlacer (neutral health), LavaPlacer, LockedRoomPlacer
│       ├── gui/
│       │   ├── main.py          # SAREnvGUI controller
│       │   ├── user.py          # Keyboard input + LLM threading
│       │   ├── info.py          # InfoPanel (stats, object table)
│       │   ├── chat.py          # ChatPanel (LLM messages)
│       │   └── theme.json       # pygame_gui UI theme
│       ├── llm/
│       │   ├── client.py        # LLM API client (OpenAI, Google)
│       │   ├── parser.py        # LLM response cleaning
│       │   ├── process_prompts.py  # Prompt generation from game state
│       │   └── pathfinding.py   # Pathfinding queries for LLM context
│       └── tutorial_env.py      # Single-room tutorial environment
├── experiment/                  # excluded from the installable package
│   ├── main.py                  # Entry point: interactive GUI mode (python -m experiment.main)
│   ├── experiment_main.py       # Entry point: full research experiment
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
│   ├── test_environment.py
│   ├── test_lava.py
│   ├── test_reachability.py
│   └── playground.py
├── docs/
├── requirements.txt
└── pyproject.toml
```

---

## Setup & Installation

### Prerequisites

- Python 3.9+
- A display server (X11 or native on macOS/Windows)
- Optional: Tobii eye tracker + `tobii-research` SDK for eye-tracking experiments

### Install

```bash
# Clone the repository
git clone <repo-url>
cd mosaic

# Install dependencies
pip install -r requirements.txt

# Or install as a package
pip install -e .
```

### API Keys (for LLM features)

Set environment variables for your chosen LLM provider:

```bash
export OPENAI_API_KEY="sk-..."
export GOOGLE_API_KEY="..."
```

---

## Running the Game

`experiment/` is excluded from the installable package, so its entry points are run as modules
(`python -m`) from the repo root, with `src/` on `PYTHONPATH` (e.g. `PYTHONPATH=src`, or run from
inside `src/`):

### Interactive Mode (play the game)

```bash
PYTHONPATH=src python -m experiment.main
```

Launches the full Pygame GUI with a 2×2 room grid, real/fake victims, lava, locked rooms, and an optional LLM assistant.

### Experiment Mode (research)

```bash
PYTHONPATH=src python -m experiment.experiment_main
```

Runs the full experiment protocol: visual search task, multi-object tracking, tutorial, main SAR game, and cognitive surveys. Requires Tobii hardware and the `ixp` package.

### Replay a Session

```bash
PYTHONPATH=src python -m experiment.replay --file <path_to_lsl_recording.json>
```

---

## Game Concept & Mechanics

### Objective

Rescue **all real victims** before time runs out. Real victims are displayed as a **cross shape (✚)**. Fake victims (decoys) appear as a **T-shape (⊤)** — picking them up applies a score penalty.

### Building Layout

The building is a grid of rooms (default 2×2 or 3×3), each separated by walls with doorways. Some doors are **locked** and require a matching colored key found elsewhere in the building.

### Hazards

| Hazard | Effect |
|--------|--------|
| Lava | Instant mission failure if stepped on |
| Locked doors | Block passage until matching key is collected |
| Time limit | Mission fails if max steps exceeded |

### Victim Health

All victims (real and fake) have a health bar that depletes over time while they are visible to the agent. This creates urgency — visible victims that are not rescued will eventually be lost.

### Scoring

| Event | Reward |
|-------|--------|
| Rescue real victim | +1.0 |
| Pick up fake victim (decoy) | -0.5 |
| Complete mission (all rescued) | +1.0 bonus |

---

## Controls

| Key | Action |
|-----|--------|
| `↑` or `W` | Move forward |
| `←` | Rotate left |
| `→` | Rotate right |
| `Space` | Toggle / interact (open doors) |
| `Tab` or `Page Up` | Pick up / rescue victim |
| `Left Shift` or `Page Down` | Drop held object |
| `Alt` (left or right) | Ask LLM for a suggestion |
| `F11` | Toggle fullscreen |
| `Backspace` | Reset current mission |
| `ESC` | Quit |

---

## Architecture

The codebase uses several design patterns to keep systems decoupled and extensible:

- **Strategy Pattern** — Camera system (`camera.py`): swap strategies at runtime.
- **Factory Pattern** — `build_sar_env()` constructs configured environments.
- **Template Method** — `SARLevelGen` defines level generation hooks for subclasses.
- **Observer Pattern** — GUI event loop dispatches keyboard events to handlers.

### High-Level Data Flow

```
Keyboard Input
      │
      ▼
ManualControl.handle_key()
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
  └── ChatPanel.update()
```

---

## Core Systems

### Environment

**Class:** `PickupVictimEnv` — `src/mosaic/sar/env.py`

The main Gymnasium-compatible environment. Inherits from `SARLevelGen`.

**Key configuration parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `room_size` | 8 | Tiles per room (interior) |
| `num_rows` | 2 | Rows of rooms |
| `num_cols` | 2 | Columns of rooms |
| `add_lava` | True | Whether to place lava |
| `lava_per_room` | 2 | Lava tiles per room |
| `locked_room_prob` | 0.35–0.5 | Probability a door is locked |
| `agent_pov` | True | Enable first-person view camera |

**Key methods:**

| Method | Description |
|--------|-------------|
| `gen_mission()` | Generates a full level: rooms, doors, keys, lava, victims |
| `reset()` | Initializes an episode; computes `max_steps` |
| `step(action)` | Processes one action; updates victim health, checks end conditions |
| `get_mission_status()` | Returns `{status, saved_victims, remaining_victims}` |
| `switch_camera(strategy)` | Swaps camera strategy at runtime |

---

### Level Generation

**Class:** `SARLevelGen` — `src/mosaic/core/level.py`

Inherits from MiniGrid's `MiniGridEnv`. Provides:
- Multi-room grid construction
- Pluggable camera injection
- Base rendering with camera-aware crop

Subclassed by `PickupVictimEnv` and `TutorialEnv`.

---

### Victim System

**File:** `src/mosaic/sar/objects.py`

#### Real Victims (`Victim`)

- Rendered as a symmetric **cross (✚)**
- Directional variants: `victim_up`, `victim_down`, `victim_left`, `victim_right`
- Have a `health` property (0.0 → 1.0) with a rendered health bar
- Reward on rescue: **+1.0**

#### Fake Victims (`FakeVictim`)

- Rendered as an asymmetric **T-shape (⊤)**
- Left/right shift variants to vary appearance
- Same health depletion system as real victims
- Penalty on accidental rescue: **-0.5**

#### Placement (`VictimPlacer`) — `src/mosaic/sar/placers.py`

- Real victims placed preferentially in locked rooms (harder to reach)
- Fake victims distributed across all accessible rooms
- Reachability verified before placement (no inaccessible victims)

---

### Hazards & Obstacles

**Lava** (`LavaPlacer`) — `src/mosaic/sar/placers.py`

- Placed randomly within rooms, avoiding doorways and agent spawn
- Stepping on lava immediately terminates the episode with failure
- Density controlled by `lava_per_room` parameter

**Locked Doors & Keys**

- A subset of room connections are randomly locked (controlled by `locked_room_prob`)
- A matching colored key is placed in a reachable (unlocked) room
- The agent must carry the key to toggle the locked door open

---

### Mission System

**Class:** `PickupAllVictimsInstr` — `src/mosaic/sar/instructions.py`

Manages mission completion logic:
- Tracks rescued vs. remaining real victims
- Reports `success` when all real victims rescued
- Reports `failure` on lava contact or timeout

**Dynamic `max_steps` formula:**

```
max_steps = (room_exploration_steps + door_handling_steps
             + victim_rescue_steps) × safety_buffer

Where:
  room_exploration_steps = num_rooms × room_size² × 0.5
  door_handling_steps    = num_locked_doors × key_search_factor
  victim_rescue_steps    = num_victims × 5
  safety_buffer          = 1.0×
```

---

### Camera System

**File:** `src/mosaic/core/camera.py`

Four interchangeable camera strategies:

#### `FullviewCamera`
Shows the entire grid at once. Best for small maps or debugging.

#### `AgentCenteredCamera`
Centers the viewport on the agent. Entire room plus border tiles always visible.

#### `EdgeFollowCamera` *(default)*
Camera moves only when the agent approaches the edge of the viewport (dead-zone tracking). Viewport: 8×8 tiles, margin: 3 tiles.

#### `AgentFOVCamera`
Clips the view to the boundaries of the agent's current room. Natural wall framing.

#### `AgentConeCamera`
Room-bounded view with MiniGrid's line-of-sight cone. Tiles outside the agent's forward sightline are blacked out.

**Switching cameras at runtime:**

```python
env.switch_camera("edge_follow")   # or "full", "centered", "fov", "cone"
```

---

### Observation Encoding

**Class:** `GameObservation` — `src/mosaic/sar/observations.py`

Returns a dict on each step:

| Key | Type | Description |
|-----|------|-------------|
| `image` | ndarray (H×W×3) | RGB camera frame |
| `direction` | int | Agent facing: 0=E, 1=S, 2=W, 3=N |
| `agent_x`, `agent_y` | int | Agent grid position |
| `grid` | ndarray (H×W) | Full map encoded as integers (see below) |
| `carrying` | str or None | Color of held object |
| `step_count` | int | Steps elapsed this episode |
| `max_steps` | int | Episode step limit |
| `mission_status` | str | `"success"`, `"failure"`, or `"incomplete"` |
| `saved_victims` | int | Count of rescued real victims |
| `remaining_victims` | int | Count of unrescued real victims |
| `num_rows`, `num_cols` | int | Map dimensions in rooms |
| `room_size` | int | Room size in tiles |
| `cam_top_x`, `cam_top_y` | int | Camera viewport top-left tile |
| `cam_view_w`, `cam_view_h` | int | Camera viewport size in tiles |

**Grid integer encoding:**

```
0       = Empty floor
1       = Wall
4       = Lava
5       = Victim (real)
6       = FakeVictim (decoy)

Doors:  10 + (color_index × 3) + door_state
  door_state: 0=open, 1=closed, 2=locked
  color_index: red=0, green=1, blue=2, purple=3, yellow=4, grey=5
  Range: 10–27

Keys:   30 + color_index
  Range: 30–35
```

---

### Reward System

**Class:** `RescueAction` — `src/mosaic/sar/actions.py`

All rewards are sparse (returned only on specific events):

| Event | Reward |
|-------|--------|
| Rescue a real victim | +1.0 |
| Pick up a fake victim | -0.5 |
| All victims rescued (mission complete) | +1.0 bonus |
| Step on lava | episode terminated, no reward |
| Timeout | episode terminated, no reward |

---

## GUI Components

### Main Window (`SAREnvGUI`) — `src/mosaic/gui/main.py`

The top-level Pygame controller. Layout:

```
┌──────────────────────────┬──────────────┐
│                          │              │
│   Game Viewport          │  Info Panel  │
│   (800 × 800 px)         │  (400 px)    │
│                          │              │
│                          ├──────────────┤
│                          │  Chat Panel  │
│                          │  (LLM msgs)  │
└──────────────────────────┴──────────────┘
```

- 30 FPS rendering loop
- `F11` toggles fullscreen with dynamic UI scaling
- `pygame_gui` used for all UI widgets

---

### InfoPanel — `src/mosaic/gui/info.py`

Displays mission-critical information on the right panel:

- **Mission Status** header (victims saved / remaining)
- **Metrics:** cumulative reward, step count, elapsed time
- **Object Table:** lists all game objects in range with:
  - Type, color, location (room coordinates)
  - Visibility (in current camera view or not)
  - Reachability (BFS distance, or blocked)
  - Whether a tool (key) is required
- **Compass:** cardinal direction indicator for agent facing
- **Controls legend:** key bindings reminder

---

### ChatPanel — `src/mosaic/gui/chat.py`

Displays messages from the LLM assistant:
- New messages cause a **blinking highlight** effect
- Message colors: agent suggestions in cornflower blue, errors in crimson
- Polls for async LLM responses each frame

---

### User Input (`ManualControl` subclass) — `src/mosaic/gui/user.py`

Handles all keyboard input and dispatches:
- Movement and interaction actions to `env.step()`
- LLM queries on `Alt` key (runs in a background thread to avoid frame drops)
- Episode reset on `Backspace`

---

## LLM Integration

**Files:** `src/mosaic/llm/`

### Supported Providers

| Provider | Models |
|----------|--------|
| OpenAI | `gpt-4o-mini`, `gpt-5-mini-2025-08-07`, etc. |
| Google | `gemini-2.5-flash`, etc. |
| Dummy | No-op client for testing without API keys |

### How It Works

1. Player presses `Alt` (or the auto-nudge interval fires).
2. `process_prompts.py` builds a structured prompt from the current `GameObservation`:
   - Agent position and facing direction
   - List of visible and reachable objects with pathfinding distances
   - Current mission status
   - Prompt type: `sparse` (minimal) or `detailed` (full grid scan + strategy)
3. The prompt is sent to the configured LLM provider asynchronously.
4. `parser.py` cleans the response (removes coordinates, reformats object labels).
5. The cleaned suggestion appears in the ChatPanel.

### Configuration

```yaml
# configs/experiment.yaml
game:
  prompt_type: sparse          # "sparse" or "detailed"
  openai_model: gpt-5-mini-2025-08-07
  llm_nudge_interval: 50       # auto-prompt every N steps (0 = disabled)
```

---

## Experiment Framework

**File:** `src/experiment/experiment_main.py`

### Task Sequence

```
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

### Data Collection

Every frame during the main game is recorded as a `SARGameTrial` object streamed via LSL (Lab Streaming Layer):

- Full observation dict (grid state, agent position, camera bounds)
- Action taken and resulting reward
- LLM prompt + response (if any) with timing
- Eye gaze coordinates (via Tobii LSL bridge)
- Experiment metadata (participant ID, trial number, task order)

This enables full post-hoc replay and analysis without re-running the experiment.

---

## Eye-Tracking Integration

**Package:** `ixp` (internal experiment framework) + `tobii-research`

- `TobiiEyeTracker` is initialized and calibrated before the experiment begins
- Optional recalibration between tasks
- Gaze data is streamed in sync with game state via LSL
- Area of Interest (AOI) analysis can be run post-hoc using recorded gaze + frame data

The game can run without a Tobii device; eye-tracking is automatically skipped if the hardware is not detected.

---

## Configuration

### `configs/experiment.yaml` (full experiment)

```yaml
game:
  display: 0              # Display index (0 = primary)
  fullscreen: false
  max_time: 10            # Mission time limit in minutes
  prompt_type: sparse     # LLM prompt verbosity
  openai_model: gpt-5-mini-2025-08-07
  num_rows: 2             # Rooms tall
  num_cols: 2             # Rooms wide
  llm_nudge_interval: 50  # Steps between automatic LLM prompts

vs:                       # Visual Search task config
  ...

mot:                      # Multi-Object Tracking task config
  ...

surveys:                  # NASA-TLX and SART config
  ...
```

### `configs/config.yml` (base / dev)

Simpler config used when running `python -m experiment.main` directly. Override individual parameters here for quick iteration.

---

## Testing

```bash
# Run all tests
pytest tests/

# Individual test files
pytest tests/test_environment.py
pytest tests/test_lava.py
pytest tests/test_reachability.py
```

| Test File | What it validates |
|-----------|-------------------|
| `test_environment.py` | Environment initialization, step, reset |
| `test_lava.py` | Lava placement doesn't block all paths |
| `test_lava_50_percent.py` | High-density lava placement edge cases |
| `test_reachability.py` | All objects (victims, keys) are reachable from spawn |
| `test_strict_reachability.py` | Strict BFS reachability checks |
| `playground.py` | Manual development / debugging entrypoint |

---

## Replay System

**File:** `src/experiment/replay.py`

Reads a JSON-encoded LSL recording and replays the session in a Pygame window.

```bash
PYTHONPATH=src python -m experiment.replay --file <recording.json>
```

**Playback controls:**

| Key | Action |
|-----|--------|
| `Space` | Pause / resume |
| `←` | Step back one frame |
| `→` | Step forward one frame |
| `ESC` | Quit |

The replay reconstructs each frame from the encoded `grid` field in the recording — no environment re-execution required.

---

## Key Classes & APIs

### `PickupVictimEnv` (`src/mosaic/sar/env.py`)

```python
env = PickupVictimEnv(
    room_size=8,
    num_rows=2,
    num_cols=2,
    add_lava=True,
    render_mode="rgb_array"
)
obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step(action)
status = env.get_mission_status()
# status = {"status": "incomplete", "saved_victims": 1, "remaining_victims": 2}
```

### `SAREnvGUI` (`src/mosaic/gui/main.py`)

```python
gui = SAREnvGUI(env, config)
gui.run()  # Blocking Pygame loop
```

### `GameObservation` (`src/mosaic/sar/observations.py`)

```python
obs_processor = GameObservation(env)
obs = obs_processor.process()  # Returns observation dict
```

### `build_sar_env()` (factory)

```python
from mosaic.sar.env import build_sar_env
env = build_sar_env(config_dict)
```

### Camera switching

```python
env.switch_camera("edge_follow")   # EdgeFollowCamera (default)
env.switch_camera("full")          # FullviewCamera
env.switch_camera("centered")      # AgentCenteredCamera
env.switch_camera("fov")           # AgentFOVCamera
env.switch_camera("cone")          # AgentConeCamera
```

### LLM Client (`src/mosaic/llm/client.py`)

```python
from mosaic.llm.client import build_llm_client
client = build_llm_client(provider="openai", model="gpt-4o-mini")
response = client.query(prompt_string)
```

---

*Last updated: 2026-03-28*
