# Getting Started

## Prerequisites

- Python 3.9+
- A display server (X11 or native on macOS/Windows)
- Optional: Tobii eye tracker + `tobii-research` SDK for eye-tracking experiments

## Install

```bash
git clone https://github.com/iHuman-Lab/mosaic.git
cd mosaic

# Install dependencies
pip install -r requirements.txt

# Or install as a package
pip install -e .
```

## API keys (for LLM features)

Set environment variables for your chosen LLM provider:

```bash
export OPENAI_API_KEY="sk-..."
export GOOGLE_API_KEY="..."
```

## Running the game

`experiment/` is excluded from the installable package, so its entry points are run as modules
(`python -m`) from the repo root, with `src/` on `PYTHONPATH`:

### Interactive mode (play the game)

```bash
PYTHONPATH=src python -m experiment.main
```

Launches the full Pygame GUI with a 2x2 room grid, real/fake victims, lava, locked rooms, and an optional LLM assistant.

### Experiment mode (research)

```bash
PYTHONPATH=src python -m experiment.experiment_main
```

Runs the full experiment protocol: visual search task, multi-object tracking, tutorial, main SAR game, and cognitive surveys. Requires Tobii hardware and the `ixp` package.

### Replay a session

```bash
PYTHONPATH=src python -m experiment.replay --file <path_to_lsl_recording.json>
```

## A minimal script

```python
from mosaic.sar.env import build_sar_env
from mosaic.sar.placers import VictimPlacer
from mosaic.gui.main import SAREnvGUI

env = build_sar_env(
    screen_size=800,
    victim_placer=VictimPlacer(num_real_victims=3),
)
gui = SAREnvGUI(env, config={"fullscreen": False})
gui.run()
```

See [Architecture](architecture.md) for how `mosaic/` (generic, installable) and `experiment/`
(this lab's calibrated study code) are separated.
