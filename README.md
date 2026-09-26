<div align="center">

# 🧩 MOSAIC

### *A Modular System for Adaptive Human–AI Collaboration*

A grid-based Search and Rescue simulation platform for studying human–AI teaming, where every second counts!

Built on top of [MiniGrid](https://github.com/Farama-Foundation/Minigrid) 🎮

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![MiniGrid](https://img.shields.io/badge/built%20on-MiniGrid-green.svg)](https://github.com/Farama-Foundation/Minigrid)

</div>

---

## 🎯 What is MOSAIC?

MOSAIC is named after the art of constructing a coherent picture from individual tiles. The platform combines modular components — simulation environments, AI agents, human interfaces, multimodal sensing, and analytics — to support reproducible Human–AI collaboration research. The name also reflects the grid-based structure of the underlying environments, where complex collaborative behaviors emerge from interactions within a tiled world.

Its first testbed is a search-and-rescue scenario: a building on fire, victims trapped, some rooms locked, lava blocking your path. Your mission — human, AI, or both together — is to save everyone before time runs out. But MOSAIC isn't tied to search and rescue; the same modular pieces can support other collaborative domains as the platform grows.

## ✨ Features

| Feature                    | Description                                               |
| --------------------------- | --------------------------------------------------------- |
| 🏢 **Multi-Room Layouts**   | Navigate through configurable grid-based buildings        |
| 🎯 **Real vs Fake Victims** | Distinguish cross-shaped victims ✚ from T-shaped decoys ⊤ |
| 🔥 **Lava Hazards**         | One wrong step and it's game over!                        |
| 🔐 **Locked Rooms**         | Find keys to unlock doors and reach trapped victims       |
| 🎮 **Interactive GUI**      | Beautiful Pygame interface with real-time info             |
| 🤖 **RL-Ready**             | Gymnasium compatible for training your rescue agents      |
| 📡 **Lab Streaming Layer**  | Sync with eye trackers, EEG, and other physiological sensors |
| 🧠 **LLM Integration**      | Prompt-driven agent reasoning via a pluggable LLM client   |

## 🚀 Quick Start

### Installation

```bash
pip install ihuman-mosaic
```

The PyPI name is `ihuman-mosaic`; the Python package is `mosaic` (`import mosaic`). Python 3.10 or newer.

Check it works, then play:

```bash
mosaic demo    # headless random-agent episode, prints a result
mosaic play    # play a search-and-rescue mission in the GUI (no API key needed)
```

To work on MOSAIC itself, clone the repo and run `pip install -e ".[experiment]"`.

### Run Your First Rescue Mission

```python
from mosaic.gui.main import SAREnvGUI
from mosaic.llm.client import DummyLLMClient
from mosaic.sar.env import build_sar_env
from mosaic.sar.placers import LavaPlacer, LockedRoomPlacer, VictimPlacer

# Set up the mission: placers decide what goes where
env = build_sar_env(
    screen_size=800,
    num_rows=3,
    num_cols=3,
    room_size=8,
    victim_placer=VictimPlacer(num_real_victims=3),        # 3 victims to save
    lava_placer=LavaPlacer(lava_per_room=2),               # danger mode: ON 🔥
    locked_room_placer=LockedRoomPlacer(locked_room_prob=0.5),  # 50% rooms locked 🔐
)

# Launch the mission!
env.reset(seed=0)
SAREnvGUI(env, config={"fullscreen": False}, llm_client=DummyLLMClient()).run()
```

No display? The same `env` is a regular Gymnasium environment:

```python
obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
```

## ⚙️ Configuration

| Parameter                                 | Where                | Description                    | Default |
| ----------------------------------------- | -------------------- | ------------------------------ | ------- |
| `num_rows`, `num_cols`                    | `build_sar_env`      | Building size (rooms)          | 5, 5    |
| `room_size`                               | `build_sar_env`      | Tiles per room                 | 14      |
| `num_real_victims`                        | `VictimPlacer`       | Victims to rescue              | 1       |
| `lava_per_room`                           | `LavaPlacer`         | Lava tiles per room 🔥          | 0       |
| `locked_room_prob`                        | `LockedRoomPlacer`   | Chance a room is locked 🔐      | 0.35    |
| `llm_client`                              | `SAREnvGUI`          | Advisor AI (`DummyLLMClient` = none) | -   |

## 🎮 Controls

| Key     | Action                       |
| ------- | ----------------------------- |
| ⬆️       | Move forward                 |
| ⬅️ ➡️     | Rotate left/right            |
| `Space` | Toggle/interact (open doors) |
| `Tab`   | Pickup/rescue victim         |
| `Shift` | Drop object                  |
| `F11`   | Toggle fullscreen            |
| `ESC`   | Quit mission                 |

## 📁 Project Structure

```
src/
├── mosaic/              # 📦 Installable package — generic, reusable defaults
│   ├── core/            # 🏗️ Base environment & level generation
│   ├── gui/             # 🖼️ Pygame GUI components
│   ├── sar/             # 🚨 SAR task mechanics (placement, actions, observations)
│   └── llm/             # 🧠 LLM-driven agent reasoning
└── experiment/           # 🧪 This lab's study: tuned parameters, orchestration, sensors
    ├── sensors/           #    LSL-synced sensors (eye tracker, EEG, etc.)
    ├── placers.py         #    Study-calibrated victim health/decay tuning
    ├── pacing.py          #    Study-specific max-steps formula
    ├── game.py            #    Trial/task orchestration
    ├── main.py            # 🚀 Dev/demo entry point (python -m experiment.main)
    └── experiment_main.py # 🧪 Full experiment entry point (python -m experiment.experiment_main)
```

`mosaic/` ships generic defaults (neutral rewards, neutral victim health, MiniGrid's own
max-steps fallback) so `pip install ihuman-mosaic` alone runs a working SAR episode. `experiment/`
supplies this lab's exact calibrated values on top, via the constructor injection points
`mosaic/sar/` exposes — see `REFERENCE.md` for the full list.

## 🤝 Contributing

Found a bug? Have an idea? PRs are welcome!

## 📄 License

MIT License - Feel free to use this for your research!

---

<div align="center">

**Built with ❤️ for the Human–AI collaboration research community**

*MOSAIC: many pieces, one picture.*

</div>
