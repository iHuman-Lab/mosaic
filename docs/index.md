# MOSAIC

*A Modular System for Adaptive Human–AI Collaboration*

A grid-based Search and Rescue simulation platform for studying human–AI teaming, built on top of [MiniGrid](https://github.com/Farama-Foundation/Minigrid).

## What is MOSAIC?

MOSAIC is named after the art of constructing a coherent picture from individual tiles. The platform combines modular components — simulation environments, AI agents, human interfaces, multimodal sensing, and analytics — to support reproducible Human–AI collaboration research. The name also reflects the grid-based structure of the underlying environments, where complex collaborative behaviors emerge from interactions within a tiled world.

Its first testbed is a search-and-rescue scenario: a building on fire, victims trapped, some rooms locked, lava blocking your path. Your mission — human, AI, or both together — is to save everyone before time runs out. MOSAIC isn't tied to search and rescue; the same modular pieces can support other collaborative domains as the platform grows.

## Features

| Feature | Description |
| --- | --- |
| **Multi-Room Layouts** | Navigate through configurable grid-based buildings |
| **Real vs Fake Victims** | Distinguish cross-shaped victims (✚) from T-shaped decoys (⊤) |
| **Lava Hazards** | One wrong step and it's game over |
| **Locked Rooms** | Find keys to unlock doors and reach trapped victims |
| **Interactive GUI** | Pygame interface with real-time info panels |
| **RL-Ready** | Gymnasium compatible for training rescue agents |
| **Lab Streaming Layer** | Sync with eye trackers, EEG, and other physiological sensors |
| **LLM Integration** | Prompt-driven agent reasoning via a pluggable LLM client |

## Where to go next

- [Getting Started](getting-started.md) — install MOSAIC and run your first mission
- [Architecture](architecture.md) — how `mosaic/` (the installable package) and `experiment/` (this lab's study wiring) fit together
- [API Reference](api.md) — auto-generated reference from the source docstrings
