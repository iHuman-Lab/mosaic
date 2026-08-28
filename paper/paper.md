---
title: 'MOSAIC: A Modular System for Adaptive Human-AI Collaboration'
tags:
  - Python
  - human-AI teaming
  - human-computer interaction
  - reinforcement learning
  - large language models
  - eye tracking
authors:
  - name: FIXME Author Name
    orcid: 0000-0000-0000-0000
    affiliation: 1
    corresponding: true
affiliations:
  - name: FIXME Institution, Department
    index: 1
date: FIXME DD Month YYYY
bibliography: paper.bib
---

<!--
TODO before submission:
- Replace every FIXME above with real author name(s), ORCID iDs, and affiliation(s).
- List every significant contributor as an author (JOSS requires this to reflect actual
  authorship, not just repo ownership).
- Set the submission date.
-->

# Summary

MOSAIC is a modular, grid-based Search and Rescue (SAR) simulation platform for studying
human-AI collaboration. Built on top of MiniGrid [@minigrid] and Gymnasium [@gymnasium], it
places a human or AI agent (or both, cooperating) in a multi-room building on fire, where real
victims (rendered as a cross) must be distinguished from decoy victims (rendered as a T-shape)
and rescued before a time limit expires, while navigating locked doors and lava hazards. The
platform separates a generic, installable core (`mosaic`) — environment dynamics, victim and
hazard placement, camera strategies, and GUI components — from study-specific orchestration
(`experiment`), which supplies calibrated pacing, tuned placement policies, and integration with
physiological sensing. Every major component (victim/hazard placers, the LLM client, and GUI
panels) is injected via constructor parameters rather than subclassed, so a new study can swap
in custom behavior without forking the core package.

# Statement of need

Studying how humans and AI systems collaborate under time pressure and uncertainty requires an
environment that is simultaneously (1) a valid reinforcement-learning testbed, (2) instrumented
enough for cognitive/behavioral research, and (3) flexible enough to support an LLM-based
assistant without hard-coding a single provider. Existing grid-world environments such as
MiniGrid provide the RL substrate but no built-in support for eye-tracking synchronization,
LLM-driven advisory agents, or the mixed-initiative "rescue" task structure needed for human-AI
teaming studies; general-purpose HCI experiment frameworks, conversely, do not provide a
Gymnasium-compatible environment. MOSAIC addresses this gap by combining a Gymnasium-compatible
SAR environment with Lab Streaming Layer (LSL) integration for eye-tracking, a
provider-agnostic LLM client contract for advisory agents, and standard human-factors measures
(NASA-TLX [@hart1988nasa]; SART [@taylor1990situational]) already wired into its experiment
framework. Researchers in human-AI teaming, situation awareness, and mixed-initiative RL can use
MOSAIC's neutral core directly, or fork the pattern already used by `experiment/` to calibrate
their own study without modifying the installable package.

# Functionality

- A configurable multi-room grid environment (`mosaic.sar.env.PickupVictimEnv`) with pluggable
  victim, lava, and locked-door placement policies.
- Five interchangeable camera strategies (full view, agent-centered, edge-follow, field-of-view,
  and line-of-sight cone), switchable at runtime.
- A Pygame GUI with real-time mission status, an object table with reachability analysis, and an
  LLM chat panel.
- A provider-agnostic `LLMClient` contract, decoupled from any specific vendor, with a no-op
  `DummyLLMClient` for testing.
- An experiment framework wiring the environment to Tobii eye-tracking and LSL data streaming,
  plus a session replay tool.

# Acknowledgements

We acknowledge contributions from the iHuman Lab research group.

# References
