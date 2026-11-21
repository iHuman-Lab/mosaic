---
title: "MOSAIC: A Modular Search-and-Rescue Testbed for Human-AI Collaboration Research"
tags:
  - Python
  - human-AI collaboration
  - human-AI interaction
  - human-AI teaming
  - human-LLM collaboration
  - large language models
  - search and rescue
  - reinforcement learning
  - Gymnasium
  - MiniGrid
  - human factors
  - eye tracking
  - Lab Streaming Layer
  - human-subjects experimentation
authors:
  - name: Bennett Dogbey
    orcid: 0009-0004-0980-1397
    affiliation: 1
  - name: Elahe Oveisi
    orcid: 0000-0002-6313-3260
    affiliation: 1
  - name: Hemanth Manjunatha
    orcid: 0000-0003-2207-8960
    corresponding: true
    affiliation: 1
affiliations:
  - name: Department of Mechanical and Aerospace Engineering, Oklahoma State University, Stillwater, Oklahoma, USA
    index: 1

date: 7 November 2026
bibliography: paper.bib
---

# Summary
Research on human-AI collaboration typically requires controlled environments that are complex enough to necessitate meaningful decisions and repeatable enough to facilitate systematic experimentation [cite]. MOSAIC is an open-source, grid-based, search and rescue environment that was developed for this purpose. The MOSAIC simulation framework was built upon MiniGrid and Gymnasium [@chevalier2023minigrid;@towers2026gymnasium]. A participant acts as a first responder who has to find and rescue as many victims as possible. The mission has a time limit and participants may not always be able to find every victim. During the missions, a LLM agent offers recommendations based on the current state of the task.

MOSAIC can record structured information about the environment, participant actions, mission progress, and interactions with the LLM assistant. Researchers are free to vary the environment (layout, placement of victims and hazards, visual perspective, reward structure, AI client, prompts, and pacing) without needing to change the underlying task. The software separates the reusable search and rescue mechanics from study-specific configuration and provides an experiment layer for integrating participant measures, synchronizing sensing, and replaying experimental sessions. We thus hope to enable reproducible research into how people seek, interpret, trust, and act on LLM assistance amid workload, uncertainty, and time pressure through MOSAIC.

# Statement of need
Large language models (LLMs) have greatly accelerated the rise of AI-assisted decision making [cite]. People increasingly ask LLMs for advice on important decisions, often in time-pressured settings and under incomplete information [cite]. This raises two questions: how do people interact with an LLM when making decisions, and how should an LLM be designed so that its advice fits the way people make decisions? This is an alignment problem [cite], for an LLM that is accurate but ignored, overtrusted, or difficult to verify does not help the person it is supposed to support [cite].

There are a number of different aspects to the human-machine alignment: trust, reliance, situation awareness [@taylor1990situational], workload [@hart1988nasa], or how to adapt advice to a particular person. Physiological computing, which uses signals such as eye movements and EEG to infer a person's state [cite], offers one way to do the last of these: an LLM could adapt its advice to the person's trust, workload, or attention. All of these require a specific experimental setup: a realistic task, consistent between participants (for the ability to compare), and detailed information about the human actions, LLM output, their synchronicity, and human physiological reactivity to LLM advice. All of these factors cannot be measured by the human task score alone. In practice, each experiment would have to build these facilities from scratch. While there are individual tools (simulation, interfaces, LLM modules, sensors synchronization, replay) for each of the pieces, these are developed independently of each other, without sychronization capabilities, tailored to a specific experiment setup.

MOSAIC is a structured platform for this work. It is a Gymnasium-compatible search and rescue task with an operator interface, a replaceable LLM advisory channel, and logging and replay tied to the environment's observations. Reward scales, pacing, hazard density, and sensor settings are set at runtime, so researchers can change a study's conditions without changing the task.

# State of the field
Several open-source platforms exist to facilitate each aspect of human-AI interaction. Platforms for the learning aspect include MiniGrid [@chevalier2023minigrid], PettingZoo [@terry2021pettingzoo], Unity ML-Agents [@juliani2018unity], AI2-THOR [@kolve2017ai2], and Habitat [@savva2019habitat]. These enable agents to learn and be evaluated in grid, multi-agent and 3D embodied environments, but mostly do not include a human participant. Another set of tools exists to facilitate the human aspect.

On the human side, Overcooked-AI is a cooperative environment for human and AI agents[@carroll2019utility]. The Human-Agent Teaming Rapid Experimentation software (MATRX) simulates human-agent teams and has been utilized for urban search and rescue [@matrx2023;@schoonderwoerd2022design]. PsychoPy is used to design and run behavioral experiments [@peirce2019psychopy2], and the Lab Streaming Layer (LSL) [@kothe2025lab] synchronizes various data streams, such as eye tracking data and EEG data. Most of these tools do not offer a Gymnasium API for the AI agents to learn through [cite], and one must wire up the human side and the learning side by hand. We designed MOSAIC to address both sides.

MOSAIC builds upon MiniGrid, so that the same search and rescue task can be used as a Gymnasium learning environment for the agents or as a task for human participants. To support the human participants, we also develop a human interface, pluggable LLM assistance, behavioral logging, and LSL-based synchronization with eye-tracking, electroencephalography (EEG) recording and other possible physiological measures.

# Software design
MOSAIC has three design goals. The same search and rescue task must serve an agent playing through code, a human playing in real time, and a researcher running a controlled experiment. Study details such as rewards and victim placement must change without touching the task. And everything must run without a model provider or sensor hardware. To meet them, MOSAIC is split into an installable `mosaic` package, which holds the task, and an `experiment` layer, which holds one study's settings (\autoref{fig:architecture}). The components are:

| Component             | What it does                                                                                                  |
| --------------------- | ------------------------------------------------------------------------------------------------------------- |
| Environment           | Runs the mission. It is built on MiniGrid and keeps Gymnasium's `reset()` and `step()`.                       |
| Placers               | Decide where victims, lava, locked rooms, and keys go.                                                        |
| Rescue action         | Decides what happens when a victim is picked up, and the reward.                                              |
| Observation processor | Builds the game state: the map, the agent's position, mission progress, and victim health.                    |
| Camera                | Decides which part of the map the participant sees.                                                           |
| Participant interface | Shows the map, mission information, LLM advice, and screen-edge feedback, and turns key presses into actions. |
| LLM client            | Sends text to an LLM and returns text. An offline dummy client is the default.                                |
| `experiment` layer    | Adds one study's pacing, rewards, placers, LLM providers, eye tracking, LSL recording, and replay.            |

Each component can be replaced without changing the others, and each has a neutral default, so a study configures only what it needs. For example, the reusable rescue action rewards \(+1\), \(-1\), and \(-2\) for a rescued, wrong, and dead victim, and the study uses \(+10\), \(-10\), and \(-20\) without changing that code. The cost is more setup than a fixed environment. Two further choices concern the data.

What the participant sees on screen is kept separate from the state that is recorded. Every step also reports what happened, such as `victim_rescued` or `wrong_victim`, instead of leaving it to be guessed from the reward. This lets rewards change between studies without breaking the interface or the analysis. The LLM works from the same recorded state.

The LLM client knows nothing about the task or the provider, so any provider can be used. Advice is requested in a background thread, so the game keeps running while the LLM answers. The cost is that only text goes in and out, so provider features beyond text are not available.

The experiment layer records one JSON game state per frame through LSL, using the external `ixp` package, next to a Tobii eye-tracking stream. LSL puts all streams on a shared clock, so they can be aligned afterwards. The replay utility redraws the game from these states instead of running it again, at the cost of storing a full state for every frame. Over 100 `pytest` tests run without a network, a provider, or sensor hardware.

![Overview of MOSAIC's layered architecture. The installable `mosaic`
package defines the SAR task, participant interface, structured state and
event boundary, and provider-independent LLM interface. The study-specific
`experiment` layer supplies calibrated implementations, provider adapters,
instrumentation, and state-based replay.
\label{fig:architecture}](figures/mosaic-architecture-overview.svg){width="100%"}

# Research impact statement
MOSAIC has been used for human-LLM teaming studies in search and rescue. In a study accepted to the IEEE International Conference on Systems, Man, and Cybernetics (SMC) 2026 and available as a preprint [@oveisi2026llm], participants completed missions under two LLM-guided conditions and a no-LLM baseline while their eye movements were recorded. LLM guidance improved task efficiency and it shifted visual attention toward the chat interface. Novices relied passively on the LLM, while experts kept scanning the environment to check its advice. A second on-going study [@oveisillm] uses MOSAIC's synchronized physiological monitoring to examine how LLM-mediated communication affects physiological responses.

We plan to use MOSAIC to study adaptive LLM assistance, trust and situation awareness in LLM-assisted decisions (including EEG signatures), and human-LLM teams as coupled dynamical systems, where reliance and advice can reinforce one another and produce instabilities. The package is documented, tested in continuous integration, and has example notebooks.

# AI usage disclosure

We used generative AI (Claude Code, Anthropic) to help write the source code, mostly routine implementation, and to improve the wording of the documentation and this paper. We designed the software, including its architecture, abstractions, and public interfaces, and the ideas, claims, and sentences are ours. We reviewed all AI-generated code and take full responsibility for the software and this paper. Correctness is checked with the `pytest` suite, which runs in continuous integration, and with manual testing.

# Acknowledgements

This work received no external funding.

# References

<!--
Leave empty. JOSS generates this section from paper.bib.
-->
