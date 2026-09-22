---
title: "MOSAIC: A Modular Search-and-Rescue Testbed for Human-AI Collaboration Research"
tags:
  - Python
  - human-AI collaboration
  - human-AI teaming
  - search and rescue
  - reinforcement learning
  - Gymnasium
  - MiniGrid
  - human-subjects experimentation
authors:
  - name: Bennett Dogbey
    orcid: 0000-0000-0000-0000 # TODO: real ORCID
    corresponding: true # TODO: confirm corresponding author
    affiliation: 1
  - name: Elahe Oveisi
    orcid: 0000-0000-0000-0000 # TODO: real ORCID
    affiliation: 1
  - name: Hemanth Manjunatha
    orcid: 0000-0000-0000-0000 # TODO: real ORCID
    affiliation: 1
affiliations:
  - name: TODO institution, department, city, country
    index: 1
  
  - name: TODO institution, department, city, country
    index: 1

date: 1 September 2026 # TODO: update to the actual submission date
bibliography: paper.bib
---

# Summary

Research on human–AI collaboration requires controlled environments that are complex enough to produce meaningful decisions yet repeatable enough to support systematic experimentation. MOSAIC is an open-source, grid-based search-and-rescue platform designed for this purpose. Built on MiniGrid and Gymnasium [@chevalier2023minigrid; @towers2026gymnasium], it places a human participant in a time-constrained, multi-room mission involving real victims, decoys, locked doors, keys, and environmental hazards. During the mission, an AI assistant can provide natural-language recommendations based on the current task state, while the participant retains responsibility for deciding whether and how to follow that advice.

MOSAIC records structured information about the environment, participant actions, mission progress, and interactions with the AI assistant. Researchers can vary the environment layout, victim and hazard placement, visual perspective, reward structure, AI client, prompting method, and experimental pacing without rewriting the underlying task. The software separates reusable search-and-rescue mechanics from study-specific configuration and provides an experiment layer for integrating participant measures, synchronized sensing, and session replay. MOSAIC therefore supports reproducible investigations of how people seek, interpret, trust, and act on AI assistance under workload, uncertainty, and time pressure.

# Statement of need

Research on human-AI interaction requires tasks that simulate realistic challenges such as time pressure, incomplete information, and tangible consequences for errors, while maintaining enough control for consistent replication across participants. Simply measuring the final task score is insufficient; researchers need detailed data on the sequence of participant actions, the timing and content of AI advice, and aligned physiological signals.

Currently, assembling this experimental setup is a labor-intensive process for each research. It involves integrating multiple components such as simulators, participant interfaces, AI modules, sensor synchronization, logging, and replay tools that were often developed independently without a shared timing framework. This leads to substantial, often redundant framework that is tightly coupled to specific study designs, limiting reusability.

MOSAIC provides an integrated, Gymnasium-compatible framework for human-AI teaming, human factors, and reinforcement learning research. It features a configurable search-and-rescue task, operator interface, and modular advisory channel, with consistent logging and replay tied to environment observations. Task parameters such as reward scales, pacing, hazard density, and sensor settings are injected at runtime to maintain neutrality and adaptability.

By separating core task mechanics from experiment layers and supporting replaceable components, MOSAIC enables researchers to manipulate conditions while preserving a consistent task foundation, facilitating rigorous study of AI-assisted decision-making under realistic constraints.

# State of the field
Simulation testbeds provide controlled and flexible environments for studying human–AI interaction by allowing researchers to manipulate task conditions, reproduce experimental scenarios, and adapt task parameters to different research questions. Several open-source platforms support different parts of this research. MiniGrid [@chevalier2023minigrid] provides lightweight and configurable 2D grid-world environments for reinforcement learning and goal-oriented tasks. PettingZoo [@terry2021pettingzoo] offers a standardized framework for multi-agent reinforcement learning, while Unity ML-Agents [@juliani2018unity] supports the development and training of intelligent agents in configurable 3D environments. More complex embodied-AI platforms, such as AI2-THOR [@kolve2017ai2] and Habitat [@savva2019habitat], provide interactive 3D environments for studying navigation, perception, object interaction, and agent decision-making. Although these platforms offer flexible foundations for AI research, their main focus is generally on agent training, multi-agent learning, or embodied-agent performance.
 
Other platforms support specific aspects of human–AI interaction and human-subject experimentation. Overcooked-AI provides a cooperative environment in which humans and AI agents work together toward shared objectives, making it useful for studying human–AI coordination [@carroll2019utility]. More general human–agent teaming frameworks, such as MATRX, support the rapid development of collaborative tasks and have also been used to implement urban search-and-rescue scenarios [@matrx_2023], [@schoonderwoerd2022design]. Other tools address different parts of the experimental workflow. PsychoPy [@peirce2019psychopy2], for example, is widely used to design controlled behavioral and cognitive experiments, while Lab Streaming Layer (LSL) [@kothe2025lab] supports the synchronization of multimodal data streams, including eye-tracking and EEG. Although these platforms provide important foundations, their capabilities are generally distributed across separate tools, with different systems supporting task simulation, human–AI coordination, experimental control, or physiological data synchronization. MOSAIC brings these elements together within a single configurable framework by combining a search-and-rescue task environment, human interaction, AI and LLM assistance, experimental manipulation, behavioral and performance logging, and synchronized physiological sensing. This integrated structure allows researchers to relate specific task events, human actions, and AI interactions to behavioral and physiological responses, supporting the study of decision-making, situation awareness, workload, visual attention, trust, and reliance on AI within the same experimental environment.
 
 
MOSAIC builds on MiniGrid as its lightweight and configurable simulation foundation while extending it to support a broader human–AI experimental workflow. Human–AI studies often need more than a simple task environment alone. Researchers may need an interactive human interface, configurable experimental manipulations, AI or LLM-based assistance, behavioral and performance logging, and synchronized physiological measurements. These capabilities are important for investigating processes such as decision-making, situation awareness, mental workload, visual attention, trust, and reliance on AI. Although many of these functions are available through existing tools, integrating them into a single experiment and synchronizing physiological signals with task events, human actions, and AI interactions often requires substantial study-specific implementation.
 
MOSAIC addresses this gap by combining a configurable search-and-rescue grid environment, human interaction, integrable AI and LLM assistance, task and behavioral logging, and LSL-based synchronization with eye-tracking, EEG, and other physiological measurements within a common framework. This integration let the researchers to evaluate cognitive and behavioral processes together with task performance and physiological responses while maintaining systematic control over experimental conditions.


# Software design
MOSAIC was designed to support three uses of the same Search and Rescue task: programmatic interaction by an artificial agent, real-time interaction by a human participant, and integration into controlled experimental protocols. Combining these concerns in a single application would make study-specific changes—such as reward calibration, victim placement, or AI-provider selection—difficult to separate from the underlying task. MOSAIC therefore defines a stable environment core and composes human-facing and study-specific behavior around it.

The core builds on MiniGrid and retains Gymnasium’s `reset()` and `step()` interaction model. The environment contains the mission lifecycle and action dispatch, while variable behavior is delegated to configurable components. Placement strategies determine the distribution of victims, hazards, and locked rooms; an action component implements rescue outcomes; an observation processor constructs research-relevant state; and a camera strategy determines the participant’s visible region. This requires more explicit configuration than a fixed environment, but allows layout, visibility, rewards, and task difficulty to be manipulated independently while preserving the task’s action semantics.

MOSAIC distinguishes between visual presentation and structured research state. A camera strategy produces the RGB frame displayed in the game viewport. Separately, the observation processor produces an encoded grid together with agent pose, mission progress, victim health, and camera bounds. This structured representation is used by the information panel, AI-advisory pipeline, trial logger, and replay utility. Outcomes such as `victim_rescued`, `wrong_victim`, `victim_died`, and `mission_complete` are communicated as semantic events through the step information rather than inferred from reward magnitude. This permits reward values to change between studies without requiring the interface or data-processing tools to reinterpret their meaning.

A second design decision separates reusable software from experimental calibration. The installable `mosaic` package contains the environment, neutral placement and reward defaults, interchangeable cameras, participant interface, and AI-advisory contract. The repository’s `experiment` layer is excluded from the distributed package and composes the core with the present study’s pacing, victim-health dynamics, hazard placement, trial sequence, sensing, and provider configuration. For example, the reusable rescue action assigns rewards of \(+1\), \(-1\), and \(-2\) to its three pickup outcomes, whereas the study configures \(+10\), \(-10\), and \(-20\) without modifying the reusable action logic. The additional composition required at study startup makes these assumptions explicit and prevents one protocol’s calibration from becoming the default for subsequent research.

MOSAIC favors composition and constructor injection over a deep hierarchy of experiment-specific subclasses. Placers, rescue behavior, observation processing, cameras, AI clients, and GUI components can be replaced independently, while neutral implementations provide a working default configuration. Inheritance is retained when behavior must occur at a precise point in the environment lifecycle. The study-specific environment, for example, applies visibility-dependent victim-health depletion after executing an action but before constructing the returned observation. Restricting inheritance to such ordering-sensitive behavior keeps most experimental variation localized and independently testable.

The participant interface is an adapter over the same action and state interfaces used by programmatic agents. Keyboard input is translated into environment actions, while the GUI combines the RGB frame with mission information, advisory messages, and event-driven visual feedback. AI advice is accessed through a narrow `LLMClient` contract that accepts text and returns text. Prompt construction and response processing are separate configurable operations, and provider-specific adapters remain in the experiment layer. The reusable package supplies an offline dummy client, allowing the environment and interface to operate without credentials or network access. Advice requests run in a background thread over a copied observation, preserving a consistent state snapshot while the rendering loop continues.

The experiment layer records structured game-state snapshots through Lab Streaming Layer alongside the eye-tracking stream. The replay utility reconstructs the game from the recorded grid states rather than re-executing the environment. Recording complete state requires more storage than recording actions alone, but avoids relying on exact regeneration of human choices or external model responses. The same component boundaries support headless tests using injected clients, components, and clocks, allowing task mechanics, advisory integration, and visual feedback to be evaluated without a model provider or specialized sensing hardware.

![Overview of MOSAIC's layered architecture. The installable `mosaic`
package defines the SAR task, participant interface, structured state and
event boundary, and provider-independent AI interface. The study-specific
`experiment` layer supplies calibrated implementations, provider adapters,
instrumentation, and state-based replay.
\label{fig:architecture}](figures/mosaic-architecture-overview.svg){width="100%"}

<!--
OUTLINE - 300-400 words. Do not draft yet.

Organize the discussion around design decisions, not a module tour:

   1. Reusable Gymnasium/MiniGrid environment core.
   2. SAR mechanics and observations.
   3. Separation between reusable `mosaic` code and study-specific `experiment`
      code.
   4. Constructor-injected study components and neutral defaults.
   5. Human interaction through the GUI.
   6. Provider-independent AI-advisory interface.
   7. Experimental logging, sensing, and replay.
   8. Composition versus inheritance.
   9. Reusability versus study-specific calibration.
  1.  Reproducibility and testability.

Grounding pointers for the drafter - verify each against the code before making
any claim about it:
  - src/mosaic/sar/env.py, src/mosaic/sar/observations.py, src/mosaic/sar/actions.py
  - src/mosaic/core/level.py, src/mosaic/core/camera.py, src/mosaic/core/placers.py
  - src/mosaic/gui/main.py (SAREnvGUI), src/mosaic/gui/chat.py,
    src/mosaic/gui/feedback.py
  - src/mosaic/llm/client.py, src/mosaic/llm/parser.py
  - src/experiment/experiment.py, src/experiment/replay.py
  - src/experiment/sensors/eye_tracker/
  - tests/
-->

<!--
RESERVED - one optional architecture figure. NOT CREATED YET.

Intended content:

    Human / AI
        |
    GUI and advisory interface
        |
    MOSAIC SAR environment
        |
    Observations, events and measurements
        |
    Logging, sensing and replay

When the figure exists, place it in paper/figures/ and reference it as:
    ![Caption.\label{fig:architecture}](figures/architecture.png)
-->

# Research impact statement

<!--
OUTLINE - 150-250 words. Do not draft yet.

AWAITING AUTHOR CONFIRMATION. Request verified evidence only:
  - Completed or ongoing studies that used MOSAIC.
  - Pilot-study use and approved participant information.
  - Data streams actually collected and the research questions they enabled.
  - Publications, preprints, posters, presentations, theses, or datasets.
  - Use by collaborators or other research groups.
  - Reproducible demonstrations or benchmarks.

CONSTRAINT: projected or planned future applications are NOT evidence of impact
and must not be presented as such. Do not invent studies, participant counts,
or results.
-->

# AI usage disclosure

<!--
OUTLINE - 60-100 words. Do not draft yet.

AWAITING AUTHOR CONFIRMATION of:
  - Which generative-AI tools were used in the software, the documentation, or
    the manuscript.
  - What those tools contributed.
  - What the human authors reviewed and decided.
  - How tests, code review, and manual verification checked correctness.

Do not write the final disclosure until the authors confirm these facts.
-->

# Acknowledgements

<!--
OUTLINE - 40-80 words. Do not draft yet.

AWAITING AUTHOR CONFIRMATION of:
  - Funding agencies and grant numbers.
  - Institutional and laboratory support.
  - Non-author contributors.
  - Equipment support.

Do not invent acknowledgements, funders, or grant numbers.
-->

# References

<!--
Leave empty. JOSS generates this section from paper.bib.
-->
