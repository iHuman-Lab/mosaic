# Contributing to MOSAIC

Thanks for your interest in contributing to MOSAIC! This document covers how to report issues,
propose changes, and get help.

## Reporting bugs

Open a [GitHub issue](https://github.com/iHuman-Lab/mosaic/issues) with:

- What you expected to happen vs. what actually happened
- Steps to reproduce (a minimal script or command is ideal)
- Your Python version and OS
- The full error traceback, if any

## Suggesting features

Open an issue describing the use case first, especially for anything touching `experiment/`
(this lab's study-specific code) vs. `mosaic/` (the generic, installable package) — see
[Architecture](https://ihuman-lab.github.io/mosaic/architecture/) for that split. Features
belong in `mosaic/` only if they're generically useful outside this lab's specific study.

## Getting support

For questions about using MOSAIC, open a GitHub issue with the `question` label. For anything
related to this lab's ongoing research use of MOSAIC, reach out via the
[iHuman Lab website](https://github.com/iHuman-Lab).

## Making a pull request

1. Fork the repo and create a branch off `main`.
2. Install the package in editable mode: `pip install -e ".[experiment,docs]"`.
3. Make your change. Keep `mosaic/` generic and neutral — study-specific tuning belongs in
   `experiment/`.
4. Add or update tests in `tests/` for any behavior change. Run the suite:
   ```bash
   PYTHONPATH=src SDL_VIDEODRIVER=dummy pytest tests/
   ```
5. If you touched public behavior, update the relevant page under `docs/` (the docs site is
   built with mkdocs-material; `mkdocs serve` to preview locally).
6. Open a pull request against `main` describing the change and why it's needed.

## Code style

- No unnecessary abstractions — prefer the simplest change that solves the problem.
- Follow the existing dependency-injection pattern (`Placer`, `LLMClient`, GUI components are
  all constructor-injected) rather than adding new subclassing or global state.
- Tests use fakes passed through constructors rather than monkeypatching.
