"""A shared leaderboard for the tutorial notebooks.

Every attendee's notebook writes one small file into a folder that all users on the server can write to, and a presenter's notebook reads them all.
One file per person means no locking and no clashes.

    submit_score("mosaic", nickname, score, steps=..., prediction=...)    # from an attendee's notebook
    show_leaderboard("mosaic")                                              # from anyone's, on the projector

The folder is ``<base>/<game>/``, where ``<base>`` is the ``leaderboard: dir:`` setting of ``config.yaml`` (the shared folder on the server). If that
folder cannot be written, scores go to ``./scores`` instead, so the notebooks also work on a laptop.
"""
import getpass
import json
import os
import pathlib
import tempfile
import time

import pandas as pd


def _folder(game, base):
    """The folder of ``game`` under ``base``, or a local ``scores`` folder when the shared one is missing or not writable."""
    for candidate in (base, "scores"):
        if not candidate:
            continue
        folder = pathlib.Path(candidate).expanduser() / game
        try:
            folder.mkdir(parents=True, exist_ok=True)
            if os.access(folder, os.W_OK):
                return folder
        except OSError:
            continue
    raise OSError("No writable folder for scores")


def submit_score(game, nickname, score, steps=0, prediction=None, base=None, predicts="score", extra=None):
    """Save this person's score. Only the best result of each person is kept: a higher score wins, and fewer steps break a tie.

    ``prediction`` is the guess made before playing, and ``predicts`` says what it was a guess of: ``"score"`` or ``"steps"``."""
    folder = _folder(game, base)
    user = getpass.getuser()
    path = folder / f"{user}.json"
    entry = {"user": user, "nickname": nickname or user, "score": float(score), "steps": int(steps),
             "prediction": None if prediction is None else float(prediction), "predicts": predicts, "time": time.time(), **(extra or {})}
    if path.exists():
        try:
            old = json.loads(path.read_text())
            if (old["score"], -old["steps"]) >= (entry["score"], -entry["steps"]):
                entry = {**old, "nickname": entry["nickname"]}          # keep the better result
        except (OSError, ValueError, KeyError):
            pass
    with tempfile.NamedTemporaryFile("w", dir=folder, suffix=".tmp", delete=False) as handle:
        json.dump(entry, handle)
    os.chmod(handle.name, 0o644)
    os.replace(handle.name, path)                                       # all at once: a reader never sees half a file
    return folder


def load_scores(game, base=None):
    """Every saved score of ``game`` as a list of dictionaries."""
    folder = _folder(game, base)
    scores = []
    for path in sorted(folder.glob("*.json")):
        try:
            scores.append(json.loads(path.read_text()))
        except (OSError, ValueError):
            continue                                                    # a file being replaced: skip it this time
    return scores


def leaderboard(game, base=None, reference=None, top=15):
    """The ranked table: best score first, fewer steps on a tie. ``reference`` adds rows for an agent, such as ``{"random agent": (3.0, 400)}``."""
    rows = [{"player": s["nickname"], "score": s["score"], "steps": s["steps"]} for s in load_scores(game, base)]
    rows += [{"player": f"[{name}]", "score": score, "steps": steps} for name, (score, steps) in (reference or {}).items()]
    table = pd.DataFrame(rows, columns=["player", "score", "steps"])
    table = table.sort_values(["score", "steps"], ascending=[False, True]).reset_index(drop=True).head(top)
    table.index = table.index + 1
    table.index.name = "rank"
    return table


def forecasters(game, base=None, top=5):
    """Who guessed their own score best: the smallest gap between the prediction and the result."""
    rows = []
    for s in load_scores(game, base):
        if s.get("prediction") is not None:
            what = s.get("predicts", "score")                      # a guess of the score, or of the steps
            rows.append({"player": s["nickname"], "guessed": what, "predicted": s["prediction"], "actual": s[what],
                         "off by": abs(s["prediction"] - s[what])})
    table = pd.DataFrame(rows, columns=["player", "guessed", "predicted", "actual", "off by"])
    table = table.sort_values("off by").reset_index(drop=True).head(top)
    table.index = table.index + 1
    table.index.name = "rank"
    return table


def show_leaderboard(game, base=None, reference=None, top=15):
    """Show the ranked table and the best forecasters in a notebook."""
    from IPython.display import display
    print(f"{game}: {len(load_scores(game, base))} players")
    display(leaderboard(game, base, reference, top))
    guesses = forecasters(game, base)
    if len(guesses):
        print("Best at predicting their own score:")
        display(guesses)
