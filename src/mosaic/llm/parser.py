"""Post-processing for LLM tactical responses.

Converts structured object labels (YellowDoor1, RedKey2, Victim3) that
are useful for the model's reasoning into natural prose for the operator.
"""

import re

_COLORS = "red|green|blue|purple|yellow|grey"

# YellowDoor1 / RedKey2 / GreenBall → the yellow door / the red key / the green ball
_COLOR_OBJ_RE = re.compile(
    rf"\b(?P<color>{_COLORS})(?P<obj>door|key|ball|box)\d*\b",
    flags=re.IGNORECASE,
)

# FakeVictim / FakeVictim3 → a decoy  (must come before victim pattern)
_FAKE_VICTIM_RE = re.compile(r"\bfakevictim\d*\b", flags=re.IGNORECASE)

# Victim3 → a victim  (only numbered — bare "victim" is already natural)
_VICTIM_NUMBERED_RE = re.compile(r"\bvictim\d+\b", flags=re.IGNORECASE)

# Coordinate references: "to (2,3)" / "at (0,1)" / "in room (1,2)"
_COORD_PREP_RE = re.compile(r"\s+(?:to|at|in room)\s+\(\d+,\s*\d+\)", re.IGNORECASE)

# Bare coordinates: (0,3)
_COORD_BARE_RE = re.compile(r"\(\d+,\s*\d+\)")

# Duplicate articles introduced by substitutions: "the the yellow door"
_DOUBLE_ARTICLE_RE = re.compile(r"\bthe\s+the\b", re.IGNORECASE)


def clean_response(text: str) -> str:
    if "<START>" in text and "<END>" in text:
        text = text.split("<START>")[1].split("<END>")[0].strip()
    else:
        text = text.strip()
    text = _FAKE_VICTIM_RE.sub("a decoy", text)
    text = _VICTIM_NUMBERED_RE.sub("a victim", text)
    text = _COLOR_OBJ_RE.sub(
        lambda m: f"the {m.group('color').lower()} {m.group('obj').lower()}", text
    )
    text = _COORD_PREP_RE.sub("", text)
    text = _COORD_BARE_RE.sub("", text)
    text = _DOUBLE_ARTICLE_RE.sub("the", text)
    text = re.sub(r"  +", " ", text).strip()
    return text
