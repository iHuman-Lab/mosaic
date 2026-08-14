import sys
from pathlib import Path

# Put src/ on the path so tests import `mosaic.*` and `experiment.*` the way the
# application does (see src/experiment/game.py), rather than as `src.mosaic.*`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
