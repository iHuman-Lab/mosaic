"""Victim placement calibrated for this study."""

from minigrid.core.world_object import Door, Lava
from mosaic.sar.placers import VictimPlacer


class LavaRiskVictimPlacer(VictimPlacer):
    """VictimPlacer with this study's health, depletion, and locked-room rules."""

    # Depletion rate by facing-relative-to-lava x distance tier.
    _DEPLETE_RATES = {
        "toward": {"near": 3.25, "medium": 2.2, "safe": 0.75, "door": 0.25},
        "perp": {"near": 2.5, "medium": 1.0, "safe": 0.5, "door": 0.1},
        "away": {"near": 1.5, "medium": 0.75, "safe": 0.25, "door": 0.05},
    }
    # Starting health by facing.
    _STARTING_HEALTH = {"up": 0.90, "left": 0.75, "right": 0.75, "down": 0.60}

    def __init__(self, *args, important_victim="up", **kwargs):
        super().__init__(*args, **kwargs)
        self.important_victim = important_victim
        self._lava = []
        self._doors = []

    def prepare(self, level_gen):
        """Rebuild the lava and door cache for this placement cycle."""
        lava, doors = [], []
        for y in range(level_gen.height):
            for x in range(level_gen.width):
                obj = level_gen.grid.get(x, y)
                if isinstance(obj, Lava):
                    lava.append((x, y))
                elif isinstance(obj, Door):
                    doors.append((x, y))
        self._lava = lava
        self._doors = doors

    def victim_directions(self, level_gen, room, n):
        """Locked rooms hold only the important victim; others get the even split."""
        if room.locked:
            return [self.important_victim] * n
        return super().victim_directions(level_gen, room, n)

    def configure_victim(self, level_gen, room, victim, position):
        """Set deplete_rate and health from proximity to lava and doors."""
        direction = getattr(victim, "direction", None)

        if direction == "down":
            victim.deplete_rate, victim.health = 5.0, 0.75
            return

        if not self._lava:
            victim.deplete_rate, victim.health = 0.5, 0.90
            return

        nearest_lava = min(
            self._lava,
            key=lambda t: abs(position[0] - t[0]) + abs(position[1] - t[1]),
        )
        d_lava = abs(position[0] - nearest_lava[0]) + abs(position[1] - nearest_lava[1])
        d_door = min(
            (abs(position[0] - t[0]) + abs(position[1] - t[1]) for t in self._doors),
            default=float("inf"),
        )

        tier = self._lava_tier(d_lava, d_door)
        orientation = self._lava_orientation(direction, nearest_lava, position)

        victim.deplete_rate = self._DEPLETE_RATES[orientation][tier]
        victim.health = self._STARTING_HEALTH.get(direction, 0.90)

    def _lava_tier(self, d_lava, d_door):
        """Classify distance into a deplete-rate tier."""
        if d_door <= 2:
            return "door"
        if d_lava <= 2:
            return "near"
        if d_lava <= 5:
            return "medium"
        return "safe"

    def _lava_orientation(self, direction, nearest_lava, pos):
        """Return 'toward', 'away', or 'perp' relative to nearest lava."""
        dx = nearest_lava[0] - pos[0]
        dy = nearest_lava[1] - pos[1]
        if abs(dx) >= abs(dy):
            toward, away = ("right", "left") if dx > 0 else ("left", "right")
        else:
            toward, away = ("down", "up") if dy > 0 else ("up", "down")
        if direction == toward:
            return "toward"
        if direction == away:
            return "away"
        return "perp"
