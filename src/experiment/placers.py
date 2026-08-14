"""Victim and lava placement calibrated for this study.

mosaic's VictimPlacer/LavaPlacer are deliberately bare-minimum (random valid
cell, nothing fancy — see mosaic/sar/placers.py). The sector-spread placement
strategy and the health/decay calibration this lab's study actually needs
live here instead, fully self-contained.
"""

import random

from minigrid.core.world_object import Lava

from mosaic.sar.objects import FakeVictim, Victim
from mosaic.sar.placers import LavaPlacer, VictimPlacer

_NEIGHBORS_4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_NEIGHBORS_8 = _NEIGHBORS_4 + [(-1, -1), (1, -1), (-1, 1), (1, 1)]


def _sector_candidates(level_gen, room, n, excluded):
    """Yield a list of free candidate cells for each of n sectors in a room's interior.

    Sectors are shuffled before iteration. Because this is a generator, ``excluded``
    is evaluated lazily — callers can mutate the set between yields and the next
    sector will respect the updated contents.
    """
    top_x, top_y = room.top
    size_x, size_y = room.size
    inner_w, inner_h = size_x - 2, size_y - 2

    cols = max(1, round(((inner_w / inner_h) * n) ** 0.5))
    rows = max(1, (n + cols - 1) // cols)
    sx, sy = inner_w / cols, inner_h / rows

    sectors = [(c, r) for c in range(cols) for r in range(rows)]
    random.shuffle(sectors)

    for c, r in sectors[:n]:
        x0 = top_x + 1 + int(c * sx)
        x1 = top_x + 1 + (inner_w if c == cols - 1 else int((c + 1) * sx))
        y0 = top_y + 1 + int(r * sy)
        y1 = top_y + 1 + (inner_h if r == rows - 1 else int((r + 1) * sy))
        yield [
            (x, y)
            for x in range(x0, x1)
            for y in range(y0, y1)
            if level_gen.grid.get(x, y) is None and (x, y) not in excluded
        ]


class SectorSpreadLavaPlacer(LavaPlacer):
    """LavaPlacer that spreads tiles across room sectors and keeps them apart
    from each other and from doors — restores the placement quality mosaic's
    bare-minimum LavaPlacer doesn't provide."""

    def place_in_room(self, level_gen, i, j, num_lava=None):
        if num_lava is None:
            num_lava = self.lava_per_room

        room = level_gen.get_room(i, j)
        top_x, top_y = room.top
        size_x, size_y = room.size
        forbidden = self._door_forbidden_cells(level_gen, top_x, top_y, size_x, size_y)

        for candidates in _sector_candidates(level_gen, room, num_lava, forbidden):
            if candidates:
                x, y = random.choice(candidates)
                level_gen.put_obj(Lava(), x, y)
                for dx, dy in _NEIGHBORS_8:
                    forbidden.add((x + dx, y + dy))


class LavaRiskVictimPlacer(VictimPlacer):
    """VictimPlacer with this study's health, depletion, locked-room, and
    fake-victim rules.

    Fully self-contained: mosaic's VictimPlacer no longer provides sector-spread
    placement or fake-victim decoys, so this class reimplements them directly
    rather than relying on hooks into the base class.
    """

    SHIFTS = ["left", "right"]

    # Depletion rate by facing-relative-to-lava x distance tier.
    _DEPLETE_RATES = {
        "toward": {"near": 3.25, "medium": 2.2, "safe": 0.75, "door": 0.25},
        "perp": {"near": 2.5, "medium": 1.0, "safe": 0.5, "door": 0.1},
        "away": {"near": 1.5, "medium": 0.75, "safe": 0.25, "door": 0.05},
    }
    # Starting health by facing.
    _STARTING_HEALTH = {"up": 0.90, "left": 0.75, "right": 0.75, "down": 0.60}

    def __init__(self, *args, important_victim="up", num_fake_victims=3, **kwargs):
        super().__init__(*args, **kwargs)
        self.important_victim = important_victim
        self.num_fake_victims = num_fake_victims

    def place_fake_victims(self, level_gen, i, j, forbidden):
        """Place fake victims in a room, never directly in front of a door."""
        room = level_gen.get_room(i, j)
        candidates = self._room_candidates(level_gen, room, forbidden)
        random.shuffle(candidates)
        for pos in candidates[: self.num_fake_victims]:
            obj = FakeVictim(
                random.choice(self.SHIFTS), random.choice(self.DIRECTIONS), color="red"
            )
            level_gen.put_obj(obj, pos[0], pos[1])

    def _directions_for_room(self, room, n):
        """Locked rooms hold only the important victim; others get an even split."""
        if room.locked:
            return [self.important_victim] * n
        per_dir = n // 4
        dirs = self.DIRECTIONS * per_dir + random.choices(
            self.DIRECTIONS, k=n - 4 * per_dir
        )
        random.shuffle(dirs)
        return dirs

    def _configure_victim(self, victim, position, lava, doors):
        """Set deplete_rate and health from proximity to lava and doors."""
        direction = getattr(victim, "direction", None)

        if direction == "down":
            victim.deplete_rate, victim.health = 5.0, 0.75
            return

        if not lava:
            victim.deplete_rate, victim.health = 0.5, 0.90
            return

        nearest_lava = min(
            lava,
            key=lambda t: abs(position[0] - t[0]) + abs(position[1] - t[1]),
        )
        d_lava = abs(position[0] - nearest_lava[0]) + abs(position[1] - nearest_lava[1])
        d_door = min(
            (abs(position[0] - t[0]) + abs(position[1] - t[1]) for t in doors),
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

    def place_all(self, level_gen, num_rows, num_cols):
        """Place victims spread across room sectors, with this study's
        health/decay and locked-room targeting."""
        lava = [
            (x, y)
            for y in range(level_gen.height)
            for x in range(level_gen.width)
            if isinstance(level_gen.grid.get(x, y), Lava)
        ]
        doors = self._door_positions(level_gen)
        door_forbidden = {
            (dx + ox, dy + oy) for dx, dy in doors for ox, oy in _NEIGHBORS_4
        }

        n = self.num_real_victims
        for i in range(num_rows):
            for j in range(num_cols):
                room = level_gen.get_room(i, j)
                directions = self._directions_for_room(room, n)

                top_x, top_y = room.top
                size_x, size_y = room.size
                # Pre-compute interior boundary; re-filtered at point of use so
                # stale entries (cells occupied by previously placed victims
                # this room) are excluded.
                interior = [
                    (x, y)
                    for y in range(top_y + 1, top_y + size_y - 1)
                    for x in range(top_x + 1, top_x + size_x - 1)
                    if (x, y) not in door_forbidden
                ]
                for candidates, direction in zip(
                    _sector_candidates(
                        level_gen, room, len(directions), door_forbidden
                    ),
                    directions,
                ):
                    pool = candidates or [
                        p for p in interior if level_gen.grid.get(*p) is None
                    ]
                    if not pool:
                        continue  # room is full, skip this victim
                    victim = Victim(direction, color="red")
                    pos = random.choice(pool)
                    level_gen.put_obj(victim, pos[0], pos[1])
                    self._configure_victim(victim, pos, lava, doors)

                self.place_fake_victims(level_gen, i, j, door_forbidden)
