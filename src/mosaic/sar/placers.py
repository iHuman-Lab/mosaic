import random

from minigrid.core.world_object import Door, Lava

from ..core.placers import Placer
from .objects import Victim

_NEIGHBORS_4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


class LockedRoomPlacer(Placer):
    """Handles placement of locked rooms and their corresponding keys."""

    def __init__(self, locked_room_prob=0.35):
        self.locked_room_prob = locked_room_prob

    def _place_key(self, level_gen, num_cols, num_rows, locked_room, door):
        """Place the key for a locked door in any unlocked room other than the locked one."""
        while True:
            ki, kj = level_gen._rand_int(0, num_cols), level_gen._rand_int(0, num_rows)
            key_room = level_gen.get_room(ki, kj)
            if key_room is locked_room or getattr(key_room, "locked", False):
                continue
            level_gen.add_object(ki, kj, "key", door.color)
            return

    def place_all(self, level_gen, num_rows, num_cols):
        n_locked = max(1, int(num_cols * num_rows * self.locked_room_prob))
        added = 0

        while added < n_locked:
            i, j = level_gen._rand_int(0, num_cols), level_gen._rand_int(0, num_rows)
            room = level_gen.get_room(i, j)

            if room.locked:
                continue

            empty_doors = [idx for idx, d in enumerate(room.doors) if d is None]
            if not empty_doors:
                continue

            door_idx = random.choice(empty_doors)
            if room.neighbors[door_idx] is None:
                continue

            door, _ = level_gen.add_door(i, j, door_idx, locked=True)
            room.locked = True
            self._place_key(level_gen, num_cols, num_rows, room, door)
            added += 1


class VictimPlacer(Placer):
    """Places real victims at random unoccupied cells, never in front of a
    door. Victims get VictimBase's neutral defaults (health=1.0,
    deplete_rate=1.0) and a random direction — bare minimum, same shape as
    LockedRoomPlacer. Everything else victim-related — fake-victim decoys,
    custom health/decay, targeted placement, spread-out distribution, etc. —
    doesn't belong here — write a separate Placer in experiment/ instead;
    see experiment/placers.py.
    """

    DIRECTIONS = ["up", "down", "left", "right"]

    def __init__(self, num_real_victims=1):
        self.num_real_victims = num_real_victims

    def _door_positions(self, level_gen):
        """Single-pass scan for door cells."""
        return [
            (x, y)
            for y in range(level_gen.height)
            for x in range(level_gen.width)
            if isinstance(level_gen.grid.get(x, y), Door)
        ]

    def _room_candidates(self, level_gen, room, forbidden):
        """Free, non-forbidden interior cells of a room, read live off the grid."""
        top_x, top_y = room.top
        size_x, size_y = room.size
        return [
            (x, y)
            for y in range(top_y + 1, top_y + size_y - 1)
            for x in range(top_x + 1, top_x + size_x - 1)
            if level_gen.grid.get(x, y) is None and (x, y) not in forbidden
        ]

    def place_all(self, level_gen, num_rows, num_cols):
        """Place victims at random cells in each room, never in front of a door."""
        door_positions = self._door_positions(level_gen)

        # Cells immediately adjacent to any door — victims must not be placed here.
        door_forbidden = {
            (dx + ox, dy + oy) for dx, dy in door_positions for ox, oy in _NEIGHBORS_4
        }

        for i in range(num_rows):
            for j in range(num_cols):
                room = level_gen.get_room(i, j)
                for _ in range(self.num_real_victims):
                    candidates = self._room_candidates(level_gen, room, door_forbidden)
                    if not candidates:
                        break  # room is full, skip remaining victims
                    pos = random.choice(candidates)
                    victim = Victim(random.choice(self.DIRECTIONS), color="red")
                    level_gen.put_obj(victim, pos[0], pos[1])


class LavaPlacer(Placer):
    """Handles placement of lava obstacles in the environment."""

    def __init__(self, lava_per_room=0, lava_probability=0.3, enabled=True):
        self.lava_per_room = lava_per_room
        self.lava_probability = lava_probability
        self.enabled = enabled

    def _door_forbidden_cells(self, level_gen, top_x, top_y, size_x, size_y):
        """Return cells adjacent to any door in this room — forbidden for lava."""

        forbidden = set()
        for bx in range(top_x, top_x + size_x):
            for by in range(top_y, top_y + size_y):
                if isinstance(level_gen.grid.get(bx, by), Door):
                    for dx, dy in _NEIGHBORS_4:
                        forbidden.add((bx + dx, by + dy))
        return forbidden

    def _lava_count_for_room(self):
        """Return how many lava tiles to place, or 0 to skip."""
        if self.lava_per_room > 0:
            return self.lava_per_room
        if random.random() < self.lava_probability:
            return random.randint(1, 3)
        return 0

    def place_in_room(self, level_gen, i, j, num_lava=None):
        """Place lava tiles at random cells in the room, never adjacent to a door."""
        if num_lava is None:
            num_lava = self.lava_per_room

        room = level_gen.get_room(i, j)
        top_x, top_y = room.top
        size_x, size_y = room.size
        forbidden = self._door_forbidden_cells(level_gen, top_x, top_y, size_x, size_y)

        for _ in range(num_lava):
            candidates = [
                (x, y)
                for y in range(top_y + 1, top_y + size_y - 1)
                for x in range(top_x + 1, top_x + size_x - 1)
                if level_gen.grid.get(x, y) is None and (x, y) not in forbidden
            ]
            if not candidates:
                break
            x, y = random.choice(candidates)
            level_gen.put_obj(Lava(), x, y)

    def place_all(self, level_gen, num_rows, num_cols, skip_locked_rooms=False):
        """Place lava in all rooms based on configuration."""
        if not self.enabled:
            return

        for i in range(num_rows):
            for j in range(num_cols):
                room = level_gen.get_room(i, j)
                if skip_locked_rooms and getattr(room, "locked", False):
                    continue
                num_lava = self._lava_count_for_room()
                if num_lava > 0:
                    self.place_in_room(level_gen, i, j, num_lava)
