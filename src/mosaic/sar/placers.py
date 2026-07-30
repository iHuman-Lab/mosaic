import random

from minigrid.core.world_object import Door, Lava

from .objects import REAL_VICTIMS, FakeVictim, Victim

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


class LockedRoomPlacer:
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


class VictimPlacer:
    """Handles placement of victims and fake victims."""

    DIRECTIONS = ["up", "down", "left", "right"]
    SHIFTS = ["left", "right"]

    _DEPLETE_RATES = {
        "toward": {"near": 3.25, "medium": 2.2, "safe": 0.75, "door": 0.25},
        "perp": {"near": 2.5, "medium": 1.0, "safe": 0.5, "door": 0.1},
        "away": {"near": 1.5, "medium": 0.75, "safe": 0.25, "door": 0.05},
    }
    _STARTING_HEALTH = {"up": 0.90, "left": 0.75, "right": 0.75, "down": 0.60}

    def __init__(self, num_fake_victims=3, num_real_victims=1, important_victim="up"):
        self.num_fake_victims = num_fake_victims
        self.num_real_victims = num_real_victims
        self.important_victim = important_victim

    def _make_victim(self, direction):
        return Victim(direction, color="red")

    def _collect_lava_and_door_positions(self, level_gen):
        """Single-pass scan returning (lava_positions, door_positions)."""

        lava, doors = [], []
        for y in range(level_gen.height):
            for x in range(level_gen.width):
                obj = level_gen.grid.get(x, y)
                if isinstance(obj, Lava):
                    lava.append((x, y))
                elif isinstance(obj, Door):
                    doors.append((x, y))
        return lava, doors

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

    def _assign_health(self, victim, pos, lava_positions, door_positions):
        """Set victim.deplete_rate and victim.health based on proximity to lava and doors."""
        direction = getattr(victim, "direction", None)

        if direction == "down":
            victim.deplete_rate, victim.health = 5.0, 0.75
            return

        if not lava_positions:
            victim.deplete_rate, victim.health = 0.5, 0.90
            return

        nearest_lava = min(
            lava_positions, key=lambda t: abs(pos[0] - t[0]) + abs(pos[1] - t[1])
        )
        d_lava = abs(pos[0] - nearest_lava[0]) + abs(pos[1] - nearest_lava[1])
        d_door = min(
            (abs(pos[0] - t[0]) + abs(pos[1] - t[1]) for t in door_positions),
            default=float("inf"),
        )

        tier = self._lava_tier(d_lava, d_door)
        orientation = self._lava_orientation(direction, nearest_lava, pos)

        victim.deplete_rate = self._DEPLETE_RATES[orientation][tier]
        victim.health = self._STARTING_HEALTH.get(direction, 0.90)

    def place_fake_victims(self, level_gen, i, j, forbidden):
        """Place fake victims in a room, never directly in front of a door."""
        room = level_gen.get_room(i, j)
        top_x, top_y = room.top
        size_x, size_y = room.size
        candidates = [
            (x, y)
            for y in range(top_y + 1, top_y + size_y - 1)
            for x in range(top_x + 1, top_x + size_x - 1)
            if level_gen.grid.get(x, y) is None and (x, y) not in forbidden
        ]
        random.shuffle(candidates)
        for pos in candidates[: self.num_fake_victims]:
            obj = FakeVictim(
                random.choice(self.SHIFTS), random.choice(self.DIRECTIONS), color="red"
            )
            level_gen.grid.set(pos[0], pos[1], obj)

    def _make_direction_list(self, n, locked):
        """Return a shuffled list of n victim directions for a room."""
        if locked:
            return [self.important_victim] * n
        per_dir = n // 4
        dirs = self.DIRECTIONS * per_dir + random.choices(
            self.DIRECTIONS, k=n - 4 * per_dir
        )
        random.shuffle(dirs)
        return dirs

    def _place_sector_victims(
        self,
        level_gen,
        room,
        directions,
        forbidden,
        lava_positions,
        door_positions,
    ):
        """Place victims spread across room sectors, never in front of doors."""
        top_x, top_y = room.top
        size_x, size_y = room.size
        # Pre-compute interior boundary; re-filtered at point of use so stale
        # entries (cells occupied by previously placed victims) are excluded.
        interior = [
            (x, y)
            for y in range(top_y + 1, top_y + size_y - 1)
            for x in range(top_x + 1, top_x + size_x - 1)
            if (x, y) not in forbidden
        ]
        for candidates, direction in zip(
            _sector_candidates(level_gen, room, len(directions), forbidden), directions
        ):
            pool = candidates or [p for p in interior if level_gen.grid.get(*p) is None]
            if not pool:
                continue  # room is full, skip this victim
            victim = self._make_victim(direction)
            pos = random.choice(pool)
            level_gen.grid.set(pos[0], pos[1], victim)
            self._assign_health(victim, pos, lava_positions, door_positions)

    def place_all(self, level_gen, num_rows, num_cols):
        """Place victims in each room spread across sectors, never in front of doors."""
        lava_positions, door_positions = self._collect_lava_and_door_positions(
            level_gen
        )

        # Cells immediately adjacent to any door — victims must not be placed here.
        door_forbidden = {
            (dx + ox, dy + oy)
            for dx, dy in door_positions
            for ox, oy in _NEIGHBORS_4
        }

        n = self.num_real_victims
        for i in range(num_rows):
            for j in range(num_cols):
                room = level_gen.get_room(i, j)
                directions = self._make_direction_list(n, room.locked)

                self._place_sector_victims(
                    level_gen,
                    room,
                    directions,
                    door_forbidden,
                    lava_positions,
                    door_positions,
                )
                self.place_fake_victims(level_gen, i, j, door_forbidden)


class VictimTracker:
    """Tracks alive victims, handles health decay and battery display."""

    def __init__(self):
        self._positions = []  # list of (x, y, obj)

    def initialize(self, grid, width, height):
        positions = []
        for y in range(height):
            for x in range(width):
                obj = grid.get(x, y)
                if isinstance(obj, REAL_VICTIMS):
                    positions.append((x, y, obj))
        self._positions = positions

    @property
    def count(self):
        return len(self._positions)

    def sync_after_pickup(self, grid):
        self._positions = [
            (x, y, obj) for x, y, obj in self._positions if grid.get(x, y) is obj
        ]

    def _visible(self, camera, grid_width, grid_height):
        x0, y0, x1, y1 = camera.get_visible_bounds(grid_width, grid_height)
        return [
            (x, y, obj)
            for x, y, obj in self._positions
            if x0 <= x < x1 and y0 <= y < y1
        ]

    def show_visible_batteries(
        self, camera, grid_width, grid_height, seconds: float = 10.0
    ):
        for _, _, obj in self._visible(camera, grid_width, grid_height):
            obj.show_battery(seconds)

    def hide_all_batteries(self):
        for _, _, obj in self._positions:
            obj.hide_battery()

    def decay(self, camera, grid_width, grid_height, deplete_amount):
        for _, _, obj in self._visible(camera, grid_width, grid_height):
            obj.deplete(deplete_amount)


class LavaPlacer:
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
        """Place lava tiles spread across room sectors, never adjacent to doors or each other."""
        if num_lava is None:
            num_lava = self.lava_per_room

        room = level_gen.get_room(i, j)
        top_x, top_y = room.top
        size_x, size_y = room.size
        forbidden = self._door_forbidden_cells(level_gen, top_x, top_y, size_x, size_y)

        for candidates in _sector_candidates(level_gen, room, num_lava, forbidden):
            if candidates:
                x, y = random.choice(candidates)
                level_gen.grid.set(x, y, Lava())
                for dx, dy in _NEIGHBORS_8:
                    forbidden.add((x + dx, y + dy))

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
