import time

from minigrid.core.constants import COLORS, IDX_TO_OBJECT, OBJECT_TO_IDX
from minigrid.core.world_object import WorldObj
from minigrid.utils.rendering import fill_coords, point_in_rect

# Register new objects
new_objects = [
    "victim_up",
    "victim_down",
    "victim_right",
    "victim_left",
    "fake_victim_left_up",
    "fake_victim_left_down",
    "fake_victim_left_left",
    "fake_victim_left_right",
    "fake_victim_right_up",
    "fake_victim_right_down",
    "fake_victim_right_left",
    "fake_victim_right_right",
]
for new_object in new_objects:
    if new_object not in OBJECT_TO_IDX:
        OBJECT_TO_IDX[new_object] = len(OBJECT_TO_IDX)
        IDX_TO_OBJECT[len(IDX_TO_OBJECT)] = new_object


class VictimBase(WorldObj):
    """Base class for all victim objects with common functionality."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.health = 1.0
        self.deplete_rate = 1.0

    def deplete(self, amount):
        """Decrease health by amount * deplete_rate, clamped to [0, 1].
        Negative deplete_rate means the victim regenerates health."""
        delta = amount * self.deplete_rate
        if delta >= 0:
            self.health = max(0.0, self.health - delta)
        else:
            self.health = min(1.0, self.health - delta)

    def can_overlap(self):
        """Victims cannot be walked over."""
        return False

    def can_pickup(self):
        """Victims can be picked up."""
        return True

    def render(self, img):
        """Render the victim using defined coordinates."""
        for coords in self._get_render_coords():
            fill_coords(img, point_in_rect(*coords), COLORS[self.color])
        return img

    def _get_render_coords(self):
        """Get rendering coordinates. Subclasses should override."""
        raise NotImplementedError


class Victim(VictimBase):
    """Real victim with symmetric cross shape."""

    # Coordinate mapping for each direction
    _COORDS = {
        "up": [
            (0.45, 0.55, 0.30, 0.80),  # Body vertical
            (0.25, 0.75, 0.30, 0.40),  # Arms horizontal
        ],
        "down": [
            (0.45, 0.55, 0.20, 0.70),  # Body vertical
            (0.25, 0.75, 0.60, 0.70),  # Arms horizontal
        ],
        "left": [
            (0.20, 0.70, 0.45, 0.55),  # Body horizontal
            (0.20, 0.30, 0.25, 0.75),  # Arms vertical
        ],
        "right": [
            (0.30, 0.80, 0.45, 0.55),  # Body horizontal
            (0.70, 0.80, 0.25, 0.75),  # Arms vertical
        ],
    }

    def __init__(self, direction, color="red"):
        """
        Create a victim.

        Args:
            direction: Direction the victim is facing ("up", "down", "left", "right")
            color: Color of the victim (default "red")
        """
        self.direction = direction
        self._battery_show_until = 0.0
        super().__init__(f"victim_{direction}", color)

    def show_battery(self, seconds: float = 5.0):
        self._battery_show_until = time.time() + seconds

    def hide_battery(self):
        self._battery_show_until = 0.0

    def encode(self):
        type_idx, color_idx, _ = super().encode()
        health_int = int(self.health * 20)
        battery_on = int(time.time() < self._battery_show_until)
        return (type_idx, color_idx, health_int + 21 * battery_on)

    def _get_render_coords(self):
        return self._COORDS[self.direction]

    def render(self, img):
        if time.time() < self._battery_show_until:
            fill_coords(
                img,
                point_in_rect(0.05, 0.98, 0.98 - self.health * 0.93, 0.98),
                (255, 255, 255),
            )
        super().render(img)
        return img


class FakeVictim(VictimBase):
    """Fake victim with asymmetric T-shape."""

    # Coordinate mapping for each shift and direction combination
    _COORDS = {
        ("left", "up"): [
            (0.40, 0.50, 0.30, 0.80),  # Vertical line (left-shifted)
            (0.20, 0.60, 0.30, 0.40),  # Horizontal top (left-shifted)
        ],
        ("left", "down"): [
            (0.40, 0.50, 0.20, 0.70),  # Vertical line (left-shifted)
            (0.20, 0.60, 0.60, 0.70),  # Horizontal bottom (left-shifted)
        ],
        ("left", "left"): [
            (0.20, 0.70, 0.40, 0.50),  # Horizontal line (left-shifted up)
            (0.20, 0.30, 0.20, 0.60),  # Vertical left (left-shifted up)
        ],
        ("left", "right"): [
            (0.30, 0.80, 0.40, 0.50),  # Horizontal line (left-shifted up)
            (0.70, 0.80, 0.20, 0.60),  # Vertical right (left-shifted up)
        ],
        ("right", "up"): [
            (0.50, 0.60, 0.30, 0.80),  # Vertical line (right-shifted)
            (0.40, 0.80, 0.30, 0.40),  # Horizontal top (right-shifted)
        ],
        ("right", "down"): [
            (0.50, 0.60, 0.20, 0.70),  # Vertical line (right-shifted)
            (0.40, 0.80, 0.60, 0.70),  # Horizontal bottom (right-shifted)
        ],
        ("right", "left"): [
            (0.20, 0.70, 0.50, 0.60),  # Horizontal line (right-shifted down)
            (0.20, 0.30, 0.30, 0.70),  # Vertical left (right-shifted down)
        ],
        ("right", "right"): [
            (0.30, 0.80, 0.50, 0.60),  # Horizontal line (right-shifted down)
            (0.70, 0.80, 0.30, 0.70),  # Vertical right (right-shifted down)
        ],
    }

    def __init__(self, shift, direction, color="red"):
        """
        Create a fake victim.

        Args:
            shift: Shift direction ("left" or "right")
            direction: Direction the T is pointing ("up", "down", "left", "right")
            color: Color of the fake victim (default "red")
        """
        self.shift = shift
        self.direction = direction
        super().__init__(f"fake_victim_{shift}_{direction}", color)

    def _get_render_coords(self):
        return self._COORDS[(self.shift, self.direction)]


# Constants for victim type checking
REAL_VICTIMS = (Victim,)
FAKE_VICTIMS = (FakeVictim,)
