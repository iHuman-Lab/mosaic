from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class CameraConfig:
    """Configuration for camera behavior."""

    view_tiles: tuple[int, int] = (8, 8)
    margin: int = 3
    tile_size: int = 64


class CameraStrategy(ABC):
    """Abstract base class for different camera behaviors."""

    def _render_full(self, grid, tile_size, agent_pos, agent_dir) -> np.ndarray:
        return grid.render(tile_size, agent_pos, agent_dir, highlight_mask=None)

    @abstractmethod
    def get_crop(self, grid, agent_pos, agent_dir, **kwargs) -> np.ndarray:
        """Return a cropped view of the grid."""
        pass

    def get_visible_bounds(self, grid_width: int, grid_height: int) -> tuple[int, int, int, int]:
        """Return (x0, y0, x1, y1) tile bounds currently visible. Default: full grid."""
        return 0, 0, grid_width, grid_height


class FullviewCamera(CameraStrategy):
    def __init__(self, tile_size=32):
        self.tile_size = tile_size

    def get_crop(self, grid, agent_pos, agent_dir, **kwargs):
        return self._render_full(grid, self.tile_size, agent_pos, agent_dir)


class AgentCenteredCamera(CameraStrategy):
    """Camera that stays centered on the agent's room."""

    def __init__(self, extra_tiles=(4, 4), tile_size=32):
        self.extra_tiles = extra_tiles
        self.tile_size = tile_size

    def get_crop(self, grid, agent_pos, agent_dir, room=None, **kwargs) -> np.ndarray:
        """Get a crop centered on the agent's current room."""
        agent_x, agent_y = agent_pos
        room_w, room_h = room.size

        width_tiles = room_w + self.extra_tiles[0]
        height_tiles = room_h + self.extra_tiles[1]

        top_x = agent_x - width_tiles // 2
        top_y = agent_y - height_tiles // 2

        top_x = max(0, min(top_x, grid.width - width_tiles))
        top_y = max(0, min(top_y, grid.height - height_tiles))

        bot_x = top_x + width_tiles
        bot_y = top_y + height_tiles

        px_min, px_max = top_x * self.tile_size, bot_x * self.tile_size
        py_min, py_max = top_y * self.tile_size, bot_y * self.tile_size

        full_img = self._render_full(grid, self.tile_size, agent_pos, agent_dir)
        return full_img[py_min:py_max, px_min:px_max, :]


class EdgeFollowCamera(CameraStrategy):
    """Camera that follows the agent with a dead-zone margin."""

    def __init__(self, config: CameraConfig = None):
        self.config = config or CameraConfig()
        self.top_x = 0
        self.top_y = 0
        self.initialized = False

    def _initialize(self, agent_x, agent_y):
        """Initialize camera position."""
        view_w, view_h = self.config.view_tiles
        self.top_x = max(0, agent_x - view_w // 2)
        self.top_y = max(0, agent_y - view_h // 2)
        self.initialized = True

    def _update_position(self, agent_x, agent_y, grid_width, grid_height):
        """Update camera position based on agent movement."""
        if not self.initialized:
            self._initialize(agent_x, agent_y)

        view_w, view_h = self.config.view_tiles
        margin = self.config.margin

        # Calculate dead-zone boundaries
        dead_zone_left = self.top_x + margin
        dead_zone_right = self.top_x + view_w - margin - 1
        dead_zone_top = self.top_y + margin
        dead_zone_bottom = self.top_y + view_h - margin - 1

        # Move camera if agent exits dead-zone
        if agent_x < dead_zone_left:
            self.top_x -= dead_zone_left - agent_x
        elif agent_x > dead_zone_right:
            self.top_x += agent_x - dead_zone_right

        if agent_y < dead_zone_top:
            self.top_y -= dead_zone_top - agent_y
        elif agent_y > dead_zone_bottom:
            self.top_y += agent_y - dead_zone_bottom

        # Clamp to grid bounds
        self.top_x = max(0, min(self.top_x, grid_width - view_w))
        self.top_y = max(0, min(self.top_y, grid_height - view_h))

    def get_crop(
        self, grid, agent_pos, agent_dir, grid_width=None, grid_height=None, **kwargs
    ) -> np.ndarray:
        """Get a crop that follows the agent with edge-following behavior."""
        agent_x, agent_y = agent_pos
        self._update_position(agent_x, agent_y, grid_width, grid_height)

        view_w, view_h = self.config.view_tiles
        tile_size = self.config.tile_size

        # Render full grid
        full_img = self._render_full(grid, tile_size, agent_pos, agent_dir)

        # Calculate pixel boundaries
        px_min = self.top_x * tile_size
        px_max = (self.top_x + view_w) * tile_size
        py_min = self.top_y * tile_size
        py_max = (self.top_y + view_h) * tile_size

        return full_img[py_min:py_max, px_min:px_max, :]

    def get_visible_bounds(self, grid_width: int, grid_height: int) -> tuple[int, int, int, int]:
        view_w, view_h = self.config.view_tiles
        return self.top_x, self.top_y, self.top_x + view_w, self.top_y + view_h

    def reset(self):
        """Reset camera state."""
        self.initialized = False


class AgentFOVCamera(CameraStrategy):
    """Shows only the agent's current room.

    The crop is always room_size × room_size tiles (always square).
    Walls between rooms naturally prevent the agent from seeing into
    adjacent rooms — no additional fog-of-war mask is needed.
    """

    def __init__(self, config: CameraConfig = None):
        self.config = config or CameraConfig()
        self._last_room = None

    def get_crop(self, grid, agent_pos, agent_dir, room=None, **_) -> np.ndarray:
        self._last_room = room
        ts = self.config.tile_size
        full_img = self._render_full(grid, ts, agent_pos, agent_dir)

        if room is None:
            return full_img

        rx, ry = room.top
        rw, rh = room.size
        return full_img[ry * ts:(ry + rh) * ts, rx * ts:(rx + rw) * ts, :]

    def get_visible_bounds(self, grid_width: int, grid_height: int) -> tuple[int, int, int, int]:
        if self._last_room is None:
            return 0, 0, grid_width, grid_height
        rx, ry = self._last_room.top
        rw, rh = self._last_room.size
        return rx, ry, rx + rw, ry + rh

    def reset(self):
        self._last_room = None


class AgentConeCamera(AgentFOVCamera):
    """Shows the agent's current room with a forward-cone blackout.

    Extends AgentFOVCamera: same room crop, but tiles outside MiniGrid's
    built-in forward visibility cone are rendered black.
    """

    def _visibility_mask(self, env, grid_width: int, grid_height: int) -> np.ndarray:
        """Return a (grid_width, grid_height) bool array using MiniGrid's FOV cone."""
        _, vis_mask = env.gen_obs_grid()
        view_size = env.agent_view_size
        f_vec = env.dir_vec
        r_vec = env.right_vec
        top_left = env.agent_pos + f_vec * (view_size - 1) - r_vec * (view_size // 2)

        visible = np.zeros((grid_width, grid_height), dtype=bool)
        for vis_j in range(view_size):
            for vis_i in range(view_size):
                if not vis_mask[vis_i, vis_j]:
                    continue
                wx, wy = top_left - f_vec * vis_j + r_vec * vis_i
                wx, wy = int(wx), int(wy)
                if 0 <= wx < grid_width and 0 <= wy < grid_height:
                    visible[wx, wy] = True
        return visible

    def get_crop(self, grid, agent_pos, agent_dir, room=None, grid_width=None, grid_height=None, env=None, **_) -> np.ndarray:
        # Get the room crop from the parent
        crop = super().get_crop(grid, agent_pos, agent_dir, room=room)

        if room is None or env is None:
            return crop

        visible = self._visibility_mask(env, grid_width, grid_height)
        ts = self.config.tile_size
        rx, ry = room.top
        rw, rh = room.size
        result = crop.copy()
        for cy in range(rh):
            for cx in range(rw):
                if not visible[rx + cx, ry + cy]:
                    result[cy * ts:(cy + 1) * ts, cx * ts:(cx + 1) * ts] = 0
        return result
