from minigrid.core.world_object import Door

from ..core.level import SARLevelGen
from .actions import RescueAction
from .instructions import PickupAllVictimsInstr, calculate_max_steps
from .objects import REAL_VICTIMS
from .observations import GameObservation
from .placers import LavaPlacer, LockedRoomPlacer, VictimPlacer, VictimTracker


def build_sar_env(
    screen_size: int,
    num_rows: int = 5,
    num_cols: int = 5,
    num_fake_victims: int = 10,
    num_real_victims: int = 4,
    important_victim: str = "up",
    lava_per_room: int = 8,
    locked_room_prob: float = 0.35,
    tile_size: int = 64,
    **kwargs,
) -> "PickupVictimEnv":
    """Factory that creates a fully configured PickupVictimEnv."""
    victim_placer = VictimPlacer(
        num_fake_victims=num_fake_victims,
        num_real_victims=num_real_victims,
        important_victim=important_victim,
    )
    return PickupVictimEnv(
        num_rows=num_rows,
        num_cols=num_cols,
        screen_size=screen_size,
        render_mode="rgb_array",
        agent_pov=True,
        add_lava=True,
        lava_per_room=lava_per_room,
        locked_room_prob=locked_room_prob,
        tile_size=tile_size,
        victim_placer=victim_placer,
        **kwargs,
    )


class PickupVictimEnv(SARLevelGen):
    def __init__(
        self,
        room_size=14,
        num_rows=3,
        num_cols=3,
        num_dists=18,
        unblocking=False,
        add_lava=True,
        lava_per_room=8,
        lava_probability=0.5,
        locked_room_prob=0.5,
        victim_placer=None,
        **kwargs,
    ):
        # We add many distractors to increase the probability
        # of ambiguous locations within the same room
        super().__init__(
            room_size=room_size,
            num_rows=num_rows,
            num_cols=num_cols,
            num_dists=num_dists,
            locations=False,
            unblocking=unblocking,  # Configurable: False ensures all victims are reachable
            implicit_unlock=False,
            **kwargs,
        )
        self.victim_placer = victim_placer
        self.victim_tracker = VictimTracker()
        self.locked_room_placer = LockedRoomPlacer(locked_room_prob)
        self.lava_placer = LavaPlacer(
            lava_per_room=lava_per_room,
            lava_probability=lava_probability,
            enabled=add_lava,
        )

        # Custom actions
        self.rescue_action = RescueAction(self, fallback=self._step)
        self.observation = GameObservation()

    def _count_objects_by_type(self, obj_types):
        return len(self._find_objects_by_type(obj_types))

    def _find_objects_by_type(self, obj_types):
        objects = []
        for y in range(self.height):
            for x in range(self.width):
                obj = self.grid.get(x, y)
                if isinstance(obj, obj_types):
                    objects.append(obj)
        return objects

    def get_all_victims(self):
        """
        Returns a list of all victim objects currently present in the environment.
        """
        return self._find_objects_by_type(REAL_VICTIMS)

    def get_mission_status(self):
        """
        Returns the current mission status.

        Returns:
            dict: Dictionary containing:
                - 'status': 'success', 'failure', or 'incomplete'
                - 'saved_victims': Number of victims saved
                - 'total_victims': Total number of victims in the mission
                - 'remaining_victims': Number of victims left to save
        """
        if hasattr(self, "instrs") and self.instrs is not None:
            status = self.instrs.verify(self)
        else:
            status = "incomplete"

        return {
            "status": status,
            "saved_victims": self.saved_victims,
            "remaining_victims": getattr(self, "total_victims", 0) - self.saved_victims,
        }

    def validate_instrs(self, instrs):
        """
        Override to handle custom instruction types.

        Args:
            instrs: Instruction object to validate
        """
        # Allow our custom PickupAllVictimsInstr without validation
        if isinstance(instrs, PickupAllVictimsInstr):
            return

        # Fall back to parent validation for standard instructions
        super().validate_instrs(instrs)

    def num_navs_needed(self, instrs):
        """
        Override to handle custom instruction types.

        Args:
            instrs: Instruction object

        Returns:
            int: Number of navigation actions needed (for mission generation)
        """
        # For PickupAllVictimsInstr, return number of victims
        if isinstance(instrs, PickupAllVictimsInstr):
            return instrs.num_victims

        # Fall back to parent method for standard instructions
        return super().num_navs_needed(instrs)

    def reset(self, **kwargs):
        """Reset the environment and all stats."""
        self.saved_victims = 0
        self.fixed_max_steps = calculate_max_steps(
            room_size=self.room_size,
            num_cols=self.num_cols,
            num_rows=self.num_rows,
            victims_per_room=self.victim_placer.num_real_victims,
            num_doors=self._count_objects_by_type(Door),
        )
        self.max_steps = self.fixed_max_steps
        self._deplete_amount = 17.5 / self.fixed_max_steps
        obs, info = super().reset(**kwargs)
        self.instrs.reset_verifier(self)
        self.total_victims = self._count_objects_by_type(REAL_VICTIMS)
        self.victim_tracker.initialize(self.grid, self.width, self.height)
        self.camera.reset()
        obs = self.observation.process_observation(obs, self)
        return obs, info

    def gen_mission(self):
        """Generate the mission layout and instructions."""

        self.locked_room_placer.place_all(self, self.num_rows, self.num_cols)

        self.connect_all()

        # Add lava obstacles (before victims to avoid blocking them)
        self.lava_placer.place_all(self, self.num_rows, self.num_cols)

        # Place agent at the free cell farthest from all lava, outside locked rooms
        from minigrid.core.world_object import Lava

        lava_positions = [
            (x, y)
            for y in range(self.height)
            for x in range(self.width)
            if isinstance(self.grid.get(x, y), Lava)
        ]

        def min_lava_dist(x, y):
            if not lava_positions:
                return float("inf")
            return min(abs(x - lx) + abs(y - ly) for lx, ly in lava_positions)

        # Collect all free cells in non-locked rooms, pick the one farthest from lava
        best_pos, best_dist = None, -1
        for y in range(self.height):
            for x in range(self.width):
                if self.grid.get(x, y) is not None:
                    continue
                room = self.room_from_pos(x, y)
                if getattr(room, "locked", False):
                    continue
                d = min_lava_dist(x, y)
                if d > best_dist:
                    best_dist, best_pos = d, (x, y)

        if best_pos is not None:
            self.agent_pos = best_pos
            self.agent_dir = self._rand_int(0, 4)
            self.grid.set(*best_pos, None)
        else:
            while True:
                self.place_agent()
                if not self.room_from_pos(*self.agent_pos).locked:
                    break

        # Check that all objects (including victims) are reachable from agent start position
        if not self.unblocking:
            self.check_objs_reachable()

        # Add victims after checking reachability
        self.victim_placer.place_all(self, self.num_rows, self.num_cols)

        # Victim placer may have placed an object at agent_pos since it only
        # sees empty cells at placement time. Re-clear it so minigrid's reset
        # assertion (start_cell is None or can_overlap()) doesn't fail.
        self.grid.set(*self.agent_pos, None)

        victims = self.get_all_victims()

        # Create instruction to pick up all victims
        self.instrs = PickupAllVictimsInstr(victims)

    def _step(self, action):
        return super().step(action)

    def _execute_action(self, action):
        if action == self.actions.pickup:
            obs, reward, terminated, truncated, info = self.rescue_action.execute()
            self.victim_tracker.sync_after_pickup(self.grid)
            if hasattr(self, "instrs") and self.instrs is not None:
                if self.instrs.verify(self) == "success":
                    terminated = True
                    reward += 1.0
                    info["mission_complete"] = True
        else:
            obs, reward, terminated, truncated, info = self._step(action)
        return obs, reward, terminated, truncated, info

    def show_all_victim_batteries(self, seconds: float = 10.0):
        self.victim_tracker.show_visible_batteries(
            self.camera, self.width, self.height, seconds
        )

    def hide_all_victim_batteries(self):
        self.victim_tracker.hide_all_batteries()

    def _decay_visible_victim_health(self):
        self.victim_tracker.decay(
            self.camera, self.width, self.height, self._deplete_amount
        )

    def step(self, action):
        obs, reward, terminated, truncated, info = self._execute_action(action)
        self._decay_visible_victim_health()
        obs = self.observation.process_observation(obs, self)
        return obs, reward, terminated, truncated, info
