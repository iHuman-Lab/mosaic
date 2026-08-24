from ..core.level import SARLevelGen
from ..core.placers import Placer
from .actions import BaseAction, RescueAction
from .instructions import PickupAllVictimsInstr
from .objects import REAL_VICTIMS
from .observations import GameObservation, ObservationProcessor
from .placers import LavaPlacer, LockedRoomPlacer, VictimPlacer


def build_sar_env(
    screen_size: int,
    room_size=14,
    num_rows: int = 5,
    num_cols: int = 5,
    tile_size: int = 64,
    victim_placer: Placer = None,
    lava_placer: Placer = None,
    locked_room_placer: Placer = None,
    env_cls=None,
    **kwargs,
) -> "PickupVictimEnv":
    """Factory that creates a fully configured PickupVictimEnv (or a subclass,
    via env_cls — e.g. to override calculate_max_steps() with custom pacing).

    Each placer is passed as an already-constructed instance — configure it
    yourself (e.g. VictimPlacer(num_real_victims=6)) before passing it in.
    Defaults to each placer's base class, unconfigured, when omitted."""
    env_cls = env_cls or PickupVictimEnv
    return env_cls(
        room_size=room_size,
        num_rows=num_rows,
        num_cols=num_cols,
        screen_size=screen_size,
        render_mode="rgb_array",
        agent_pov=True,
        tile_size=tile_size,
        victim_placer=victim_placer or VictimPlacer(),
        lava_placer=lava_placer or LavaPlacer(),
        locked_room_placer=locked_room_placer or LockedRoomPlacer(),
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
        victim_placer: Placer = None,
        locked_room_placer: Placer = None,
        lava_placer: Placer = None,
        action: BaseAction = None,
        observation: ObservationProcessor = None,
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
        self.victim_placer = victim_placer or VictimPlacer()
        self._victims = []  # this episode's real victims; each tracks its own cur_pos
        self.total_victims = 0
        self.locked_room_placer = locked_room_placer or LockedRoomPlacer()
        self.lava_placer = lava_placer or LavaPlacer()

        # Pickup handler — the one action with custom behavior
        self.action = action or RescueAction()
        self.action.env = self
        self.action.fallback = self._step
        self.observation = observation or GameObservation()

    def count_objects_by_type(self, obj_types):
        return len(self.find_objects_by_type(obj_types))

    def find_objects_by_type(self, obj_types):
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
        return [obj for obj in self._victims if obj.cur_pos is not None]

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
            "remaining_victims": self.total_victims - self.saved_victims,
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

    def calculate_max_steps(self):
        """Compute this episode's max_steps. Override in a subclass for custom
        pacing. Default: None — MiniGrid's own generic max-steps fallback (see
        RoomGridLevel.reset()) stands unmodified.
        """
        return

    def reset(self, **kwargs):
        """Reset the environment and all stats."""
        for components in (
            self.victim_placer,
            self.locked_room_placer,
            self.lava_placer,
            self.action,
            self.observation,
        ):
            components.reset(self)

        self.saved_victims = 0
        # self._victims/total_victims are set from gen_mission(), called inside
        # super().reset() below, once this episode's victims are actually placed.
        # RoomGridLevel.reset() (a MiniGrid base class) already computes its own
        # generic self.max_steps fallback here, and already calls
        # self.instrs.reset_verifier(self), if none was fixed at __init__ time.
        obs, info = super().reset(**kwargs)
        max_steps = self.calculate_max_steps()
        if max_steps is not None:
            self.max_steps = max_steps
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

        # Re-scan now that this episode's victims are placed. Positions aren't
        # tracked separately — every victim carries its own cur_pos (set by
        # put_obj()).
        self._victims = self.find_objects_by_type(REAL_VICTIMS)
        self.total_victims = len(self._victims)

        # Create instruction to pick up all victims. num_victims is read from
        # env.total_victims in reset_verifier(), called right after this by
        # RoomGridLevel.reset() once self.instrs exists.
        self.instrs = PickupAllVictimsInstr()

    def _step(self, action):
        """Bypass PickupVictimEnv.step() (and any subclass's step() override)
        and go straight to MiniGrid's own movement/toggle/drop handling. A
        stable anchor point any subclass's step() override can call — using
        super().step() directly from a subclass would hit this class's own
        step() instead, one level up, not MiniGrid's.
        """
        return super().step(action)

    def step(self, action_id):
        if action_id == self.actions.pickup:
            obs, reward, terminated, truncated, info = self.action.execute()
        else:
            obs, reward, terminated, truncated, info = self._step(action_id)
        obs = self.observation.process_observation(obs, self)
        info.setdefault("events", [])
        return obs, reward, terminated, truncated, info
