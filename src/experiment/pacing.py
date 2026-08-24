from minigrid.core.world_object import Door
from mosaic.sar.env import PickupVictimEnv


class TunedPickupVictimEnv(PickupVictimEnv):
    """PickupVictimEnv with this lab's human-pacing max_steps formula (calibrated
    from pilot data — see git history for the rationale), visible-victim health
    decay, and the battery-display trigger methods that go with it. mosaic's
    PickupVictimEnv is bare minimum and has no health/decay concept at all —
    this class owns all of it, since decay is a per-step effect that has to run
    at a specific point in step() (before observation processing, so the
    returned obs reflects this step's decay), not something a hook could add
    onto the base class's step() cleanly. mosaic/gui/main.py wires these two
    methods defensively (getattr(..., None)) since not every env provides them."""

    human_exploration_factor = 0.1
    steps_per_victim = 4
    safety_buffer = 1.0

    def __init__(self, *args, deplete_amount_fn=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.deplete_amount_fn = deplete_amount_fn or (lambda max_steps: 1.0)

    def step(self, action_id):
        if action_id == self.actions.pickup:
            obs, reward, terminated, truncated, info = self.action.execute()
        else:
            obs, reward, terminated, truncated, info = self._step(action_id)

        info.setdefault("events", [])

        amount = self.deplete_amount_fn(self.max_steps)
        x0, y0, x1, y1 = self.camera.get_visible_bounds(self.width, self.height)
        for obj in self._victims:
            if obj.cur_pos is None:
                continue  # picked up
            x, y = obj.cur_pos
            if x0 <= x < x1 and y0 <= y < y1:
                was_alive = obj.health > 0
                obj.deplete(amount)
                if was_alive and obj.health <= 0:
                    info["events"].append({"type": "victim_died", "reward": 0.0})

        obs = self.observation.process_observation(obs, self)
        return obs, reward, terminated, truncated, info

    def show_all_victim_batteries(self, seconds: float = 5.0):
        for victim in self.get_all_victims():
            victim.show_battery(seconds)

    def hide_all_victim_batteries(self):
        for victim in self.get_all_victims():
            victim.hide_battery()

    def calculate_max_steps(self):
        """Human-friendly max step limit for this episode's generated level."""
        num_rooms = self.num_rows * self.num_cols
        num_doors = self.count_objects_by_type(Door)
        victims_per_room = getattr(self.victim_placer, "num_real_victims", 0)

        # 1. Human exploration cost per room
        nav_time_room = self.human_exploration_factor * (self.room_size**2)

        # 2. Base exploration of all rooms
        exploration_cost = num_rooms * nav_time_room

        # 3. Door & key backtracking cost
        # Each door ≈ one extra room traversal
        door_cost = num_doors * nav_time_room

        # 4. Victim rescue cost
        total_victims = num_rooms * victims_per_room
        victim_cost = total_victims * self.steps_per_victim

        # 5. Total before buffer
        raw_steps = exploration_cost + door_cost + victim_cost

        # 6. Safety buffer for human mistakes
        return int(raw_steps * self.safety_buffer)
