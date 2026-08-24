import random

from minigrid.core.constants import COLOR_NAMES
from minigrid.core.world_object import Ball, Door, Key
from minigrid.envs.babyai.core.verifier import ObjDesc, OpenInstr, PickupInstr

from .core.level import SARLevelGen
from .sar.actions import RescueAction
from .sar.placers import LavaPlacer


class TutorialEnv(SARLevelGen):
    def __init__(
        self,
        config=None,
        locked_door_prob=0.65,
        lava_per_room=3,
        lava_probability=0.75,
        **kwargs,
    ):
        config = config or {}
        kwargs.setdefault("num_rows", 1)
        kwargs.setdefault("num_cols", 1)
        super().__init__(**kwargs)
        self.rescue_action = RescueAction(self)
        self.locked_door_prob = locked_door_prob
        self.lava_placer = LavaPlacer(
            lava_per_room=lava_per_room, lava_probability=lava_probability
        )

    def step(self, action):
        if action == self.actions.forward:
            fx, fy = self.front_pos
            # Block movement to wall boundary cells — would crash in a single-room grid
            if fx == 0 or fy == 0 or fx == self.width - 1 or fy == self.height - 1:
                return self.gen_obs(), 0, False, False, {}
        return super().step(action)

    def gen_mission(self):
        self.agent_pos = (1, 1)
        self.agent_dir = 0

        self.lava_placer.place_all(self, self.num_rows, self.num_cols)
        target = self._place_random_objects()

        if isinstance(target, Door):
            self.mission = f"Open the {target.color} door"
            self.instrs = OpenInstr(ObjDesc(target.type, target.color))
        else:
            self.mission = f"Pick up the {target.color} {target.type}"
            self.instrs = PickupInstr(ObjDesc(target.type, target.color))

    def _place_random_objects(self):
        # Always place a door — locked with configurable probability
        door_color = random.choice(COLOR_NAMES)
        locked = random.random() < self.locked_door_prob
        door = Door(door_color, is_locked=locked)

        # Pick a random wall (0=right, 1=bottom, 2=left, 3=top) and position along it
        wall = random.randint(0, 3)
        if wall == 0:
            x, y = self.width - 1, random.randint(1, self.height - 2)
        elif wall == 1:
            x, y = random.randint(1, self.width - 2), self.height - 1
        elif wall == 2:
            x, y = 0, random.randint(1, self.height - 2)
        else:
            x, y = random.randint(1, self.width - 2), 0
        self.grid.set(x, y, door)

        # Always place two balls of distinct colors (neither matches door_color)
        other_colors = [c for c in COLOR_NAMES if c != door_color]
        random.shuffle(other_colors)
        balls = [Ball(color) for color in other_colors[:2]]
        for ball in balls:
            self.place_in_room(0, 0, ball)

        # Locked: place matching key, mission is to open the door
        if locked:
            self.place_in_room(0, 0, Key(door_color))
            return door

        # Unlocked: place a key as an additional distractor, pick random target
        key = Key(other_colors[2])
        self.place_in_room(0, 0, key)
        return random.choice(balls + [key])

    def get_mission_status(self):
        status = "tutorial"
        return {
            "saved_victims": 0,
            "remaining_victims": 0,
            "status": status,
        }
