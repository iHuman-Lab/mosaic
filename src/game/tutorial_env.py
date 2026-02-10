from __future__ import annotations
from .core.level import SARLevelGen
from minigrid.core.world_object import Door, Key
from .sar.objects import REAL_VICTIMS, VictimDown, FakeVictim


class TutorialEnv(SARLevelGen):
    """Tutorial: pickup, unlock, drop in one room."""

    def __init__(self, start_part: int = 1, total_parts: int = 3, **kwargs):
        kwargs.setdefault("num_rows", 1); kwargs.setdefault("num_cols", 1)
        super().__init__(**kwargs)
        self.current_part = start_part; self.total_parts = total_parts
        self.part1_picked = False; self.part1_start_room = None

    def next_part(self):
        self.current_part = min(self.current_part + 1, self.total_parts); self.reset()

    def _set_if_free(self, x, y, obj):
        if 0 <= x < self.width and 0 <= y < self.height and self.grid.get(x, y) is None:
            self.grid.set(x, y, obj)

    def _place(self, x, y, obj, overwrite=False):
        if not (0 <= x < self.width and 0 <= y < self.height):
            return
        if overwrite:
            try: self.grid.set(x, y, obj); return
            except Exception: pass
        self._set_if_free(x, y, obj)

    def gen_mission(self):
        self.connect_all(); cx = cy = self.room_size // 2
        try: self.grid.set(cx, cy, None)
        except Exception: pass
        try: self.agent_pos, self.agent_dir = (1, 1), 0
        except Exception: self.place_agent()

        if self.current_part == 1:
            self._set_if_free(cx, cy, VictimDown())
            d = Door("red", is_locked=False)
            try: d.is_open = False
            except Exception: pass
            self._place(self.width - 1, cy, d, overwrite=True)
            self._set_if_free(cx - 1, cy, Key("red"))
            try: self.part1_start_room = self.room_from_pos(*self.agent_pos)
            except Exception: self.part1_start_room = None

        elif self.current_part == 2:
            d2 = Door("red", is_locked=True)
            try: d2.is_open = False
            except Exception: pass
            self._place(self.width - 1, cy, d2, overwrite=True)
            self._set_if_free(cx, cy, Key("red"))
            spots = [(cx - 1, cy), (cx + 1, cy), (cx, cy - 1)]
            self._set_if_free(spots[0][0], spots[0][1], FakeVictim("left", "up", color="red"))
            self._set_if_free(spots[1][0], spots[1][1], FakeVictim("right", "up", color="red"))
            self._set_if_free(spots[2][0], spots[2][1], VictimDown())
        else:
            self._set_if_free(cx, cy, Key("red"))
        self._set_if_free(cx, cy + 1, Key("blue"))
        self.instrs = type("TutorialInstr", (), {"surface": lambda s, e: [], "reset_verifier": lambda s, e: setattr(s, "env", e) or None, "verify": lambda s, *a, **k: "incomplete"})()

    def step(self, action):
        pickup = getattr(self.actions, "pickup", None)
        if action == pickup:
            fwd = getattr(self, "front_pos", None)
            if fwd is None: return super().step(action)
            fx, fy = int(fwd[0]), int(fwd[1])
            if not (0 <= fx < self.width and 0 <= fy < self.height): return super().step(action)
            obj = self.grid.get(fx, fy)
            if isinstance(obj, REAL_VICTIMS):
                self.grid.set(fx, fy, None); self.saved_victims = getattr(self, "saved_victims", 0) + 1
                self.part1_picked = self.part1_picked or (self.current_part == 1)
                return self.gen_obs(), 1.0, False, False, {}
            try:
                from minigrid.core.world_object import Key as MGKey
                if isinstance(obj, MGKey) or getattr(obj, "type", None) == "key":
                    self.carrying = obj; self.grid.set(fx, fy, None)
                    return self.gen_obs(), 0.0, False, False, {"picked_key": True}
            except Exception: pass
            return super().step(action)

        if action == getattr(self.actions, "drop", None):
            if getattr(self, "carrying", None) and getattr(self.carrying, "type", None) == "key":
                fwd = getattr(self, "front_pos", None)
                if fwd is not None and self.grid.get(int(fwd[0]), int(fwd[1])) is None:
                    self.grid.set(int(fwd[0]), int(fwd[1]), self.carrying); self.carrying = None
                    return self.gen_obs(), 0.0, False, False, {"dropped": True}
            return self.gen_obs(), 0.0, False, False, {"dropped": False}

        if action == getattr(self.actions, "toggle", None):
            fwd = getattr(self, "front_pos", None)
            try:
                if fwd is not None:
                    fx, fy = int(fwd[0]), int(fwd[1])
                    from minigrid.core.world_object import Door as MGDoor
                    obj = self.grid.get(fx, fy)
                    if isinstance(obj, MGDoor) and getattr(obj, "is_locked", False):
                        carrying = getattr(self, "carrying", None)
                        if carrying is not None and getattr(carrying, "type", None) == "key" and getattr(carrying, "color", None) == getattr(obj, "color", None):
                            obj.is_locked = False
                            try: obj.is_open = True
                            except Exception: pass
                            self.carrying = None
                            try: self.next_part(); return self.gen_obs(), 0.0, False, False, {"tutorial_advanced": True}
                            except Exception: pass
                        else: return super().step(action)
            except Exception: pass
            pre_open = True
            try:
                from minigrid.core.world_object import Door as MGDoor
                if fwd is not None:
                    o = self.grid.get(int(fwd[0]), int(fwd[1])); pre_open = isinstance(o, MGDoor) and getattr(o, "is_open", False)
            except Exception: pre_open = True
            obs, reward, term, trunc, info = super().step(action)
            post_open = False
            try:
                if fwd is not None:
                    from minigrid.core.world_object import Door as MGDoor
                    o2 = self.grid.get(int(fwd[0]), int(fwd[1])); post_open = isinstance(o2, MGDoor) and getattr(o2, "is_open", False)
            except Exception: post_open = False
            if not pre_open and post_open:
                try: self.next_part(); return self.gen_obs(), 0.0, False, False, {"tutorial_advanced": True}
                except Exception: pass
            return obs, reward, term, trunc, info

        obs, reward, term, trunc, info = super().step(action)
        try:
            if self.current_part == 1 and self.part1_picked and self.part1_start_room is not None:
                if self.room_from_pos(*self.agent_pos) is not self.part1_start_room:
                    self.next_part(); return obs, reward, True, False, {"tutorial_advanced": True}
        except Exception: pass
        return obs, reward, term, trunc, info

    def validate_instrs(self, instrs):
        if instrs is None or (hasattr(instrs, "surface") and hasattr(instrs, "verify")): return
        return super().validate_instrs(instrs)

    def num_navs_needed(self, instrs):
        if instrs is None or (hasattr(instrs, "surface") and hasattr(instrs, "verify")): return 0
        return super().num_navs_needed(instrs)

    def _find_objects_by_type(self, obj_types):
        return [self.grid.get(x, y) for x in range(self.width) for y in range(self.height) if isinstance(self.grid.get(x, y), obj_types)]

    def get_all_victims(self): return self._find_objects_by_type(REAL_VICTIMS)
    def get_mission_status(self): return {"status": "incomplete", "saved_victims": getattr(self, "saved_victims", 0), "remaining_victims": len(self.get_all_victims())}


class LockedDoorRoom(SARLevelGen):
    """Small room with a locked door and a red key to pick up and use."""
    def __init__(self, **kwargs): kwargs.setdefault("num_rows", 1); kwargs.setdefault("num_cols", 1); super().__init__(**kwargs)
    def _set_if_free(self, x, y, obj):
        if 0 <= x < self.width and 0 <= y < self.height and self.grid.get(x, y) is None: self.grid.set(x, y, obj)
    def gen_mission(self):
        self.connect_all(); cx = cy = self.room_size // 2
        try: self.grid.set(cx, cy, None)
        except Exception: pass
        try: self.agent_pos, self.agent_dir = (1, 1), 0
        except Exception: self.place_agent()
        door = Door("red", is_locked=True)
        try: door.is_open = False
        except Exception: pass
        self._set_if_free(self.width - 1, cy, door); self._set_if_free(cx, cy, Key("red"))
        self.instrs = type("LockedInstr", (), {"surface": lambda s, e: [], "reset_verifier": lambda s, e: setattr(s, "env", e) or None, "verify": lambda s, *a, **k: "incomplete"})()
    def step(self, action):
        pickup = getattr(self.actions, "pickup", None)
        if action == pickup:
            fwd = getattr(self, "front_pos", None)
            if fwd is None: return super().step(action)
            fx, fy = int(fwd[0]), int(fwd[1])
            if not (0 <= fx < self.width and 0 <= fy < self.height): return super().step(action)
            try:
                from minigrid.core.world_object import Key as MGKey
                obj = self.grid.get(fx, fy)
                if isinstance(obj, MGKey) or getattr(obj, "type", None) == "key":
                    self.carrying = obj; self.grid.set(fx, fy, None); return self.gen_obs(), 0.0, False, False, {"picked_key": True}
            except Exception: pass
            return super().step(action)
        if action == getattr(self.actions, "toggle", None):
            fwd = getattr(self, "front_pos", None)
            try:
                if fwd is not None:
                    fx, fy = int(fwd[0]), int(fwd[1])
                    from minigrid.core.world_object import Door as MGDoor
                    obj = self.grid.get(fx, fy)
                    if isinstance(obj, MGDoor) and getattr(obj, "is_locked", False):
                        carrying = getattr(self, "carrying", None)
                        if carrying is not None and getattr(carrying, "type", None) == "key" and getattr(carrying, "color", None) == getattr(obj, "color", None):
                            obj.is_locked = False
                            try: obj.is_open = True
                            except Exception: pass
                            self.carrying = None; return self.gen_obs(), 0.0, False, False, {"unlocked": True}
                        else: return super().step(action)
            except Exception: pass
            return super().step(action)
        return super().step(action)
