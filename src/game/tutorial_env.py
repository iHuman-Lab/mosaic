from __future__ import annotations
from minigrid.core.world_object import Door, Key
from .core.level import SARLevelGen
from .sar.objects import REAL_VICTIMS, VictimDown, FakeVictim


class OptimizedSAREnv(SARLevelGen):
    """Single-room tutorial with 3 progressive parts.
    Only one room is shown at a time; opening the door advances to the next."""

    def __init__(self, config: dict | None = None, **kwargs):
        config = config or {}
        kwargs.pop("start_part", None)
        kwargs.pop("total_parts", None)
        kwargs.setdefault("num_rows", 1)
        kwargs.setdefault("num_cols", 1)
        super().__init__(**kwargs)
        self.current_part = config.get("start_part", 1)
        self.total_parts = config.get("total_parts", 3)
        self.saved_victims = 0

    # ── grid generation ─────────────────────────────────────────

    def gen_mission(self):
        self.connect_all()
        cx = cy = self.room_size // 2
        try: self.grid.set(cx, cy, None)
        except Exception: pass
        try: self.agent_pos, self.agent_dir = (1, 1), 0
        except Exception: self.place_agent()

        {1: self._room1, 2: self._room2, 3: self._room3
        }.get(self.current_part, self._room3)(cx, cy)

        self.mission = f"Tutorial part {self.current_part}/{self.total_parts}"
        self.instrs = type("I", (), {
            "surface": lambda s, e: [],
            "reset_verifier": lambda s, e: setattr(s, "env", e) or None,
            "verify": lambda s, *a, **k: "incomplete",
        })()

    def _room1(self, cx, cy):
        """Victim + key + unlocked door."""
        self._put(cx, cy, VictimDown())
        self._put(cx - 1, cy, Key("red"))
        self._put(cx, cy + 1, Key("blue"))
        self._door(locked=False)

    def _room2(self, cx, cy):
        """Locked door + key + real & fake victims."""
        self._put(cx, cy, Key("red"))
        self._put(cx - 1, cy, FakeVictim("left", "up", color="red"))
        self._put(cx + 1, cy, FakeVictim("right", "up", color="red"))
        self._put(cx, cy - 1, VictimDown())
        self._put(cx, cy + 1, Key("blue"))
        self._door(locked=True)

    def _room3(self, cx, cy):
        """Final room — keys only."""
        self._put(cx, cy, Key("red"))
        self._put(cx, cy + 1, Key("blue"))

    # ── helpers ──────────────────────────────────────────────────

    def _put(self, x, y, obj):
        if 0 <= x < self.width and 0 <= y < self.height and self.grid.get(x, y) is None:
            self.grid.set(x, y, obj)

    def _door(self, locked):
        cy = self.room_size // 2
        d = Door("red", is_locked=locked)
        try: d.is_open = False
        except Exception: pass
        self.grid.set(self.width - 1, cy, d)

    def _advance(self):
        if self.current_part < self.total_parts:
            self.current_part += 1
            self.reset()

    def _front(self):
        f = getattr(self, "front_pos", None)
        if f is None: return None
        pos = f() if callable(f) else f
        try: fx, fy = int(pos[0]), int(pos[1])
        except Exception: return None
        return (fx, fy) if 0 <= fx < self.width and 0 <= fy < self.height else None

    # ── step logic ───────────────────────────────────────────────

    def step(self, action):
        a = self.actions
        if action == getattr(a, "pickup", None): return self._do_pickup()
        if action == getattr(a, "drop", None):   return self._do_drop()
        if action == getattr(a, "toggle", None):  return self._do_toggle()
        return super().step(action)

    def _do_pickup(self):
        pos = self._front()
        if not pos: return super().step(self.actions.pickup)
        fx, fy = pos; obj = self.grid.get(fx, fy)
        if isinstance(obj, REAL_VICTIMS):
            self.grid.set(fx, fy, None); self.saved_victims += 1
            return self.gen_obs(), 1.0, False, False, {"rescued": True}
        if isinstance(obj, Key) or getattr(obj, "type", None) == "key":
            if getattr(self, "carrying", None) is not None:
                return super().step(self.actions.pickup)
            self.carrying = obj; self.grid.set(fx, fy, None)
            return self.gen_obs(), 0.0, False, False, {"picked_key": True}
        return super().step(self.actions.pickup)

    def _do_drop(self):
        c = getattr(self, "carrying", None)
        if c and getattr(c, "type", None) == "key":
            pos = self._front()
            if pos and self.grid.get(*pos) is None:
                self.grid.set(*pos, c); self.carrying = None
                return self.gen_obs(), 0.0, False, False, {"dropped": True}
        return self.gen_obs(), 0.0, False, False, {"dropped": False}

    def _do_toggle(self):
        pos = self._front()
        if pos:
            fx, fy = pos; obj = self.grid.get(fx, fy)
            if isinstance(obj, Door):
                if obj.is_locked:
                    c = getattr(self, "carrying", None)
                    if c and getattr(c, "type", None) == "key" and getattr(c, "color", None) == obj.color:
                        obj.is_locked = False; obj.is_open = True; self.carrying = None
                        self._advance()
                        return self.gen_obs(), 0.0, False, False, {"advanced": True}
                    return super().step(self.actions.toggle)
                was_open = getattr(obj, "is_open", False)
                obs, r, t, tr, info = super().step(self.actions.toggle)
                if not was_open and getattr(obj, "is_open", False):
                    self._advance()
                    return self.gen_obs(), 0.0, False, False, {"advanced": True}
                return obs, r, t, tr, info
        return super().step(self.actions.toggle)

    # ── overrides ────────────────────────────────────────────────

    def validate_instrs(self, instrs):
        if instrs is None or (hasattr(instrs, "surface") and hasattr(instrs, "verify")): return
        return super().validate_instrs(instrs)

    def num_navs_needed(self, instrs):
        if instrs is None or (hasattr(instrs, "surface") and hasattr(instrs, "verify")):
            return max(40, 4 * self.room_size)
        return super().num_navs_needed(instrs)

    # ── utilities ────────────────────────────────────────────────

    def get_all_victims(self):
        return [self.grid.get(x, y) for x in range(self.width)
                for y in range(self.height) if isinstance(self.grid.get(x, y), REAL_VICTIMS)]

    def get_mission_status(self):
        return {"part": self.current_part, "saved": self.saved_victims,
                "remaining": len(self.get_all_victims())}


# Backwards compatibility
TutorialEnv = OptimizedSAREnv