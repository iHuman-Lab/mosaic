from __future__ import annotations

from .core.level import SARLevelGen
from minigrid.core.world_object import Door, Key
from .sar.objects import REAL_VICTIMS, VictimDown, FakeVictim
from minigrid.core.world_object import Key as MGKey, Goal, Lava


class TutorialEnv(SARLevelGen):
    """Simple tutorial environment that walks the user through parts.

    Parts:
    1 - One room: a victim is placed; user practices pickup and opening an already-open door.
    2 - One room: a locked door and a key are placed; user practices picking the key and unlocking.
    3 - One room: user practices dropping the key (put action).
    """

    def __init__(self, start_part: int = 1, total_parts: int = 3, **kwargs):
        # Force a single-room layout for clarity
        # Force a single-room layout for clarity (use explicit two-room wrapper if needed)
        kwargs.setdefault("num_rows", 1)
        kwargs.setdefault("num_cols", 1)
        super().__init__(**kwargs)

        self.current_part = start_part
        self.total_parts = total_parts
        # Tutorial progression flags
        self.part1_picked = False
        self.part1_door_opened = False
        self.part2_key_picked = False
        self.part2_door_opened = False
        self.part3_drop_target = None

    def next_part(self):
        """Advance to the next tutorial part (wraps to end)."""
        if self.current_part < self.total_parts:
            self.current_part += 1
        else:
            self.current_part = self.total_parts
        self.reset()

    def gen_mission(self):
        """Create a focused single-room mission depending on the current part."""
        # Connect single room
        self.connect_all()

        # Place the agent near the center (use a safe free tile)
        center_x = self.room_size // 2
        center_y = self.room_size // 2
        # ensure grid exists before placing
        try:
            self.grid.set(center_x, center_y, None)
        except Exception:
            pass
        # Put agent at a nearby free tile
        try:
            self.agent_pos = (1, 1)
            self.agent_dir = 0
        except Exception:
            self.place_agent()

        # Part-specific setups
        if self.current_part == 1:
            # Place one real victim for pickup practice at room center
            try:
                vx, vy = center_x, center_y
                self.grid.set(vx, vy, VictimDown())
            except Exception as e:
                print("Error placing victim in tutorial part 1:", e)
                raise

            # Add a closed (but unlocked) door on the right wall so the user
            # must press Space (toggle) to open it before moving through.
            try:
                door_x = self.room_size - 1
                door_y = center_y
                door = Door("red", is_locked=False)
                # ensure door starts closed
                try:
                    door.is_open = False
                except Exception:
                    pass
                self.grid.set(door_x, door_y, door)
                # record starting room so we can detect leaving it after pickup
                try:
                    self.part1_start_room = self.room_from_pos(*self.agent_pos)
                except Exception:
                    self.part1_start_room = None
            except Exception as e:
                print("Error adding door in tutorial part 1:", e)
                raise

        elif self.current_part == 2:
            # Add a locked door and place the corresponding key in the room
            try:
                door_x = self.room_size - 1
                door_y = center_y
                self.grid.set(door_x, door_y, Door("red", is_locked=True))
            except Exception as e:
                print("Error adding locked door in tutorial part 2:", e)
                raise
            try:
                # place the key at center for practice
                key_x, key_y = center_x, center_y
                self.grid.set(key_x, key_y, Key("red"))
            except Exception as e:
                print("Error placing key in tutorial part 2:", e)
                raise

            # Place a few fake victims and one real victim around the key
            try:
                spots = [
                    (center_x - 1, center_y),
                    (center_x + 1, center_y),
                    (center_x, center_y - 1),
                ]
                # place two fake victims and one real
                self.grid.set(spots[0][0], spots[0][1], FakeVictim("left", "up", color="red"))
                self.grid.set(spots[1][0], spots[1][1], FakeVictim("right", "up", color="red"))
                self.grid.set(spots[2][0], spots[2][1], VictimDown())
            except Exception as e:
                print("Error placing victims in tutorial part 2:", e)
                raise

        elif self.current_part == 3:
            # Place a key that the user can pick up and then practice dropping
            try:
                key_x, key_y = center_x, center_y
                self.grid.set(key_x, key_y, Key("red"))
            except Exception as e:
                print("Error placing key in tutorial part 3:", e)
                raise

        # Add a blue key at (center_x, center_y+1) if free, else green key at (center_x+1, center_y+1)
        try:
            bx, by = center_x, center_y + 1
            if self.grid.get(bx, by) is None:
                self.grid.set(bx, by, Key("blue"))
            else:
                gx, gy = center_x + 1, center_y + 1
                if self.grid.get(gx, gy) is None:
                    self.grid.set(gx, gy, Key("green"))
        except Exception as e:
            print("Error placing blue/green key in tutorial room:", e)

        # Default instruction generation fallback: provide a minimal tutorial
        # instruction object so downstream code that expects an `instrs`
        # with a `surface()` method won't crash.
        class TutorialInstr:
            def surface(self, env):
                # Return an empty surface description (best-effort)
                return []

            def reset_verifier(self, env):
                # Called by level reset to initialize any verifier state.
                # For tutorial we don't need state, but store env for introspection.
                self.env = env
                return

            def verify(self, *args, **kwargs):
                # Tutorial missions are not auto-checked; always incomplete
                return "incomplete"

        self.instrs = TutorialInstr()


    def step(self, action):
        """Intercept pickup and drop actions for tutorial rooms."""
        # Handle pickup action (victims)
        if action == getattr(self.actions, "pickup", None):
            fwd_pos = getattr(self, "front_pos", None)
            if fwd_pos is None:
                return super().step(action)
            try:
                fx, fy = int(fwd_pos[0]), int(fwd_pos[1])
            except Exception:
                return super().step(action)
            if not (0 <= fx < self.width and 0 <= fy < self.height):
                return super().step(action)
            obj = self.grid.get(fx, fy)
            reward = 0
            if isinstance(obj, REAL_VICTIMS):
                self.grid.set(*fwd_pos, None)
                self.saved_victims = getattr(self, "saved_victims", 0) + 1
                reward = 1.0
                obs = self.gen_obs()
                terminated = False
                if getattr(self, "current_part", 1) == 1:
                    self.part1_picked = True
                return obs, reward, terminated, False, {}
            else:
                return super().step(action)

        # Handle drop action (drop key in front if held)
        if action == getattr(self.actions, "drop", None):
            # Only drop if agent is holding a key
            if hasattr(self, "carrying") and self.carrying is not None and self.carrying.type == "key":
                fwd_pos = getattr(self, "front_pos", None)
                if fwd_pos is not None:
                    try:
                        fx, fy = int(fwd_pos[0]), int(fwd_pos[1])
                    except Exception:
                        fx, fy = None, None
                    if fx is not None and fy is not None and 0 <= fx < self.width and 0 <= fy < self.height:
                        if self.grid.get(fx, fy) is None:
                            # Place the key in front and clear carrying
                            self.grid.set(fx, fy, self.carrying)
                            self.carrying = None
                            obs = self.gen_obs()
                            return obs, 0.0, False, False, {"dropped": True}
            # If can't drop, just do nothing (no end)
            obs = self.gen_obs()
            return obs, 0.0, False, False, {"dropped": False}

        # Execute other actions normally
        # Intercept toggle (door open) to spawn a popup single-room window
        if action == getattr(self.actions, "toggle", None):
            fwd_pos = getattr(self, "front_pos", None)
            if fwd_pos is not None:
                try:
                    fx, fy = int(fwd_pos[0]), int(fwd_pos[1])
                except Exception:
                    fx, fy = None, None

                if fx is not None and fy is not None and 0 <= fx < self.width and 0 <= fy < self.height:
                    try:
                        from minigrid.core.world_object import Door as MGDoor
                        # check door state before toggling
                        pre_obj = self.grid.get(fx, fy)
                        pre_open = isinstance(pre_obj, MGDoor) and getattr(pre_obj, "is_open", False)
                    except Exception:
                        pre_open = True

                    # perform the toggle action via parent
                    obs, reward, terminated, truncated, info = super().step(action)

                    try:
                        post_obj = self.grid.get(fx, fy)
                        post_open = isinstance(post_obj, MGDoor) and getattr(post_obj, "is_open", False)
                    except Exception:
                        post_open = False

                    # If door transitioned from closed -> open, advance the tutorial
                    # in-place so the current window is replaced by the next part.
                    if not pre_open and post_open:
                        try:
                            self.next_part()
                            obs = self.gen_obs()
                            return obs, 0.0, False, False, {"tutorial_advanced": True}
                        except Exception:
                            pass

                    return obs, reward, terminated, truncated, info

        obs, reward, terminated, truncated, info = super().step(action)

        # After movement, if we're in tutorial part 1 and the victim was
        # picked, advance when the agent leaves the starting room.
        try:
            if (
                getattr(self, "current_part", None) == 1
                and getattr(self, "part1_picked", False)
                and getattr(self, "part1_start_room", None) is not None
            ):
                try:
                    current_room = self.room_from_pos(*self.agent_pos)
                    if current_room is not self.part1_start_room:
                        # advance and reset to load part 2
                        self.next_part()
                        return obs, reward, True, False, {"tutorial_advanced": True}
                except Exception:
                    pass
        except Exception:
            pass

        return obs, reward, terminated, truncated, info

    def validate_instrs(self, instrs):
        """Accept None instructions for tutorial mode to bypass parent validation.

        The base LevelGen expects particular instruction object types; tutorial
        missions intentionally provide no instructions, so accept `None` here.
        """
        if instrs is None:
            return

        # Accept minimal tutorial instruction objects that provide the
        # interface used by the generation code (surface, verify).
        if hasattr(instrs, "surface") and hasattr(instrs, "verify"):
            return

        return super().validate_instrs(instrs)

    def num_navs_needed(self, instrs):
        """Accept tutorial instruction objects and return zero navigations.

        The base implementation expects specific instruction classes; for the
        tutorial we simply report 0 navigation goals so generation proceeds.
        """
        if instrs is None:
            return 0

        if hasattr(instrs, "surface") and hasattr(instrs, "verify"):
            return 0

        return super().num_navs_needed(instrs)

    def _find_objects_by_type(self, obj_types):
        """Utility to find objects of given types on the grid."""
        objects = []
        for x in range(self.width):
            for y in range(self.height):
                obj = self.grid.get(x, y)
                if isinstance(obj, obj_types):
                    objects.append(obj)
        return objects

    def get_all_victims(self):
        """Return list of real victims on the grid."""
        return self._find_objects_by_type(REAL_VICTIMS)

    def get_mission_status(self):
        """Return a simple mission status dict for the info panel."""
        remaining = len(self.get_all_victims())
        saved = getattr(self, "saved_victims", 0)
        status = "incomplete"
        return {"status": status, "saved_victims": saved, "remaining_victims": remaining}
