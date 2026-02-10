import pygame
from minigrid.manual_control import ManualControl


class User:
    def __init__(self, env):
        self.env = env
        self.controller = ManualControl(env)

    def handle_key(self, event):
        # Detect tutorial 'n' key without mutating the event
        try:
            if event.type == pygame.KEYDOWN and pygame.key.name(event.key) == "n":
                if hasattr(self.env, "next_part"):
                    try:
                        self.env.next_part()
                        return
                    except Exception:
                        pass
        except Exception:
            pass

        # Map common keys directly to env actions for reliable movement
        # Prefer raw key constants for reliability (avoid name mapping issues)
        try:
            raw_key = int(event.key)
        except Exception:
            raw_key = None
        try:
            key_name = pygame.key.name(event.key)
        except Exception:
            key_name = None

        # Direct constant checks (robust): Up moves forward, Tab pickup, Space toggle
        if event.type == pygame.KEYDOWN and raw_key is not None:
            # Debug: show raw key and name for troubleshooting rotation
            try:
                print(f"DEBUG_KEYS: raw_key={raw_key}, name={pygame.key.name(raw_key)}")
            except Exception:
                pass

            # Accept name variants that include '<' or '>' (shifted comma/period)
            try:
                name_check = pygame.key.name(raw_key)
            except Exception:
                name_check = key_name or ""
            if "<" in name_check or "," in name_check:
                action = getattr(self.env.actions, "left", None)
                if action is not None:
                    try:
                        self.env.step(action)
                        self.env.render()
                        return
                    except Exception:
                        pass
            if ">" in name_check or "." in name_check:
                action = getattr(self.env.actions, "right", None)
                if action is not None:
                    try:
                        self.env.step(action)
                        self.env.render()
                        return
                    except Exception:
                        pass
            # Use left/right arrow keys to rotate as requested
            if raw_key == pygame.K_LEFT:
                action = getattr(self.env.actions, "left", None)
                if action is not None:
                    try:
                        self.env.step(action)
                        self.env.render()
                        return
                    except Exception:
                        pass
            if raw_key == pygame.K_RIGHT:
                action = getattr(self.env.actions, "right", None)
                if action is not None:
                    try:
                        self.env.step(action)
                        self.env.render()
                        return
                    except Exception:
                        pass
            if raw_key == pygame.K_UP:
                action = getattr(self.env.actions, "forward", None)
                if action is not None:
                    try:
                        obs, reward, terminated, truncated, info = self.env.step(action)
                        self.env.render()
                        return
                    except Exception:
                        pass
            if raw_key == pygame.K_TAB:
                action = getattr(self.env.actions, "pickup", None)
                if action is not None:
                    try:
                        obs, reward, terminated, truncated, info = self.env.step(action)
                        self.env.render()
                        return
                    except Exception:
                        pass
            if raw_key == pygame.K_LSHIFT or raw_key == pygame.K_RSHIFT:
                action = getattr(self.env.actions, "drop", None)
                if action is not None:
                    try:
                        obs, reward, terminated, truncated, info = self.env.step(action)
                        self.env.render()
                        return
                    except Exception:
                        pass
            if raw_key == pygame.K_SPACE:
                action = getattr(self.env.actions, "toggle", None)
                if action is not None:
                    try:
                        obs, reward, terminated, truncated, info = self.env.step(action)
                        self.env.render()
                        return
                    except Exception:
                        pass
                # Rotation keys: comma (',') and period ('.') rotate left/right
                if raw_key == pygame.K_COMMA:
                    action = getattr(self.env.actions, "left", None)
                    if action is not None:
                        try:
                            self.env.step(action)
                            self.env.render()
                            return
                        except Exception:
                            pass
                if raw_key == pygame.K_PERIOD:
                    action = getattr(self.env.actions, "right", None)
                    if action is not None:
                        try:
                            self.env.step(action)
                            self.env.render()
                            return
                        except Exception:
                            pass

        # Controls: only 'up' moves forward; ',' '<' rotate left; '.' '>' rotate right
        key_to_action = {
            "up": getattr(self.env.actions, "forward", None),
            "tab": getattr(self.env.actions, "pickup", None),
            "space": getattr(self.env.actions, "toggle", None),
            ",": getattr(self.env.actions, "left", None),
            "<": getattr(self.env.actions, "left", None),
            ".": getattr(self.env.actions, "right", None),
            ">": getattr(self.env.actions, "right", None),
            # keep some common aliases
            "left": getattr(self.env.actions, "left", None),
            "right": getattr(self.env.actions, "right", None),
            "shift": getattr(self.env.actions, "drop", None),
        }

        action = key_to_action.get(key_name)
        # No absolute-move fallback: arrow keys other than 'up' should not move.
        if action is not None and event.type == pygame.KEYDOWN:
            # Execute action via env.step() and render/update state
            try:
                print(f"DEBUG: key_name={key_name}, mapped_action={action}")
                before_pos = getattr(self.env, 'agent_pos', None)
                print(f"DEBUG: agent_pos before={before_pos}")
                obs, reward, terminated, truncated, info = self.env.step(action)
                after_pos = getattr(self.env, 'agent_pos', None)
                print(f"DEBUG: agent_pos after={after_pos}")
                print(f"step={getattr(self.env, 'step_count', 0)}, reward={reward:.2f}")
                if terminated or truncated:
                    self.reset()
                else:
                    self.env.render()
            except Exception:
                # Fallback to ManualControl for anything unexpected
                fake_event = pygame.event.Event(event.type, {"key": key_name})
                self.controller.key_handler(fake_event)
            return

        # Fallback to ManualControl for other keys/events
        try:
            fake_event = pygame.event.Event(event.type, {"key": key_name})
        except Exception:
            fake_event = event
        self.controller.key_handler(fake_event)

    def get_frame(self):
        return self.env.render()

    def reset(self):
        self.controller.reset()
