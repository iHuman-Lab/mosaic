import threading
import time

from minigrid.core.actions import Actions
from minigrid.manual_control import ManualControl

from ..llm.client import ask


class User(ManualControl):
    def __init__(
        self,
        env,
        prompt_type: str = "detailed",
        model: str = "gpt-4o-mini",
        provider: str = "openai",
        max_time: float = 5.0,
    ):
        self.env = env
        self.obs = None
        self.prompt_type = prompt_type
        self.model = model
        self.provider = provider
        self.max_time = max_time * 60  # convert minutes to seconds
        self.last_llm_response: str | None = None
        self.total_steps = 0
        self.episode_ended = False
        self.on_reset = None
        self.steps_since_last_llm = 0
        self._start_time: float | None = None
        self.llm_thread: threading.Thread | None = None
        self.llm_result: tuple | None = None

    @property
    def remaining_time(self) -> float:
        if self._start_time is None:
            return self.max_time
        return max(0.0, self.max_time - (time.monotonic() - self._start_time))

    def step(self, action: Actions):
        if self._start_time is None:
            self._start_time = time.monotonic()
        self.obs, reward, terminated, truncated, info = self.env.step(action)
        self.last_action = int(action)
        self.last_reward = float(reward)
        self.total_reward += self.last_reward
        self.terminated = bool(terminated)
        self.truncated = bool(truncated)
        self.total_steps += 1
        self.steps_since_last_llm += 1
        if terminated:
            self.episode_ended = True
            self.reset()
        else:
            self.env.render()

    def handle_key(self, event):
        key: str = event.key

        if key == "escape":
            self.env.close()
            return
        if key == "backspace":
            self.reset()
            return

        key_to_action = {
            "left": Actions.left,
            "right": Actions.right,
            "up": Actions.forward,
            "space": Actions.toggle,
            "pageup": Actions.pickup,
            "pagedown": Actions.drop,
            "tab": Actions.pickup,
            "left shift": Actions.drop,
            "enter": Actions.done,
        }
        if key in key_to_action:
            action = key_to_action[key]
            self.step(action)

    def get_frame(self):
        return self.env.render()

    def reset(self):
        obs, info = self.env.reset()
        self.last_action = None
        self.last_reward = 0.0
        self.total_reward = 0.0
        self.terminated = False
        self.truncated = False
        self._start_time = None
        self.obs = obs
        if self.on_reset:
            self.on_reset()

    def ask_llm(self) -> str:
        """Ask the LLM for tactical advice based on the current game state."""

        if self.obs is None:
            self.last_llm_response = "No observation available yet."
            return self.last_llm_response

        self.last_llm_response = ask(
            self.obs,
            model=self.model,
            provider=self.provider,
            prompt_type=self.prompt_type,
        )
        self.steps_since_last_llm = 0
        return self.last_llm_response

    def ask_llm_async(self):
        """Start LLM call in a background thread. Result stored in llm_result."""
        if self.obs is None:
            self.llm_result = ("reply", "No observation available yet.")
            return

        obs_snapshot = self.obs
        self.llm_result = None

        def _run():
            try:
                result = ask(
                    obs_snapshot,
                    model=self.model,
                    provider=self.provider,
                    prompt_type=self.prompt_type,
                )
                self.llm_result = ("reply", result)
            except Exception as e:
                self.llm_result = ("error", str(e))
            self.steps_since_last_llm = 0

        self.llm_thread = threading.Thread(target=_run, daemon=True)
        self.llm_thread.start()
