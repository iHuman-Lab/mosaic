import copy
import threading
import time
from typing import Optional

from minigrid.core.actions import Actions
from minigrid.manual_control import ManualControl

from ..llm.client import DummyLLMClient, LLMClient, ask


class User(ManualControl):
    def __init__(
        self,
        env,
        llm_client: Optional[LLMClient] = None,
        prompt_type: str = "detailed",
        prompt_builder=None,
        response_processor=None,
        max_time: float = 5.0,
    ):
        self.env = env
        self.obs = None
        if llm_client is None:
            llm_client = DummyLLMClient()
        if not isinstance(llm_client, LLMClient):
            raise TypeError("llm_client must be an LLMClient")
        self.llm_client = llm_client
        self.prompt_type = prompt_type
        self.prompt_builder = prompt_builder
        self.response_processor = response_processor
        self.max_time = max_time * 60  # convert minutes to seconds
        self.last_llm_response: str | None = None
        self.total_steps = 0
        self.episode_ended = False
        self.on_reset = None
        self.on_step = None
        self.last_info: dict = {}
        self.last_action: int | None = None
        self.last_reward = 0.0
        self.total_reward = 0.0
        self.terminated = False
        self.truncated = False
        self.steps_since_last_llm = 0
        self._start_time: float | None = None
        self.llm_thread: threading.Thread | None = None
        self.llm_result: tuple | None = None

    @property
    def remaining_time(self) -> float:
        if self._start_time is None:
            return self.max_time
        return max(0.0, self.max_time - (time.monotonic() - self._start_time))

    def observation(self) -> dict:
        """What the participant side observes right now.

        action: last action (left 0, right 1, forward 2, pickup 3, drop 4,
            toggle 5, done 6), None before the first step.
        reward, total_reward: reward of the last step and summed over the episode.
        terminated, truncated: how the episode ended, if it has.
        total_steps: steps the participant took (not cleared by reset()).
        remaining_time: seconds left on the time limit.
        steps_since_last_llm: steps since the last LLM reply.
        events: events of the last step, [] after a reset.
        prompt_type: the prompt condition.
        llm_response: last answer from ask_llm() (None for ask_llm_async()).
        """
        return {
            "action": self.last_action,
            "reward": self.last_reward,
            "total_reward": self.total_reward,
            "terminated": self.terminated,
            "truncated": self.truncated,
            "total_steps": self.total_steps,
            "remaining_time": self.remaining_time,
            "steps_since_last_llm": self.steps_since_last_llm,
            "events": list(self.last_info.get("events", [])),
            "prompt_type": self.prompt_type,
            "llm_response": self.last_llm_response,
        }

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
        self.last_info = info
        if self.on_step:
            self.on_step(info)
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
        self.last_info = {}
        if self.on_reset:
            self.on_reset()

    def _ask(self, obs):
        return ask(
            obs,
            client=self.llm_client,
            prompt_type=self.prompt_type,
            prompt_builder=self.prompt_builder,
            response_processor=self.response_processor,
        )

    def ask_llm(self) -> str:
        """Ask the LLM for tactical advice based on the current game state."""

        if self.obs is None:
            self.last_llm_response = "No observation available yet."
            return self.last_llm_response

        self.last_llm_response = self._ask(self.obs)
        self.steps_since_last_llm = 0
        return self.last_llm_response

    def ask_llm_async(self):
        """Start LLM call in a background thread. Result stored in llm_result."""
        if self.obs is None:
            self.llm_result = ("reply", "No observation available yet.")
            return

        obs_snapshot = copy.deepcopy(self.obs)
        self.llm_result = None

        def _run():
            try:
                result = self._ask(obs_snapshot)
                self.last_llm_response = result
                self.llm_result = ("reply", result)
            except Exception as e:
                self.llm_result = ("error", str(e))
            self.steps_since_last_llm = 0

        self.llm_thread = threading.Thread(target=_run, daemon=True)
        self.llm_thread.start()
