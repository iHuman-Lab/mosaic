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
    ):
        self.env = env
        self.obs = None
        self.prompt_type = prompt_type
        self.model = model
        self.provider = provider
        self.last_llm_response: str | None = None
        self.total_steps = 0
        self.episode_ended = False
        self.on_reset = None
        self.steps_since_last_llm = 0

    def step(self, action: Actions):
        self.obs, reward, terminated, truncated, info = self.env.step(action)
        self.last_action = int(action)
        self.last_reward = float(reward)
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
        self.terminated = False
        self.truncated = False
        self.obs = obs
        if self.on_reset:
            self.on_reset()

    def ask_llm(self) -> str:
        """Ask the LLM for tactical advice based on the current game state."""

        if self.obs is None:
            self.last_llm_response = "No observation available yet."
            return self.last_llm_response

        self.last_llm_response = ask(
            self.obs, model=self.model, provider=self.provider, prompt_type=self.prompt_type
        )
        self.steps_since_last_llm = 0
        return self.last_llm_response
