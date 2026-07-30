from .objects import FAKE_VICTIMS, REAL_VICTIMS


class BaseAction:
    """Abstract base class for all actions."""

    def __init__(self, env, fallback=None):
        self.env = env
        self.fallback = fallback

    def execute(self):
        raise NotImplementedError


class RescueAction(BaseAction):
    """Pick up a victim (reward +1) or fake victim (penalty -0.5)."""

    def execute(self):
        fwd_pos = self.env.front_pos
        obj = self.env.grid.get(*fwd_pos)
        reward = 0

        if isinstance(obj, REAL_VICTIMS):
            self.env.grid.set(*fwd_pos, None)
            if obj.health <= 0:
                reward = -20
            else:
                self.env.saved_victims += 1
                reward = 10
        elif isinstance(obj, FAKE_VICTIMS):
            self.env.grid.set(*fwd_pos, None)
            reward = -10
        else:
            # fallback to normal pickup
            return self.fallback(self.env.actions.pickup)

        obs = self.env.gen_obs()
        # Don't terminate here - let the instruction verification system handle it
        terminated = False
        return obs, reward, terminated, False, {}
