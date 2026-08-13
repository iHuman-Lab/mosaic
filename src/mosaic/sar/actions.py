from dataclasses import dataclass

from .objects import FAKE_VICTIMS, REAL_VICTIMS


@dataclass
class RescueRewards:
    """Reward magnitudes for RescueAction. Neutral plain-RL defaults —
    override with study-calibrated values via RescueAction(rewards=...)."""

    real_victim_alive: float = 1.0
    fake_victim: float = -1.0
    real_victim_dead: float = -2.0


class BaseAction:
    """Abstract base class for all actions."""

    def __init__(self, env=None, fallback=None):
        self.env = env
        self.fallback = fallback

    def reset(self, level_gen) -> None:
        """Optional per-episode reset hook. No-op by default."""
        pass

    def execute(self):
        raise NotImplementedError

    def verify(self):
        """Check the env's mission after this action. Returns
        (terminated, bonus_reward, info). Default: checks self.env.instrs.verify().
        Any action can call this to opt into mission-completion checking; most
        actions (e.g. movement) shouldn't call it at all."""
        if getattr(self.env, "instrs", None) is not None:
            if self.env.instrs.verify(self.env) == "success":
                return True, 1.0, {"mission_complete": True}
        return False, 0.0, {}


class RescueAction(BaseAction):
    """Pick up a real victim (alive/dead) or a fake victim."""

    def __init__(self, env=None, fallback=None, rewards=None):
        super().__init__(env=env, fallback=fallback)
        self.rewards = rewards or RescueRewards()

    def execute(self):
        fwd_pos = self.env.front_pos
        obj = self.env.grid.get(*fwd_pos)
        reward = 0

        if isinstance(obj, REAL_VICTIMS):
            self.env.grid.set(*fwd_pos, None)
            if obj.health <= 0:
                reward = self.rewards.real_victim_dead
            else:
                self.env.saved_victims += 1
                reward = self.rewards.real_victim_alive
        elif isinstance(obj, FAKE_VICTIMS):
            self.env.grid.set(*fwd_pos, None)
            reward = self.rewards.fake_victim
        else:
            # fallback to normal pickup
            return self.fallback(self.env.actions.pickup)

        self.env.victim_tracker.sync_after_pickup(self.env.grid)
        terminated, bonus, info = self.verify()
        reward += bonus

        obs = self.env.gen_obs()
        return obs, reward, terminated, False, info
