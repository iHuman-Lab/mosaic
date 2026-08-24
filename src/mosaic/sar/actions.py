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
        events = []

        if isinstance(obj, REAL_VICTIMS):
            self.env.grid.set(*fwd_pos, None)
            obj.cur_pos = None
            if obj.health <= 0:
                reward = self.rewards.real_victim_dead
                events.append({"type": "dead_victim_picked", "reward": reward})
            else:
                self.env.saved_victims += 1
                reward = self.rewards.real_victim_alive
                events.append({"type": "victim_rescued", "reward": reward})
        elif isinstance(obj, FAKE_VICTIMS):
            self.env.grid.set(*fwd_pos, None)
            obj.cur_pos = None
            reward = self.rewards.fake_victim
            events.append({"type": "wrong_victim", "reward": reward})
        else:
            # fallback to normal pickup
            return self.fallback(self.env.actions.pickup)

        terminated, bonus, info = self.verify()
        reward += bonus
        # PickupAllVictimsInstr.verify() only ever succeeds once the grid has
        # zero real victims left, and only this branch removes one — so a
        # non-pickup action can never be the one to trigger mission_complete.
        if info.get("mission_complete"):
            events.append({"type": "mission_complete", "reward": bonus})
        info["events"] = events

        obs = self.env.gen_obs()
        return obs, reward, terminated, False, info
