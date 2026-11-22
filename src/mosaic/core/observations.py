"""Building blocks for observations, shared by ``sar`` and ``gui``.

An observation post-processor takes the observation dict built so far and
returns it with more in it. The env and the GUI each have a default one, and
users add their own without touching MOSAIC::

    class LavaCount(ObservationPostProcessor):
        def process_observation(self, obs, env):
            obs["lava_in_room"] = count_lava(env)
            return obs

    env = PickupVictimEnv(observation=[GameObservation(), LavaCount()])
"""

from abc import ABC, abstractmethod


class ObservationPostProcessor(ABC):
    """Takes the observation built so far and returns it with more in it.

    ``owner`` is what the observation belongs to: the env in ``sar``, the
    ``SAREnvGUI`` in ``gui``.
    """

    def reset(self, owner) -> None:
        """Optional per-episode reset hook. No-op by default."""
        pass

    @abstractmethod
    def process_observation(self, obs: dict, owner) -> dict:
        """Return ``obs`` with this post-processor's observations added."""


def add_observations(obs: dict, extras, owner) -> dict:
    """Run ``extras`` in order over ``obs``. They may only add keys: one that
    removes or replaces an existing key raises ValueError, so an extra can
    never silently change a default that the LLM prompt or replay reads."""
    for extra in extras:
        before = dict(obs)
        obs = extra.process_observation(obs, owner)
        for key, value in before.items():
            if key not in obs or obs[key] is not value:
                raise ValueError(
                    f"{type(extra).__name__} changed existing observation "
                    f"{key!r}; extras may only add new keys"
                )
    return obs


class ObservationChain(ObservationPostProcessor):
    """The first post-processor builds the observation. The rest may only
    add keys to it (see ``add_observations``)."""

    def __init__(self, *stages: ObservationPostProcessor):
        if not stages:
            raise ValueError("ObservationChain needs at least one stage")
        self.stages = stages

    def reset(self, owner) -> None:
        for stage in self.stages:
            stage.reset(owner)

    def process_observation(self, obs: dict, owner) -> dict:
        obs = self.stages[0].process_observation(obs, owner)
        return add_observations(obs, self.stages[1:], owner)
