"""Users add observations by passing post-processors, without editing MOSAIC.
Covers the env side (``observation=[...]``) and the GUI side (``observers=``)."""

import os

import pytest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from mosaic.core.observations import ObservationPostProcessor
from mosaic.gui.main import SAREnvGUI
from mosaic.gui.user import User
from mosaic.sar.env import PickupVictimEnv
from mosaic.sar.observations import GameObservation


class AgentColumnPlusOne(ObservationPostProcessor):
    def process_observation(self, obs, owner):
        obs["column_plus_one"] = obs["agent_x"] + 1
        return obs


class OverwritesGrid(ObservationPostProcessor):
    def process_observation(self, obs, owner):
        obs["grid"] = []
        return obs


class RemovesStepCount(ObservationPostProcessor):
    def process_observation(self, obs, owner):
        del obs["step_count"]
        return obs


class RecordsResets(ObservationPostProcessor):
    def __init__(self):
        self.resets = []

    def reset(self, owner):
        self.resets.append(owner)

    def process_observation(self, obs, owner):
        return obs


def _env(**kwargs):
    return PickupVictimEnv(
        render_mode="rgb_array", room_size=6, num_rows=2, num_cols=2, **kwargs
    )


def test_default_env_observation_is_unchanged():
    obs, _ = _env().reset(seed=1)
    assert "column_plus_one" not in obs and "agent_x" in obs


def test_env_extra_is_added_on_reset_and_step():
    env = _env(observation=[GameObservation(), AgentColumnPlusOne()])
    obs, _ = env.reset(seed=1)
    assert obs["column_plus_one"] == obs["agent_x"] + 1
    obs, *_ = env.step(env.actions.done)
    assert obs["column_plus_one"] == obs["agent_x"] + 1


def test_single_processor_still_accepted():
    obs, _ = _env(observation=GameObservation()).reset(seed=1)
    assert "agent_x" in obs


@pytest.mark.parametrize("bad", [OverwritesGrid, RemovesStepCount])
def test_env_extra_cannot_replace_or_remove_a_default_key(bad):
    env = _env(observation=[GameObservation(), bad()])
    with pytest.raises(ValueError, match=bad.__name__):
        env.reset(seed=1)


def test_chain_resets_every_stage():
    recorder = RecordsResets()
    env = _env(observation=[GameObservation(), recorder])
    env.reset(seed=1)
    assert recorder.resets == [env]


def test_gui_observation_has_env_participant_and_llm_pending_keys():
    gui = SAREnvGUI(_env())
    gui.reset()
    obs = gui.observation()
    assert {"agent_x", "grid", "action", "total_reward", "events"} <= set(obs)
    assert obs["llm_pending"] is False


def test_gui_extra_is_added():
    gui = SAREnvGUI(_env(), observers=[AgentColumnPlusOne()])
    gui.reset()
    assert gui.observation()["column_plus_one"] == gui.user.obs["agent_x"] + 1


def test_gui_extra_cannot_replace_a_default_key():
    gui = SAREnvGUI(_env(), observers=[OverwritesGrid()])
    gui.reset()
    with pytest.raises(ValueError, match="OverwritesGrid"):
        gui.observation()


def test_gui_observers_are_reset_with_the_gui():
    recorder = RecordsResets()
    gui = SAREnvGUI(_env(), observers=[recorder])
    gui.reset()
    assert recorder.resets == [gui]


def test_gui_observation_does_not_modify_the_users_obs():
    gui = SAREnvGUI(_env(), observers=[AgentColumnPlusOne()])
    gui.reset()
    gui.observation()
    assert "column_plus_one" not in gui.user.obs and "action" not in gui.user.obs


def test_user_observation_works_before_reset_and_after_a_step():
    env = _env()
    env.reset(seed=1)
    user = User(env)
    assert user.observation()["action"] is None
    user.reset()
    user.step(env.actions.done)
    assert user.observation()["action"] == env.actions.done
    assert user.observation()["events"] == []
