"""Tests for TunedPickupVictimEnv's victim_died detection — the only place
in this codebase health actually decays passively (mosaic/sar has no
decay tick at all; see pacing.py's own docstring)."""

from minigrid.core.actions import Actions

from experiment.pacing import TunedPickupVictimEnv
from mosaic.core.camera import FullviewCamera
from mosaic.sar.objects import Victim
from mosaic.sar.placers import VictimPlacer


class _FullViewCamera(FullviewCamera):
    """FullviewCamera plus a no-op reset() — PickupVictimEnv.reset() always
    calls self.camera.reset(), which only EdgeFollowCamera implements."""

    def reset(self):
        pass


class _NarrowCamera(_FullViewCamera):
    """Only the top-left 3x3 tiles are ever visible."""

    def get_visible_bounds(self, grid_width, grid_height):
        return 0, 0, 3, 3


def make_env(deplete_amount=1.0, camera_strategy=None):
    """A fast 2x2 environment with no naturally-placed real victims, so
    tests can place exactly the victims they need at known positions. A
    fixed full-view camera (unless overridden) keeps every placed victim
    inside the deplete loop's visibility check regardless of position."""
    env = TunedPickupVictimEnv(
        room_size=6,
        num_rows=2,
        num_cols=2,
        num_dists=4,
        victim_placer=VictimPlacer(num_real_victims=0),
        camera_strategy=camera_strategy or _FullViewCamera(),
        deplete_amount_fn=lambda max_steps: deplete_amount,
        render_mode=None,
    )
    env.reset(seed=1)
    env.max_steps = 100  # see tests/test_actions.py::_reset for why
    return env


def _open_room(env):
    for i in range(env.num_rows):
        for j in range(env.num_cols):
            room = env.get_room(i, j)
            if not getattr(room, "locked", False):
                return room
    raise AssertionError("no unlocked room found")


def _place_victims(env, positions_and_health):
    """Place real victims at (x, y, health) tuples in an unlocked room and
    sync _victims/total_victims the way gen_mission() would."""
    room = _open_room(env)
    top_x, top_y = room.top
    victims = []
    for dx, dy, health in positions_and_health:
        victim = Victim("up", color="red")
        victim.health = health
        env.put_obj(victim, top_x + 1 + dx, top_y + 1 + dy)
        victims.append(victim)
    agent_pos = (top_x + 1, top_y + 1)
    env.grid.set(*agent_pos, None)
    env.agent_pos = agent_pos
    env.agent_dir = 0  # facing right (+x)
    env._victims = env.find_objects_by_type((Victim,))
    env.total_victims = len(env._victims)
    return victims


# --- victim_died detection ---------------------------------------------------


def test_health_depletion_to_zero_emits_victim_died_once():
    env = make_env(deplete_amount=1.0)
    (victim,) = _place_victims(env, [(2, 0, 1.0)])

    _, _, _, _, info = env.step(Actions.left)

    assert victim.health == 0.0
    assert info["events"] == [{"type": "victim_died", "reward": 0.0}]


def test_already_dead_victim_is_not_re_reported():
    env = make_env(deplete_amount=1.0)
    (victim,) = _place_victims(env, [(2, 0, 1.0)])
    env.step(Actions.left)  # crosses zero, reports once

    _, _, _, _, info = env.step(Actions.left)

    assert victim.health == 0.0
    assert info["events"] == []


def test_two_victims_crossing_zero_same_step_both_reported():
    env = make_env(deplete_amount=1.0)
    _place_victims(env, [(2, 0, 1.0), (0, 2, 1.0)])

    _, _, _, _, info = env.step(Actions.left)

    assert info["events"] == [
        {"type": "victim_died", "reward": 0.0},
        {"type": "victim_died", "reward": 0.0},
    ]


def test_victim_outside_visible_bounds_is_not_depleted():
    env = make_env(deplete_amount=1.0, camera_strategy=_NarrowCamera())
    room = _open_room(env)
    top_x, top_y = room.top
    victim = Victim("up", color="red")
    victim.health = 1.0
    env.put_obj(victim, top_x + 4, top_y + 4)  # outside the narrow 3x3 window
    agent_pos = (top_x + 1, top_y + 1)
    env.grid.set(*agent_pos, None)
    env.agent_pos = agent_pos
    env.agent_dir = 0
    env._victims = [victim]
    env.total_victims = 1

    _, _, _, _, info = env.step(Actions.left)

    assert victim.health == 1.0
    assert info["events"] == []


def test_dead_victim_picked_and_victim_died_ordered_action_before_passive():
    """Action-produced events are already in info["events"] before the
    deplete loop runs, so they always precede any victim_died appended that
    same step — this is what makes gui/feedback.py's priority tie-break
    deterministic in real play."""
    env = make_env(deplete_amount=1.0)
    # dead sits at (top_x+2, top_y+1) — directly in front of the agent that
    # _place_victims() places at (top_x+1, top_y+1) facing right (dir=0).
    dead, dying = _place_victims(env, [(1, 0, 0.0), (0, 2, 1.0)])

    _, reward, terminated, _, info = env.step(Actions.pickup)

    assert info["events"] == [
        {"type": "dead_victim_picked", "reward": -2.0},
        {"type": "victim_died", "reward": 0.0},
    ]
    assert reward == -2.0
    assert terminated is False
    assert dying.health == 0.0
