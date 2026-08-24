"""Tests for RescueAction's SAR event emission (info["events"])."""

from minigrid.core.actions import Actions

from mosaic.sar.env import PickupVictimEnv
from mosaic.sar.objects import FakeVictim, Victim
from mosaic.sar.placers import VictimPlacer


def small_env(**kwargs):
    """A fast 2x2 environment with no naturally-placed real victims, so
    tests can place exactly the victims/fakes they need at known positions."""
    kwargs.setdefault("room_size", 6)
    kwargs.setdefault("num_rows", 2)
    kwargs.setdefault("num_cols", 2)
    kwargs.setdefault("num_dists", 4)
    kwargs.setdefault("victim_placer", VictimPlacer(num_real_victims=0))
    return PickupVictimEnv(render_mode=None, **kwargs)


def _reset(env, seed=1):
    """reset() with victim_placer(num_real_victims=0) leaves total_victims=0
    at reset time, so MiniGrid's own max_steps formula (based on victim
    count) degenerates to 0 — fine for the pickup-triggered tests (which
    never reach RoomGridLevel's reward formula), but any step that goes
    through RoomGridLevel.step()'s generic verify()/_reward() path would
    divide by that zero. Pin a real budget so every test is on equal
    footing regardless of which path a given step takes."""
    env.reset(seed=seed)
    env.max_steps = 100
    return env


def _open_room(env):
    """The first unlocked room — locked rooms have no guaranteed-free key/door
    layout convenient for manual object placement."""
    for i in range(env.num_rows):
        for j in range(env.num_cols):
            room = env.get_room(i, j)
            if not getattr(room, "locked", False):
                return room
    raise AssertionError("no unlocked room found")


def _face_object(env, obj, extra_victim=None):
    """Place the agent in a free interior cell of an unlocked room, facing
    right, with obj directly in front. If extra_victim is given, place it at
    a second free interior cell so picking up obj doesn't complete the
    mission (used to isolate a single event from mission_complete)."""
    room = _open_room(env)
    top_x, top_y = room.top
    agent_pos = (top_x + 1, top_y + 1)
    obj_pos = (top_x + 2, top_y + 1)
    env.grid.set(*agent_pos, None)
    env.agent_pos = agent_pos
    env.agent_dir = 0  # facing right (+x), so front_pos == obj_pos
    env.put_obj(obj, *obj_pos)
    if extra_victim is not None:
        env.put_obj(extra_victim, top_x + 1, top_y + 3)
    env._victims = env.find_objects_by_type((Victim,))
    env.total_victims = len(env._victims)


# --- pickup-triggered events -------------------------------------------------


def test_rescue_alive_victim_emits_victim_rescued_event():
    env = small_env()
    _reset(env)
    extra = Victim("up", color="red")
    _face_object(env, Victim("up", color="red"), extra_victim=extra)

    _, reward, terminated, _, info = env.step(Actions.pickup)

    assert info["events"] == [{"type": "victim_rescued", "reward": 1.0}]
    assert reward == 1.0
    assert terminated is False


def test_pickup_dead_victim_emits_dead_victim_picked_event():
    env = small_env()
    _reset(env)
    dead = Victim("up", color="red")
    dead.health = 0.0
    extra = Victim("up", color="red")
    _face_object(env, dead, extra_victim=extra)

    _, reward, terminated, _, info = env.step(Actions.pickup)

    assert info["events"] == [{"type": "dead_victim_picked", "reward": -2.0}]
    assert reward == -2.0
    assert terminated is False


def test_pickup_fake_victim_emits_wrong_victim_event():
    env = small_env()
    _reset(env)
    extra = Victim("up", color="red")
    _face_object(env, FakeVictim("left", "up", color="red"), extra_victim=extra)

    _, reward, terminated, _, info = env.step(Actions.pickup)

    assert info["events"] == [{"type": "wrong_victim", "reward": -1.0}]
    assert reward == -1.0
    assert terminated is False


def test_fallback_pickup_has_no_events_key():
    """Picking up empty air falls back to normal pickup handling — .execute()
    itself never sets "events" for this branch; sar/env.py guarantees the
    default (see test_movement_step_has_empty_events_list)."""
    env = small_env()
    _reset(env)
    room = _open_room(env)
    top_x, top_y = room.top
    agent_pos = (top_x + 1, top_y + 1)
    env.grid.set(*agent_pos, None)
    env.agent_pos = agent_pos
    env.agent_dir = 0
    env.grid.set(top_x + 2, top_y + 1, None)  # empty cell in front

    _, _, _, _, info = env.action.execute()

    assert "events" not in info


def test_movement_step_has_empty_events_list():
    env = small_env()
    _reset(env)

    _, _, _, _, info = env.step(Actions.left)

    assert info["events"] == []


# --- mission completion ------------------------------------------------------


def test_last_alive_victim_pickup_emits_rescued_and_mission_complete():
    env = small_env()
    _reset(env)
    _face_object(env, Victim("up", color="red"))

    _, reward, terminated, _, info = env.step(Actions.pickup)

    assert info["events"] == [
        {"type": "victim_rescued", "reward": 1.0},
        {"type": "mission_complete", "reward": 1.0},
    ]
    assert reward == 2.0
    assert terminated is True
    assert info["mission_complete"] is True


def test_last_dead_victim_pickup_emits_dead_and_mission_complete():
    env = small_env()
    _reset(env)
    dead = Victim("up", color="red")
    dead.health = 0.0
    _face_object(env, dead)

    _, reward, terminated, _, info = env.step(Actions.pickup)

    assert info["events"] == [
        {"type": "dead_victim_picked", "reward": -2.0},
        {"type": "mission_complete", "reward": 1.0},
    ]
    assert reward == -1.0
    assert terminated is True
    assert info["mission_complete"] is True
