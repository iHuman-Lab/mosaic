"""Tests for the split between generic victim placement and study calibration."""

from pathlib import Path

from experiment.placers import LavaRiskVictimPlacer
from mosaic.sar.env import PickupVictimEnv, build_sar_env
from mosaic.sar.objects import REAL_VICTIMS
from mosaic.sar.placers import LavaPlacer, VictimPlacer


def victims_with_pos(env):
    """[(x, y, victim)] for every real victim on the grid."""
    return [
        (x, y, obj)
        for y in range(env.height)
        for x in range(env.width)
        if isinstance(obj := env.grid.get(x, y), REAL_VICTIMS)
    ]


def room_of(env, x, y):
    """The room containing (x, y)."""
    for i in range(env.num_rows):
        for j in range(env.num_cols):
            room = env.get_room(i, j)
            tx, ty = room.top
            sx, sy = room.size
            if tx <= x < tx + sx and ty <= y < ty + sy:
                return room
    return None


def small_env(**kwargs):
    """A fast 2x2 environment."""
    kwargs.setdefault("room_size", 6)
    kwargs.setdefault("num_rows", 2)
    kwargs.setdefault("num_cols", 2)
    kwargs.setdefault("num_dists", 4)
    return PickupVictimEnv(render_mode=None, **kwargs)


# --- generic placer stays neutral ------------------------------------------


def test_generic_placer_leaves_victimbase_defaults():
    """Asserted before any step(), so this is about placement, not runtime decay."""
    env = small_env(victim_placer=VictimPlacer(num_fake_victims=2, num_real_victims=4))
    env.reset(seed=3)

    victims = victims_with_pos(env)
    assert victims
    for _, _, victim in victims:
        assert victim.health == 1.0
        assert victim.deplete_rate == 1.0


def test_generic_placer_spreads_directions():
    """The default direction list is an even split, not a constant."""
    placer = VictimPlacer(num_fake_victims=2, num_real_victims=8)
    dirs = placer.victim_directions(None, None, 8)

    assert len(dirs) == 8
    assert sorted(dirs) == sorted(VictimPlacer.DIRECTIONS * 2)


# --- hooks ------------------------------------------------------------------


class RecordingPlacer(VictimPlacer):
    """Records hook calls so the lifecycle can be asserted."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prepare_calls = 0
        self.caches = []
        self.order = []
        self.configure_args = []

    def prepare(self, level_gen):
        self.prepare_calls += 1
        self.caches.append([])
        self.order.append("prepare")

    def configure_victim(self, level_gen, room, victim, position):
        self.order.append("configure")
        self.configure_args.append((level_gen, room, victim, position))


def test_prepare_runs_once_per_placement_cycle():
    """Once per place_all() — not per room, not per victim."""
    placer = RecordingPlacer(num_fake_victims=2, num_real_victims=4)
    env = small_env(victim_placer=placer)
    env.reset(seed=5)

    assert placer.prepare_calls >= 1
    assert placer.order[0] == "prepare"
    assert placer.order.count("configure") > placer.prepare_calls


def test_prepare_rebuilds_state_each_cycle():
    """State is rebuilt per cycle, so it cannot go stale across episodes."""
    placer = RecordingPlacer(num_fake_victims=2, num_real_victims=4)
    env = small_env(victim_placer=placer)

    env.reset(seed=7)
    after_first = placer.prepare_calls
    env.reset(seed=8)

    assert placer.prepare_calls > after_first
    assert len({id(cache) for cache in placer.caches}) == placer.prepare_calls


def test_configure_victim_receives_live_context():
    """The hook gets the env, the containing room, and a victim already placed."""
    placer = RecordingPlacer(num_fake_victims=2, num_real_victims=4)
    env = small_env(victim_placer=placer)
    env.reset(seed=9)

    assert placer.configure_args
    for level_gen, room, victim, position in placer.configure_args:
        assert level_gen is env
        assert room is room_of(env, *position)
        assert isinstance(victim, REAL_VICTIMS)


# --- study calibration preserved --------------------------------------------


def configured(direction, position, lava, doors):
    """Run configure_victim in isolation and return (health, deplete_rate)."""
    placer = LavaRiskVictimPlacer()
    placer._lava, placer._doors = lava, doors
    victim = placer._make_victim(direction)
    placer.configure_victim(None, None, victim, position)
    return victim.health, victim.deplete_rate


def test_down_victims_short_circuit_regardless_of_lava():
    assert configured("down", (5, 5), [(5, 6)], []) == (0.75, 5.0)
    assert configured("down", (5, 5), [], []) == (0.75, 5.0)


def test_health_and_rate_without_lava():
    assert configured("up", (5, 5), [], []) == (0.90, 0.5)


def test_rate_by_orientation_and_distance_tier():
    """Facing toward lava depletes fastest; distance and doors soften it."""
    lava = [(5, 1)]  # directly north of the victim at (5, 5)

    # near tier (d_lava <= 2)
    assert configured("up", (5, 3), lava, [])[1] == 3.25  # toward
    assert configured("down", (5, 3), lava, [])[1] == 5.0  # short-circuits
    assert configured("left", (5, 3), lava, [])[1] == 2.5  # perp

    # medium tier (2 < d_lava <= 5)
    assert configured("up", (5, 5), lava, [])[1] == 2.2
    assert configured("left", (5, 5), lava, [])[1] == 1.0

    # safe tier (d_lava > 5)
    assert configured("up", (5, 9), lava, [])[1] == 0.75
    assert configured("left", (5, 9), lava, [])[1] == 0.5

    # a nearby door overrides the lava tier entirely
    assert configured("up", (5, 3), lava, [(5, 4)])[1] == 0.25


def test_starting_health_by_direction():
    lava = [(5, 1)]
    assert configured("up", (5, 5), lava, [])[0] == 0.90
    assert configured("left", (5, 5), lava, [])[0] == 0.75
    assert configured("right", (5, 5), lava, [])[0] == 0.75


def test_locked_rooms_hold_only_the_important_victim():
    placer = LavaRiskVictimPlacer(
        num_fake_victims=2, num_real_victims=8, important_victim="down"
    )
    env = small_env(victim_placer=placer, lava_placer=LavaPlacer(lava_per_room=2))
    env.reset(seed=11)

    locked, unlocked = [], []
    for x, y, victim in victims_with_pos(env):
        target = locked if room_of(env, x, y).locked else unlocked
        target.append(victim.direction)

    assert locked
    assert set(locked) == {"down"}
    assert len(set(unlocked)) > 1


def test_study_placer_assigns_calibrated_values_end_to_end():
    """Through a real episode, victims carry study values rather than the defaults."""
    env = build_sar_env(
        screen_size=400,
        victim_placer_cls=LavaRiskVictimPlacer,
        num_rows=2,
        num_cols=2,
        room_size=6,
        num_real_victims=4,
        num_fake_victims=2,
        lava_per_room=2,
    )
    env.reset(seed=13)

    # Every value the calibration can produce: the table, plus the "down"
    # short-circuit and the no-lava fallback.
    rates = {
        rate
        for tiers in LavaRiskVictimPlacer._DEPLETE_RATES.values()
        for rate in tiers.values()
    } | {5.0, 0.5}
    healths = set(LavaRiskVictimPlacer._STARTING_HEALTH.values()) | {0.75, 0.90}

    victims = victims_with_pos(env)
    assert victims
    for _, _, victim in victims:
        assert victim.health in healths
        assert victim.deplete_rate in rates
    # At least one victim differs from the generic default, proving calibration ran.
    assert any(
        (v.health, v.deplete_rate) != (1.0, 1.0) for _, _, v in victims
    )


# --- injection --------------------------------------------------------------


def test_build_sar_env_injects_the_placer_class():
    env = build_sar_env(
        screen_size=400,
        victim_placer_cls=LavaRiskVictimPlacer,
        num_rows=2,
        num_cols=2,
        room_size=6,
    )
    assert isinstance(env.victim_placer, LavaRiskVictimPlacer)


def test_game_wires_the_study_placer():
    """experiment/game.py must inject the study placer.

    Checked as source text because importing it requires the ixp experiment
    runner, which is not a test dependency.
    """
    source = (Path(__file__).parent.parent / "src/experiment/game.py").read_text()
    assert "victim_placer_cls=LavaRiskVictimPlacer" in source
