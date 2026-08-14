"""Tests for the split between generic victim placement and study calibration."""

from functools import partial
from pathlib import Path

from experiment.placers import LavaRiskVictimPlacer, SectorSpreadLavaPlacer
from mosaic.sar.env import PickupVictimEnv, build_sar_env
from mosaic.sar.objects import REAL_VICTIMS, Victim
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
    env = small_env(victim_placer=VictimPlacer(num_real_victims=4))
    env.reset(seed=3)

    victims = victims_with_pos(env)
    assert victims
    for _, _, victim in victims:
        assert victim.health == 1.0
        assert victim.deplete_rate == 1.0


# --- study calibration preserved --------------------------------------------


def configured(direction, position, lava, doors):
    """Run _configure_victim in isolation and return (health, deplete_rate)."""
    placer = LavaRiskVictimPlacer()
    victim = Victim(direction, color="red")
    placer._configure_victim(victim, position, lava, doors)
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
        victim_placer_cls=partial(
            LavaRiskVictimPlacer, num_real_victims=4, num_fake_victims=2
        ),
        lava_placer_cls=partial(LavaPlacer, lava_per_room=2),
        num_rows=2,
        num_cols=2,
        room_size=6,
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
    assert any((v.health, v.deplete_rate) != (1.0, 1.0) for _, _, v in victims)


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


def test_build_sar_env_injects_the_lava_placer_class():
    env = build_sar_env(
        screen_size=400,
        lava_placer_cls=SectorSpreadLavaPlacer,
        num_rows=2,
        num_cols=2,
        room_size=6,
    )
    assert isinstance(env.lava_placer, SectorSpreadLavaPlacer)


def test_game_wires_the_study_placer():
    """experiment/game.py must inject the study placers.

    Checked as source text because importing it requires the ixp experiment
    runner, which is not a test dependency.
    """
    source = (Path(__file__).parent.parent / "src/experiment/game.py").read_text()
    assert "victim_placer_cls=partial(" in source
    assert "LavaRiskVictimPlacer" in source
    assert "num_fake_victims=" in source
    assert "lava_placer_cls=partial(" in source
    assert "SectorSpreadLavaPlacer" in source
