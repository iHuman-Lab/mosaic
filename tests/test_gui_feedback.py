import dataclasses
import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pygame
import pytest

from mosaic.gui.feedback import (
    DEFAULT_VIGNETTE_STYLES,
    EdgeVignette,
    VignetteStyle,
    _build_gradient_mask,
    select_event,
)

_RESCUED = {"type": "victim_rescued", "reward": 10}
_WRONG = {"type": "wrong_victim", "reward": -10}
_DEAD_PICKED = {"type": "dead_victim_picked", "reward": -20}
_DIED = {"type": "victim_died", "reward": 0.0}
_MISSION = {"type": "mission_complete", "reward": 1.0}


class _FakeClock:
    def __init__(self):
        self.now = 0

    def __call__(self):
        return self.now


# --- select_event: pure priority/tie-break logic -----------------------------


def test_select_event_empty_returns_none():
    assert select_event([], DEFAULT_VIGNETTE_STYLES) is None


def test_select_event_single_event_returns_it():
    assert select_event([_RESCUED], DEFAULT_VIGNETTE_STYLES) is _RESCUED


def test_select_event_unknown_type_is_filtered_out():
    assert select_event([{"type": "unknown"}], DEFAULT_VIGNETTE_STYLES) is None


@pytest.mark.parametrize(
    "events,expected",
    [
        ([_RESCUED, _DEAD_PICKED], _DEAD_PICKED),
        ([_RESCUED, _WRONG], _WRONG),
        ([_RESCUED, _MISSION], _MISSION),
        ([_WRONG, _MISSION], _WRONG),
        ([_WRONG, _DIED], _DIED),
        ([_MISSION, _DEAD_PICKED], _DEAD_PICKED),
    ],
)
def test_select_event_priority_ordering(events, expected):
    assert select_event(events, DEFAULT_VIGNETTE_STYLES) is expected


def test_select_event_tie_break_is_positional_not_type_based():
    """dead_victim_picked and victim_died share priority — min() returns the
    first minimal element, so whichever appears earlier in the list wins,
    regardless of which type it is."""
    assert select_event([_DEAD_PICKED, _DIED], DEFAULT_VIGNETTE_STYLES) is _DEAD_PICKED
    assert select_event([_DIED, _DEAD_PICKED], DEFAULT_VIGNETTE_STYLES) is _DIED


# --- _build_gradient_mask: pure numpy geometry, no pygame needed ------------

_FRACTION = 0.225
_STRENGTH = 0.065


def test_mask_boundary_pixels_are_at_peak_weight():
    mask = _build_gradient_mask(64, _FRACTION, _STRENGTH)
    assert mask[0, 0] == pytest.approx(1.0)  # corner
    assert mask[0, 32] == pytest.approx(1.0)  # mid-edge, left
    assert mask[32, 0] == pytest.approx(1.0)  # mid-edge, top
    assert mask[63, 32] == pytest.approx(1.0)  # mid-edge, opposite side


def test_mask_true_center_sits_at_center_strength_floor():
    mask = _build_gradient_mask(64, _FRACTION, _STRENGTH)
    assert mask[32, 32] == pytest.approx(_STRENGTH, abs=1e-6)


def test_mask_partial_falloff_pixel_is_strictly_between_floor_and_peak():
    mask = _build_gradient_mask(64, _FRACTION, _STRENGTH)
    assert _STRENGTH < mask[7, 32] < 1.0


def test_mask_mid_edge_pixel_gets_strong_weight_not_the_old_corner_bug():
    mask = _build_gradient_mask(64, _FRACTION, _STRENGTH)
    assert mask[0, 32] > 0.9  # was ~center_strength under the old max(dist) bug


def test_mask_corner_and_mid_edge_receive_equal_weight():
    mask = _build_gradient_mask(64, _FRACTION, _STRENGTH)
    assert mask[0, 0] == pytest.approx(mask[0, 32])


def test_mask_center_strength_is_configurable():
    low = _build_gradient_mask(64, _FRACTION, 0.05)
    high = _build_gradient_mask(64, _FRACTION, 0.08)
    assert low[32, 32] == pytest.approx(0.05, abs=1e-6)
    assert high[32, 32] == pytest.approx(0.08, abs=1e-6)


def test_mask_falloff_fraction_is_configurable():
    narrow = _build_gradient_mask(64, 0.20, _STRENGTH)
    wide = _build_gradient_mask(64, 0.25, _STRENGTH)
    assert wide[10, 32] > narrow[10, 32]  # wider ramp reaches further inward


# --- EdgeVignette: construction-time validation -----------------------------


def test_non_positive_size_raises():
    with pytest.raises(ValueError):
        EdgeVignette(size=0)
    with pytest.raises(ValueError):
        EdgeVignette(size=-10)


def test_falloff_fraction_out_of_range_raises():
    with pytest.raises(ValueError):
        EdgeVignette(size=64, falloff_fraction=0.0)
    with pytest.raises(ValueError):
        EdgeVignette(size=64, falloff_fraction=0.6)


def test_center_strength_out_of_range_raises():
    with pytest.raises(ValueError):
        EdgeVignette(size=64, center_strength=-0.1)
    with pytest.raises(ValueError):
        EdgeVignette(size=64, center_strength=1.5)


def test_unknown_envelope_in_override_raises_at_construction():
    bad = {"victim_rescued": VignetteStyle((0, 0, 255), 140, 0.4, "bogus")}
    with pytest.raises(ValueError):
        EdgeVignette(size=64, styles=bad)


def test_vignette_style_is_frozen():
    with pytest.raises(dataclasses.FrozenInstanceError):
        DEFAULT_VIGNETTE_STYLES["victim_rescued"].peak_alpha = 999


# --- EdgeVignette: timing/geometry, driven by a fake clock -------------------


def test_trigger_with_no_events_leaves_vignette_inactive():
    pygame.init()
    vignette = EdgeVignette(size=64, clock_ms=_FakeClock())
    surface = pygame.Surface((64, 64), pygame.SRCALPHA)

    vignette.trigger([])
    vignette.render(surface, pygame.Rect(0, 0, 64, 64))

    assert pygame.surfarray.array_alpha(surface).max() == 0
    pygame.quit()


def test_render_after_duration_elapses_clears_and_stops_drawing():
    pygame.init()
    clock = _FakeClock()
    vignette = EdgeVignette(size=64, clock_ms=clock)
    surface = pygame.Surface((64, 64), pygame.SRCALPHA)
    rect = pygame.Rect(0, 0, 64, 64)

    vignette.trigger([_WRONG])  # duration 0.6s
    clock.now = 700
    vignette.render(surface, rect)

    assert pygame.surfarray.array_alpha(surface).max() == 0
    pygame.quit()


def test_render_mid_animation_is_edge_strong_center_faint_and_translucent():
    pygame.init()
    clock = _FakeClock()
    vignette = EdgeVignette(size=64, clock_ms=clock)
    surface = pygame.Surface((64, 64), pygame.SRCALPHA)
    rect = pygame.Rect(0, 0, 64, 64)

    vignette.trigger([_WRONG])  # peak_alpha 165, duration 0.6s, fade envelope
    clock.now = 300  # well past the short attack, before decay finishes
    vignette.render(surface, rect)

    alpha = pygame.surfarray.array_alpha(surface)
    corner_alpha = alpha[0, 0]
    center_alpha = alpha[32, 32]

    assert corner_alpha > center_alpha
    assert 0 < center_alpha
    assert center_alpha < corner_alpha * 0.15  # faint (~center_strength), well below peak
    assert 0 < corner_alpha < 255
    pygame.quit()


def test_new_trigger_restarts_animation_even_mid_flight():
    pygame.init()
    clock = _FakeClock()
    vignette = EdgeVignette(size=64, clock_ms=clock)
    surface = pygame.Surface((64, 64), pygame.SRCALPHA)
    rect = pygame.Rect(0, 0, 64, 64)

    vignette.trigger([_WRONG])
    clock.now = 500  # almost done fading out
    vignette.trigger([_DEAD_PICKED])  # a fresh, higher-priority event arrives
    clock.now += 50  # a little way into the new animation's attack ramp
    vignette.render(surface, rect)

    alpha = pygame.surfarray.array_alpha(surface)
    assert alpha[0, 0] > 0  # freshly restarted, not near-zero from the old one
    pygame.quit()


def test_edge_vignette_center_strength_param_changes_rendered_center_alpha():
    pygame.init()
    surface_low = pygame.Surface((64, 64), pygame.SRCALPHA)
    surface_high = pygame.Surface((64, 64), pygame.SRCALPHA)
    rect = pygame.Rect(0, 0, 64, 64)
    clock_low, clock_high = _FakeClock(), _FakeClock()

    low = EdgeVignette(size=64, clock_ms=clock_low, center_strength=0.05)
    high = EdgeVignette(size=64, clock_ms=clock_high, center_strength=0.08)
    low.trigger([_WRONG])
    high.trigger([_WRONG])
    clock_low.now = clock_high.now = 300
    low.render(surface_low, rect)
    high.render(surface_high, rect)

    center_low = pygame.surfarray.array_alpha(surface_low)[32, 32]
    center_high = pygame.surfarray.array_alpha(surface_high)[32, 32]
    assert center_low < center_high
    pygame.quit()


# --- EdgeVignette: styles merge/override/priority ---------------------------


def test_styles_override_replaces_an_existing_events_color():
    pygame.init()
    clock = _FakeClock()
    custom = {"victim_rescued": VignetteStyle((0, 0, 255), 140, 0.4, "fade", priority=3)}
    vignette = EdgeVignette(size=64, clock_ms=clock, styles=custom)
    surface = pygame.Surface((64, 64), pygame.SRCALPHA)
    rect = pygame.Rect(0, 0, 64, 64)

    vignette.trigger([_RESCUED])
    clock.now = 300
    vignette.render(surface, rect)

    rgb = pygame.surfarray.array3d(surface)
    assert tuple(rgb[0, 32]) == (0, 0, 255)
    pygame.quit()


def test_styles_can_add_a_wholly_new_event_type():
    pygame.init()
    clock = _FakeClock()
    custom = {"combo_bonus": VignetteStyle((255, 215, 0), 120, 0.3, "fade", priority=5)}
    vignette = EdgeVignette(size=64, clock_ms=clock, styles=custom)
    surface = pygame.Surface((64, 64), pygame.SRCALPHA)
    rect = pygame.Rect(0, 0, 64, 64)

    vignette.trigger([{"type": "combo_bonus"}])
    clock.now = 150
    vignette.render(surface, rect)

    assert pygame.surfarray.array_alpha(surface).max() > 0
    pygame.quit()


def test_new_event_type_can_control_its_own_priority():
    """A researcher-added event can outrank an existing default (here,
    dead_victim_picked, priority 0 — the highest priority among the
    defaults) by giving it a strictly lower priority number — proving
    priority is genuinely extensible, not just appearance. Priority -1 (not
    0) so this is a real outranking, not a coincidental positional tie-break
    win — dead_victim_picked is deliberately placed first in the trigger
    list, where a tie would favor it."""
    pygame.init()
    clock = _FakeClock()
    custom = {"critical_alert": VignetteStyle((255, 255, 0), 200, 0.5, "fade", priority=-1)}
    vignette = EdgeVignette(size=64, clock_ms=clock, styles=custom)
    surface = pygame.Surface((64, 64), pygame.SRCALPHA)
    rect = pygame.Rect(0, 0, 64, 64)

    vignette.trigger([_DEAD_PICKED, {"type": "critical_alert"}])
    clock.now = 300
    vignette.render(surface, rect)

    rgb = pygame.surfarray.array3d(surface)
    assert tuple(rgb[0, 32]) == (255, 255, 0)  # critical_alert's color won, not dead_victim_picked's
    pygame.quit()


def test_unknown_event_type_with_no_resolvable_style_is_silently_ignored():
    pygame.init()
    clock = _FakeClock()
    vignette = EdgeVignette(size=64, clock_ms=clock)  # no override at all
    surface = pygame.Surface((64, 64), pygame.SRCALPHA)
    rect = pygame.Rect(0, 0, 64, 64)

    vignette.trigger([{"type": "totally_unrecognized_event"}])  # must not raise
    clock.now = 100
    vignette.render(surface, rect)

    assert pygame.surfarray.array_alpha(surface).max() == 0
    pygame.quit()


def test_default_styles_merge_leaves_unlisted_events_unchanged():
    vignette = EdgeVignette(
        size=64,
        clock_ms=_FakeClock(),
        styles={"victim_rescued": VignetteStyle((0, 0, 255), 140, 0.4, "fade", priority=3)},
    )
    assert vignette._styles["mission_complete"] == DEFAULT_VIGNETTE_STYLES["mission_complete"]
    assert vignette._styles["victim_rescued"].color == (0, 0, 255)
