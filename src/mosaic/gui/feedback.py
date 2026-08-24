import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pygame


@dataclass(frozen=True)
class VignetteStyle:
    """One event type's vignette appearance. Frozen because instances are
    shared (by reference) across every EdgeVignette via DEFAULT_VIGNETTE_STYLES
    — mutating one in place would corrupt it for every instance. Use
    dataclasses.replace() for a one-off tweak of an existing style."""

    color: Tuple[int, int, int]
    peak_alpha: int
    duration: float  # seconds
    envelope: str  # key into _ENVELOPES ("fade" or "pulse")
    priority: int = 99  # lower = higher priority when events tie


DEFAULT_VIGNETTE_STYLES: Dict[str, VignetteStyle] = {
    "victim_rescued": VignetteStyle((50, 205, 50), 140, 0.4, "fade", priority=3),
    "mission_complete": VignetteStyle((50, 205, 50), 150, 1.2, "fade", priority=2),
    "wrong_victim": VignetteStyle((220, 20, 60), 165, 0.6, "fade", priority=1),
    "dead_victim_picked": VignetteStyle((139, 0, 0), 185, 0.9, "fade", priority=0),
    "victim_died": VignetteStyle((139, 0, 0), 185, 0.9, "pulse", priority=0),
}

_FADE_ATTACK_FRAC = 0.125


def select_event(events: list, styles: Dict[str, VignetteStyle]) -> Optional[dict]:
    """Pick the single highest-priority event to render — priority comes
    from each event's resolved VignetteStyle, so an event with no
    resolvable style (absent from both defaults and overrides) is filtered
    out entirely, not just deprioritized. Returns None if no event in the
    list has a resolvable style. Ties are broken positionally: min() returns
    the first minimal element, so whichever supported event appears earlier
    in the list wins."""
    supported = [e for e in events if e.get("type") in styles]
    if not supported:
        return None
    return min(supported, key=lambda e: styles[e["type"]].priority)


def _fade_envelope(progress: float) -> float:
    if progress < _FADE_ATTACK_FRAC:
        return progress / _FADE_ATTACK_FRAC
    return max(0.0, 1.0 - (progress - _FADE_ATTACK_FRAC) / (1.0 - _FADE_ATTACK_FRAC))


def _pulse_envelope(progress: float) -> float:
    return math.sin(math.pi * progress)


_ENVELOPES = {"fade": _fade_envelope, "pulse": _pulse_envelope}


def _build_gradient_mask(size: int, falloff_fraction: float, center_strength: float) -> np.ndarray:
    """(size, size) weights in [0, 1]. 1.0 on any boundary pixel (edge or
    corner alike), ramping down over falloff_fraction * size pixels inward,
    flattening to center_strength beyond that. falloff_fraction/center_strength
    are fractions of size and of peak intensity respectively, not fixed
    pixel/alpha values, since size varies from small test viewports to
    full-display real trials."""
    falloff_px = max(1.0, falloff_fraction * size)
    idx = np.arange(size)
    dist_to_edge_1d = np.minimum(idx, size - 1 - idx).astype(np.float32)
    dist = np.minimum.outer(dist_to_edge_1d, dist_to_edge_1d)  # nearest of all 4 edges
    falloff = np.clip(1.0 - dist / falloff_px, 0.0, 1.0)
    return center_strength + (1.0 - center_strength) * falloff


def _tint(mask: np.ndarray, color: tuple, peak_alpha: int) -> pygame.Surface:
    size = mask.shape[0]
    surface = pygame.Surface((size, size), pygame.SRCALPHA)
    pygame.surfarray.pixels3d(surface)[:, :] = color
    pygame.surfarray.pixels_alpha(surface)[:, :] = (mask * peak_alpha).astype(np.uint8)
    return surface


class EdgeVignette:
    """A brief, translucent perimeter-vignette flash over the square game
    viewport, color/duration/shape keyed off SAR event type. The gradient
    mask and per-color tinted surfaces are built once and reused every
    frame — only a cheap set_alpha() + blit happens per render() call."""

    def __init__(
        self,
        size: int,
        clock_ms=None,
        falloff_fraction: float = 0.225,
        center_strength: float = 0.065,
        styles: Optional[Dict[str, VignetteStyle]] = None,
    ):
        if size <= 0:
            raise ValueError("size must be positive")
        if not 0.0 < falloff_fraction <= 0.5:
            raise ValueError("falloff_fraction must be in (0, 0.5]")
        if not 0.0 <= center_strength <= 1.0:
            raise ValueError("center_strength must be in [0, 1]")

        self._styles = {**DEFAULT_VIGNETTE_STYLES, **(styles or {})}
        for event_type, style in self._styles.items():
            if style.envelope not in _ENVELOPES:
                raise ValueError(
                    f"Unknown envelope {style.envelope!r} for event {event_type!r}; "
                    f"expected one of {sorted(_ENVELOPES)}"
                )

        self._clock_ms = pygame.time.get_ticks if clock_ms is None else clock_ms
        self._mask = _build_gradient_mask(size, falloff_fraction, center_strength)
        self._tint_cache: dict = {}
        self._active = None
        self._start_ms = 0
        self._duration_ms = 0

    def _tinted_surface(self, style: VignetteStyle) -> pygame.Surface:
        key = (style.color, style.peak_alpha)
        if key not in self._tint_cache:
            self._tint_cache[key] = _tint(self._mask, *key)
        return self._tint_cache[key]

    def trigger(self, events: list) -> None:
        event = select_event(events, self._styles)
        if event is None:
            return
        style = self._styles[event["type"]]
        self._active = style
        self._start_ms = self._clock_ms()
        self._duration_ms = int(style.duration * 1000)

    def render(self, surface: pygame.Surface, viewport_rect: pygame.Rect) -> None:
        if self._active is None:
            return
        elapsed = self._clock_ms() - self._start_ms
        if elapsed >= self._duration_ms:
            self._active = None
            return
        progress = elapsed / self._duration_ms
        envelope = _ENVELOPES[self._active.envelope]
        value = max(0.0, min(1.0, envelope(progress)))
        tinted = self._tinted_surface(self._active)
        tinted.set_alpha(int(255 * value))
        surface.blit(tinted, viewport_rect.topleft)
