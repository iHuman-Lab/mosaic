def scale(panel_width: int, divisor: int, minimum: int) -> int:
    """Scale a value proportionally to panel_width, clamped to minimum."""
    return max(minimum, panel_width // divisor)
