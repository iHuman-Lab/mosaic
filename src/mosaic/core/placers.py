from abc import ABC, abstractmethod


class Placer(ABC):
    """Places objects into a generated level.

    Implementations mutate the level in place and return nothing, either by
    writing to ``level_gen.grid`` directly or through the level generator's
    helpers. Called from the environment's ``gen_mission()``.
    """

    def reset(self, level_gen) -> None:
        """Optional per-episode reset hook. No-op by default."""
        pass

    @abstractmethod
    def place_all(self, level_gen, num_rows: int, num_cols: int, **kwargs) -> None:
        """Place this components's objects into the grid."""
        pass
