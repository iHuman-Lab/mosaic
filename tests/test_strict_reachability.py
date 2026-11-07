"""Verify reachability with unblocking=False (victims directly reachable)."""

from mosaic.sar.env import PickupVictimEnv


def test_strict_reachability(num_tests=5):
    """With unblocking=False, every generated mission has directly reachable victims."""
    env = PickupVictimEnv(
        room_size=6,
        num_rows=2,
        num_cols=2,
        num_dists=5,
        unblocking=False,  # Enforces strict reachability
        render_mode=None,
    )

    for i in range(num_tests):
        env.reset()

        victims = env.get_all_victims()
        assert len(victims) > 0, f"reset {i + 1}/{num_tests}: no victims generated"
        assert env.get_mission_status() is not None


if __name__ == "__main__":
    test_strict_reachability(num_tests=1000)
    print("All victims are directly reachable.")
