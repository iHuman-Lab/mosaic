"""Verify that every generated mission passes the victim reachability check."""

from mosaic.sar.env import PickupVictimEnv


def test_victim_reachability(num_tests=10):
    """Environment generation (which enforces reachability) succeeds on every reset."""
    env = PickupVictimEnv(
        room_size=6, num_rows=3, num_cols=3, num_dists=10, render_mode=None
    )

    for i in range(num_tests):
        # reset() triggers gen_mission(); it raises if an object is unreachable.
        env.reset()

        victims = env.get_all_victims()
        assert len(victims) > 0, f"reset {i + 1}/{num_tests}: no victims generated"
        assert env.get_mission_status() is not None


if __name__ == "__main__":
    test_victim_reachability(num_tests=10)
    print("All victims are reachable.")
