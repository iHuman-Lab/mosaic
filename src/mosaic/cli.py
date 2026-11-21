import argparse

from mosaic import __version__


def _build_env(args):
    from mosaic.sar.env import build_sar_env
    from mosaic.sar.placers import LavaPlacer, LockedRoomPlacer, VictimPlacer

    return build_sar_env(
        screen_size=args.screen_size,
        num_rows=args.rows,
        num_cols=args.cols,
        room_size=args.room_size,
        victim_placer=VictimPlacer(num_real_victims=args.victims),
        lava_placer=LavaPlacer(lava_per_room=args.lava),
        locked_room_placer=LockedRoomPlacer(locked_room_prob=0.5),
    )


def _play(args):
    from mosaic.gui.main import SAREnvGUI
    from mosaic.llm.client import DummyLLMClient

    env = _build_env(args)
    env.reset(seed=args.seed)
    SAREnvGUI(env, config={"fullscreen": args.fullscreen}, llm_client=DummyLLMClient()).run()


def _demo(args):
    env = _build_env(args)
    env.reset(seed=args.seed)
    total = 0.0
    for step in range(args.steps):
        _, reward, terminated, truncated, _ = env.step(env.action_space.sample())
        total += reward
        if terminated or truncated:
            break
    print(f"Ran {step + 1} steps with a random agent, total reward {total:.2f}.")
    print("MOSAIC is installed and working. Try `mosaic play` to play it yourself.")


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="mosaic", description="MOSAIC human-AI search and rescue platform"
    )
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command")

    for name, help_ in (
        ("play", "play a search-and-rescue mission in the GUI"),
        ("demo", "headless random-agent episode (install check)"),
    ):
        p = sub.add_parser(name, help=help_)
        p.add_argument("--rows", type=int, default=3)
        p.add_argument("--cols", type=int, default=3)
        p.add_argument("--room-size", type=int, default=8)
        p.add_argument("--victims", type=int, default=3)
        p.add_argument("--lava", type=int, default=2, help="lava tiles per room")
        p.add_argument("--screen-size", type=int, default=800)
        p.add_argument("--seed", type=int, default=None)
        if name == "play":
            p.add_argument("--fullscreen", action="store_true")
        else:
            p.add_argument("--steps", type=int, default=200)

    args = parser.parse_args(argv)
    if args.command == "play":
        _play(args)
    elif args.command == "demo":
        _demo(args)
    else:
        parser.print_help()
