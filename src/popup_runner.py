import argparse
import pygame
import yaml
import os

from game.tutorial_env import TutorialEnv
from game.gui.main import SAREnvGUI

parser = argparse.ArgumentParser(description="Run a single-room popup tutorial GUI")
parser.add_argument("--room_size", type=int, default=8)
parser.add_argument("--screen_size", type=int, default=800)
parser.add_argument("--tile_size", type=int, default=32)
parser.add_argument("--render_mode", type=str, default="human")

if __name__ == "__main__":
    args = parser.parse_args()

    # Ensure pygame is initialized in this subprocess
    pygame.init()

    env = TutorialEnv(
        start_part=1,
        total_parts=1,
        room_size=args.room_size,
        num_rows=1,
        num_cols=1,
        window=None,
        screen_size=args.screen_size,
        render_mode=args.render_mode,
        agent_pov=True,
        tile_size=args.tile_size,
    )

    env.reset()
    gui = SAREnvGUI(env, fullscreen=False)
    gui.run()
