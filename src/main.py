import pygame
import yaml
from dotenv import load_dotenv

from mosaic.core.camera import AgentConeCamera
from mosaic.gui.main import SAREnvGUI
from mosaic.sar.env import build_sar_env

try:
    from mosaic.tutorial_env import TutorialEnv
except ImportError:
    pass
from utils import skip_run

load_dotenv()


# Load config
config_path = "configs/experiment.yaml"
with open(config_path, "r") as file:
    config = yaml.safe_load(file)


with skip_run("run", "sar_gui_advanced") as check, check():
    env = build_sar_env(
        screen_size=800,
        num_rows=3,
        num_cols=3,
        locked_room_prob=0.5,
        camera_strategy=AgentConeCamera(),
    )
    env.reset()
    gui = SAREnvGUI(env, config=config)
    gui.run()


with skip_run("skip", "tutorial") as check, check():
    # Access the width and height of the current display
    screen_height = pygame.display.Info().current_h

    env = TutorialEnv(
        num_rows=1,
        num_cols=1,
        screen_size=800,
        render_mode="rgb_array",
        agent_pov=True,
    )

    env.reset()
    gui = SAREnvGUI(env, config={"fullscreen": True})
    gui.run()
