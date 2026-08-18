from functools import partial

import pygame
import yaml
from dotenv import load_dotenv

from mosaic.core.camera import AgentConeCamera
from mosaic.gui.main import SAREnvGUI
from mosaic.llm.client import DummyLLMClient
from mosaic.sar.env import build_sar_env
from mosaic.sar.placers import LockedRoomPlacer

try:
    from mosaic.tutorial_env import TutorialEnv
except ImportError:
    pass
from .llm import build_llm_client
from .utils import skip_run

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
        locked_room_placer_cls=partial(LockedRoomPlacer, locked_room_prob=0.5),
        camera_strategy=AgentConeCamera(),
    )
    env.reset()
    llm_client = build_llm_client(
        provider=config.get("provider", "openai"), model=config.get("model")
    )
    gui = SAREnvGUI(env, config=config, llm_client=llm_client)
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
    gui = SAREnvGUI(env, config={"fullscreen": True}, llm_client=DummyLLMClient())
    gui.run()
