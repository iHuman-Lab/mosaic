import pygame
import yaml

from game.gui.main import SAREnvGUI
from game.sar.env import PickupVictimEnv
from game.sar.utils import VictimPlacer
try:
    from game.tutorial_env import TutorialEnv
except ImportError:
    from game.tutorial_env import OptimizedSAREnv as TutorialEnv
from utils import skip_run

# Patch: ManualControl expects string key names, pygame sends int codes
from game.gui.user import User
_orig_handle_key = User.handle_key
def _patched_handle_key(self, event):
    if isinstance(event.key, int):
        event.key = pygame.key.name(event.key)
    print(f"[PATCH] key={event.key!r}", flush=True)
    return _orig_handle_key(self, event)
User.handle_key = _patched_handle_key

# Load config
config_path = "configs/config.yml"
with open(config_path, "r") as file:
    config = yaml.safe_load(file)


with skip_run("run", "sar_gui_advanced") as check, check():
    # Access the width and height of the current display
    screen_height = pygame.display.Info().current_h
    victim_placer = VictimPlacer(
        num_fake_victims=5, num_real_victims=3, important_victim="down"
    )
    # Toggle tutorial mode here
    TUTORIAL = False

    if TUTORIAL:
        env = TutorialEnv(config={"start_part": 1}, screen_size=800, render_mode="rgb_array", agent_pov=True, tile_size=64)
    else:
        env = PickupVictimEnv(
        num_rows=3,
        num_cols=3,
        screen_size=800,
        render_mode="rgb_array",
        agent_pov=True,
        add_lava=True,
        lava_per_room=2,
        locked_room_prob=0.5,
        # camera_strategy=FullviewCamera(),
        tile_size=64,
        victim_placer=victim_placer,
    )
    env.reset()
    gui = SAREnvGUI(env, fullscreen=False)
    gui.run()