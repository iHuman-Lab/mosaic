from __future__ import annotations

import os
from typing import Any

import pygame
from ixp.task import Task

from mosaic.gui.main import SAREnvGUI
from mosaic.tutorial_env import TutorialEnv


class SARTutorial(Task):
    """Tutorial task for the SAR rescue game.

    Runs a single-room TutorialEnv where the mission (open locked door or
    pick up object) is displayed in the chat panel and refreshes on every
    episode reset.
    """

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self.gui: SAREnvGUI | None = None

    def initialize(self) -> None:
        screen_height = pygame.display.Info().current_h
        env = TutorialEnv(
            num_rows=1,
            num_cols=1,
            screen_size=screen_height,
            render_mode="rgb_array",
            agent_pov=True,
        )
        os.environ["SDL_VIDEO_FULLSCREEN_DISPLAY"] = str(self.config.get("display", 0))
        self.gui = SAREnvGUI(env, config=self.config)

        def _show_mission():
            mission = getattr(self.gui.user.env, "mission", None)
            if mission:
                self.gui.chat_panel.set_message("Mission", mission)

        self.gui.user.on_reset = _show_mission
        self.gui.reset()

    def clean_up(self) -> None:
        self.gui = None
        pygame.quit()

    def execute(self, order: str = "predefined"):
        self.initialize()
        try:
            if self.gui is not None:
                self.gui.run()
            return []
        finally:
            self.clean_up()
