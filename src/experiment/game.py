from __future__ import annotations

import os
from typing import Any

import pygame
import ujson
from ixp.task import Block, LSLTrial, Task

from game.gui.main import SAREnvGUI
from game.sar.env import build_sar_env


class SARGameTrial(LSLTrial):
    """LSL-streaming trial for the SAR rescue game.

    Streams game state as a single JSON channel at each rendered frame.
    Captures enough data to fully replay or reconstruct the session:
    env config, per-step action/reward, agent state, and full object states.

    Block.execute() calls initialize() → execute() → clean_up(), so the
    full lifecycle is managed externally by the Block.

    Parameters
    ----------
    parameters : dict
        May include ``prompt_type`` ("sparse" | "detailed"),
        ``model`` (e.g. "gpt-4o-mini"), and ``provider`` ("openai" | "google" | "dummy").
    """

    def __init__(self, trial_id: str, parameters: dict[str, Any]):
        super().__init__(trial_id, parameters)
        self.gui = None

    def initialize(self) -> None:
        config = self.parameters
        screen_height = pygame.display.Info().current_h
        env = build_sar_env(
            screen_size=screen_height,
            num_rows=config.get("num_rows"),
            num_cols=config.get("num_cols"),
        )
        os.environ["SDL_VIDEO_FULLSCREEN_DISPLAY"] = str(config.get("display", 0))
        self.gui = SAREnvGUI(env, config=config)
        self.gui.info_panel.set_trial_name(self.trial_id)
        self.gui.reset()
        self.gui.running = True
        self.gui.user.total_steps = 0
        self.gui.user.episode_ended = False

    def clean_up(self) -> None:
        self.gui = None

    def read_data(self) -> list[str] | None:
        user = self.gui.user
        obs = user.obs
        state = {
            **(obs if isinstance(obs, dict) else {}),
            "action": user.last_action,
            "reward": user.last_reward,
            "terminated": user.terminated,
            "truncated": user.truncated,
            "prompt_type": user.prompt_type,
            "llm_model": user.model,
            "llm_provider": user.provider,
            "llm_response": user.last_llm_response,
            "total_steps": user.total_steps,
            "trial_id": self.trial_id,
        }
        return [ujson.dumps(state)]

    def execute(self) -> None:
        min_steps= self.parameters.get("min_steps", 500)

        while self.gui.running:
            user = self.gui.user
            enough = user.total_steps >= int(min_steps)

            if enough and user.episode_ended:
                self.gui.close()
                break

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    if enough:
                        self.gui.close()
                    break
                self.gui.manager.process_events(event)
                self.gui.handle_user_input(event)

            if self.gui.running:
                self.gui.render(self.gui.user.get_frame())
                self.stream()


class SARGame(Task):
    """SAR game task for the ixp experiment framework.

    One block with 4 trials (sparse×openai, detailed×openai, sparse×gemini,
    detailed×gemini). Trial order is randomized at execution time.
    """

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        openai_model = config.get("openai_model", "gpt-4o-mini")
        google_model = config.get("google_model", "gemini-1.5-flash")
        block = Block("sar_game_block")
        block.add_trial(
            SARGameTrial(
                "trial_detailed_openai",
                {
                    **config,
                    "prompt_type": "detailed",
                    "provider": "openai",
                    "model": openai_model,
                },
            ),
            order=1,
        )
        block.add_trial(
            SARGameTrial(
                "trial_dummy",
                {
                    **config,
                    "prompt_type": "sparse",
                    "provider": "dummy",
                    "model": google_model,
                },
            ),
            order=2,
        )
        block.add_trial(
            SARGameTrial(
                "trial_detailed_gemini",
                {
                    **config,
                    "prompt_type": "detailed",
                    "provider": "google",
                    "model": google_model,
                },
            ),
            order=3,
        )
        self.add_block(block)

    def get_data_signature(self) -> dict[str, Any]:
        return {
            "name": "SARGame",
            "type": "GameState",
            "channel_count": 1,
            "nominal_srate": 0.0,
            "channel_format": "string",
            "source_id": "rescue-grid-sar-game",
        }

    def execute(self, order: str = "random") -> list:
        super().execute(order)
        pygame.quit()
        return []
