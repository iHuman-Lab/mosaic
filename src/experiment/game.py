from __future__ import annotations

import os
from typing import Any

import pygame
import ujson
from mosaic.core.camera import AgentFOVCamera
from mosaic.gui.main import SAREnvGUI
from mosaic.sar.actions import RescueAction, RescueRewards
from mosaic.sar.env import build_sar_env
from ixp.task import Block, LSLTrial, Task

from .llm import build_llm_client
from .pacing import TunedPickupVictimEnv
from .placers import LavaRiskVictimPlacer, SectorSpreadLavaPlacer


def _show_break_screen(display: int = 0, recalibrate: bool = False) -> None:
    """Fullscreen PsychoPy break screen; waits for SPACE, then optionally recalibrates eye tracker."""

    from psychopy import event, visual

    win = visual.Window(
        fullscr=True, screen=display, color="black", units="norm", checkTiming=False
    )
    visual.TextStim(
        win,
        text="Great work! Take a short break.\n\nPress SPACE when you are ready to continue.",
        color="white",
        height=0.08,
        wrapWidth=1.6,
    ).draw()
    win.flip()
    event.waitKeys(keyList=["space"])
    win.close()

    if recalibrate:
        from ixp.sensors.eye_tracker.tobii import TobiiEyeTracker

        tobii = TobiiEyeTracker()
        tobii.initialize()
        tobii.calibrate(screen=display)


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
            room_size=config.get("room_size", 14),
            victim_placer=LavaRiskVictimPlacer(
                num_real_victims=config.get("num_real_victims", 6),
                num_fake_victims=config.get("num_fake_victims", 12),
            ),
            lava_placer=SectorSpreadLavaPlacer(
                lava_per_room=config.get("lava_per_room", 8),
            ),
            camera_strategy=AgentFOVCamera(),
            env_cls=TunedPickupVictimEnv,
            action=RescueAction(
                rewards=RescueRewards(
                    real_victim_alive=10, fake_victim=-10, real_victim_dead=-20
                )
            ),
            deplete_amount_fn=lambda max_steps: 17.5 / max_steps,
        )
        os.environ["SDL_VIDEO_FULLSCREEN_DISPLAY"] = str(config.get("display", 0))
        provider = config.get("provider", "openai")
        llm_client = build_llm_client(provider=provider, model=config.get("model"))
        self.gui = SAREnvGUI(env, config=config, llm_client=llm_client)
        if provider != "dummy":
            reveal = getattr(env, "show_all_victim_batteries", None)
            hide = getattr(env, "hide_all_victim_batteries", None)
            if reveal and hide:
                self.gui.set_advice_callbacks(on_reply=reveal, on_end=hide)
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
            "llm_model": self.parameters.get("model"),
            "llm_provider": self.parameters.get("provider", "dummy"),
            "llm_response": user.last_llm_response,
            "total_steps": user.total_steps,
            "trial_id": self.trial_id,
        }
        return [ujson.dumps(state)]

    def execute(self) -> None:
        min_steps_percent = self.parameters.get("min_steps_percent", 50) / 100
        max_steps = self.gui.user.env.max_steps

        while self.gui.running:
            user = self.gui.user

            # Hard stop: 15-minute wall-clock limit or max_steps truncation
            if user.remaining_time <= 0 or user.truncated:
                self.gui.close()
                break

            # Soft stop: mission complete, but only after enough steps played
            enough = user.total_steps >= int(min_steps_percent * max_steps)
            if enough and user.episode_ended:
                self.gui.close()
                break

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    if enough:
                        self.gui.close()
                    break
                self.gui.handle_user_input(event)
                self.gui.manager.process_events(event)

            if self.gui.running:
                self.gui.render(self.gui.user.get_frame())
                self.stream()


class SARGame(Task):
    """SAR game task for the ixp experiment framework.

    One block with 3 trials (baseline, detailed×openai,
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
                    "model": "dummy",
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
        display = self.config.get("display", 0)
        recalibrate = self.config.get("between_trail_recalibrate_eye_tracker", False)
        for block in self.blocks:
            block.execute(
                order,
                lsl_stream=self.lsl_stream,
                after_trial_fn=lambda: _show_break_screen(display, recalibrate),
            )
        pygame.quit()
        return []
