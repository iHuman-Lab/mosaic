import numpy as np
import pygame
import pygame_gui

from .chat import ChatPanel
from .info import InfoPanel
from .user import User

pygame.init()


class SAREnvGUI:
    def __init__(self, env, config: dict | None = None):
        config = config or {}
        self.user = User(
            env,
            prompt_type=config.get("prompt_type", "detailed"),
            model=config.get("model", "gpt-4o-mini"),
            provider=config.get("provider", "openai"),
            max_time=config.get("max_time", 5.0),
        )
        self.llm_nudge_interval = config.get("llm_nudge_interval", 50)
        self.fullscreen = config.get("fullscreen", False)
        self.game_size = self.user.env.screen_size

        self.panel_width = self.game_size // 2
        self.window_size = (self.game_size + self.panel_width, self.game_size)

        self.window = env.window

        if self.window is None:
            self._setup_display(display=config.get("display", 0))

        self._calculate_offsets()

        self.running = True
        self.clock = pygame.time.Clock()

        self._create_panels()

    def _setup_display(self, display=0):
        """Set up the pygame display window based on fullscreen state."""
        if self.fullscreen:
            self.window = pygame.display.set_mode(
                (0, 0), pygame.FULLSCREEN, display=display
            )
            self.screen_size = self.window.get_size()
        else:
            self.window = pygame.display.set_mode(self.window_size)
            self.screen_size = self.window_size

    def _create_panels(self):
        """Create (or recreate) the UI manager and side panels."""
        self.manager = pygame_gui.UIManager(self.window_size, "src/game/gui/theme.json")
        chat_height = self.game_size // 2
        info_height = self.game_size - chat_height
        self.info_panel = InfoPanel(
            self.manager, self.game_size, self.panel_width, info_height
        )
        self.chat_panel = ChatPanel(
            self.manager,
            self.game_size,
            info_height,
            self.panel_width,
            chat_height,
        )
        self.chat_panel.on_blink_end = self.user.env.hide_all_victim_batteries

    def _calculate_offsets(self):
        """Calculate scale and offsets to center game content on screen."""
        scale_x = self.screen_size[0] / self.window_size[0]
        scale_y = self.screen_size[1] / self.window_size[1]
        self.scale = min(scale_x, scale_y)
        self.scaled_width = int(self.window_size[0] * self.scale)
        self.scaled_height = int(self.window_size[1] * self.scale)
        self.offset_x = (self.screen_size[0] - self.scaled_width) // 2
        self.offset_y = (self.screen_size[1] - self.scaled_height) // 2

    def render(self, frame):
        self.chat_panel.poll_llm(self.user)
        self.window.fill((0, 0, 0))
        combined_surface = self._build_combined_surface(frame)
        self.window.blit(combined_surface, (self.offset_x, self.offset_y))
        pygame.display.update()

    def _build_combined_surface(self, frame):
        surface = pygame.Surface(self.window_size, pygame.SRCALPHA)
        frame = np.transpose(frame, (1, 0, 2))
        game_surface = pygame.surfarray.make_surface(frame)
        game_surface = pygame.transform.smoothscale(
            game_surface, (self.game_size, self.game_size)
        )
        surface.blit(game_surface, (0, 0))
        self.info_panel.render(self.user.obs or {}, self.user.env, self.user)
        time_delta = self.clock.tick(30) / 1000.0
        self.manager.update(time_delta)
        self.manager.draw_ui(surface)
        if self.scale != 1.0:
            surface = pygame.transform.smoothscale(
                surface, (self.scaled_width, self.scaled_height)
            )
        return surface

    def _call_llm(self):
        self.user.ask_llm_async()

    def reset(self):
        self.user.reset()
        self.chat_panel.reset()

    def handle_user_input(self, event):
        if event.type != pygame.KEYDOWN:
            return
        if self.user.llm_thread and self.user.llm_thread.is_alive():
            return
        if event.key == pygame.K_ESCAPE:
            self.close()
        elif event.key == pygame.K_F11:
            self.toggle_fullscreen()
        elif event.key in (pygame.K_LALT, pygame.K_RALT):
            self._call_llm()
        else:
            event.key = pygame.key.name(int(event.key))
            self.user.handle_key(event)
            if (
                self.user.steps_since_last_llm > 0
                and self.user.steps_since_last_llm % self.llm_nudge_interval == 0
            ):
                self._call_llm()

    def toggle_fullscreen(self):
        """Toggle between fullscreen and windowed mode."""
        self.fullscreen = not self.fullscreen
        self._setup_display()
        self._calculate_offsets()
        self._create_panels()

    def close(self):
        self.running = False

    def run(self):
        self.reset()

        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.close()
                    break
                self.manager.process_events(event)
                self.handle_user_input(event)

            if self.running:
                self.render(self.user.get_frame())

        pygame.quit()
