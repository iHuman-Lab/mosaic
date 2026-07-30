import pygame
from pygame_gui.elements import UIPanel, UITextBox

from .utils import scale

_BG_NORMAL = pygame.Color("#1E1E28")
_BG_HIGHLIGHT = pygame.Color("#E7E7EE")  # white tint — new transmission
_HIGHLIGHT_DURATION_MS = 3000
_BLINK_INTERVAL_MS = 400


class ChatPanel:
    """Chat panel using pygame_gui built-in elements."""

    def __init__(self, manager, x_position, y_position, panel_width, panel_height):
        self.manager = manager
        self.panel_width = panel_width
        self.panel_height = panel_height
        self._highlight_until: int = 0
        self.on_blink_end = None

        s = scale(panel_width, 10, 26)
        self._title_size = scale(s, 3, 14)
        self._body_size = max(10, s // 2)

        title_h = max(1, scale(panel_width, 10, 40))
        gap = max(1, scale(panel_width, 40, 10))

        self.panel = UIPanel(
            relative_rect=pygame.Rect(
                x_position, y_position, panel_width, panel_height
            ),
            manager=manager,
        )
        self.title = UITextBox(
            html_text=f"<font pixel_size='{self._title_size}'><b>CHAT</b></font>",
            relative_rect=pygame.Rect(0, gap, panel_width, title_h),
            wrap_to_height=True,
            manager=manager,
            container=self.panel,
            object_id="#title",
        )
        self.text_box = UITextBox(
            html_text="",
            relative_rect=pygame.Rect(
                0, 0, panel_width - 15, panel_height - title_h - gap * 2
            ),
            manager=manager,
            container=self.panel,
            anchors={"top": "top", "top_target": self.title},
        )

    def _set_bg(self, colour: pygame.Color):
        self.panel.background_colour = colour
        self.panel.rebuild()
        self.text_box.background_colour = colour
        self.text_box.rebuild()

    def set_message(self, sender, text, color="#6495ED"):
        self.text_box.set_text(
            f"<font pixel_size='{self._body_size}' color='{color}'><b>{sender}:</b> {text}</font>"
        )

    def reset(self):
        self._highlight_until = 0
        self._set_bg(_BG_NORMAL)
        self.clear_message()

    def clear_message(self):
        self.text_box.set_text("")

    def _update_blink(self, now: int):
        if not self._highlight_until:
            return
        if now >= self._highlight_until:
            self._highlight_until = 0
            self._set_bg(_BG_NORMAL)
            if self.on_blink_end:
                self.on_blink_end()
        else:
            blink_on = (now // _BLINK_INTERVAL_MS) % 2 == 0
            self._set_bg(_BG_HIGHLIGHT if blink_on else _BG_NORMAL)

    def _handle_llm_result(self, user, now: int):
        if user.provider != "dummy":
            user.env.show_all_victim_batteries()
        kind, value = user.llm_result
        if kind == "reply":
            self.set_message("Agent", value)
            self._highlight_until = now + _HIGHLIGHT_DURATION_MS
        else:
            self.set_message("Error", value, color="#DC143C")
        user.llm_thread = None
        user.llm_result = None

    def poll_llm(self, user):
        now = pygame.time.get_ticks()
        self._update_blink(now)
        if user.llm_thread is None:
            return
        if user.llm_thread.is_alive():
            dots = "." * (now // 400 % 3 + 1)
            self.set_message("Agent", f"thinking{dots}", color="#888888")
        elif user.llm_result is not None:
            self._handle_llm_result(user, now)
