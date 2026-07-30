from pathlib import Path

import pygame
from pygame_gui.elements import UIImage, UIPanel, UITextBox

from .utils import scale

_COLOR_POSITIVE = "#32CD32"
_COLOR_NEGATIVE = "#DC143C"
_COLOR_NEUTRAL = "#DCDCDC"

_KEY_COLORS = {
    "red": "#FF0000",
    "green": "#00FF00",
    "blue": "#0064FF",
    "yellow": "#FFFF00",
    "purple": "#A020F0",
    "grey": "#808080",
}


def _reward_color(value: float) -> str:
    if value > 0:
        return _COLOR_POSITIVE
    if value < 0:
        return _COLOR_NEGATIVE
    return _COLOR_NEUTRAL


class InfoPanel:
    """Info panel — UIPanel + mission box + 3 metric columns + compass + controls."""

    def __init__(self, manager, x_offset, panel_width, panel_height=None):
        panel_height = panel_height if panel_height is not None else panel_width

        s = scale(panel_width, 10, 26)
        self._metric_size = s
        self._title_size = scale(s, 3, 14)
        self._label_size = max(10, s // 3)
        self._body_size = max(10, s // 2)

        PAD = 10
        W = panel_width - PAD * 2
        col_w = W // 3

        # 1 — Panel
        self.panel = UIPanel(
            relative_rect=pygame.Rect(x_offset, 0, panel_width, panel_height),
            manager=manager,
        )

        # 2 — Mission title
        self.title_box = UITextBox(
            html_text=f"<font pixel_size='{self._title_size}'><b>MISSION INFORMATION</b></font>",
            relative_rect=pygame.Rect(PAD, 4, W, -1),
            wrap_to_height=True,
            manager=manager,
            container=self.panel,
            object_id="#title",
        )

        # 3 — Mission info anchored below title
        self.info_box = UITextBox(
            html_text=self._mission_html(),
            relative_rect=pygame.Rect(PAD, 4, W, -1),
            wrap_to_height=True,
            manager=manager,
            container=self.panel,
            anchors={"top": "top", "top_target": self.title_box},
        )

        # 4 — Three metric columns anchored below info_box
        self.reward_box = UITextBox(
            html_text=self._metric_html("REWARD", "+0.0", _COLOR_NEUTRAL),
            relative_rect=pygame.Rect(PAD, 4, col_w, -1),
            wrap_to_height=True,
            manager=manager,
            container=self.panel,
            anchors={"top": "top", "top_target": self.info_box},
        )
        self.total_box = UITextBox(
            html_text=self._metric_html("TOTAL", "+0.0", _COLOR_NEUTRAL),
            relative_rect=pygame.Rect(PAD + col_w, 4, col_w, -1),
            wrap_to_height=True,
            manager=manager,
            container=self.panel,
            anchors={"top": "top", "top_target": self.info_box},
        )
        self.time_box = UITextBox(
            html_text=self._metric_html("TIME LEFT", "0:00", _COLOR_NEUTRAL),
            relative_rect=pygame.Rect(PAD + 2 * col_w, 4, W - 2 * col_w, -1),
            wrap_to_height=True,
            manager=manager,
            container=self.panel,
            anchors={"top": "top", "top_target": self.info_box},
        )

        # 5 — Compass left, controls right — anchored below reward_box
        compass_w = panel_width // 3
        controls_x = PAD + compass_w + 3
        UITextBox(
            html_text=(
                f"<font pixel_size='{self._body_size}'>"
                "Arrows &rarr; Move / Turn<br>"
                "Tab &rarr; Pick up<br>"
                "Shift &rarr; Drop Item<br>"
                "Space &rarr; Open door<br>"
                "Alt &rarr; Ask Guidance"
                "</font>"
            ),
            relative_rect=pygame.Rect(
                controls_x, 4, panel_width - controls_x - PAD, -1
            ),
            wrap_to_height=True,
            manager=manager,
            container=self.panel,
            anchors={"top": "top", "top_target": self.reward_box},
        )
        raw = pygame.image.load(
            str(Path(__file__).parent / "compass_inv.png")
        ).convert_alpha()
        self.compass_image = UIImage(
            relative_rect=pygame.Rect(PAD, 30, compass_w, compass_w),
            image_surface=pygame.transform.smoothscale(raw, (compass_w, compass_w)),
            manager=manager,
            container=self.panel,
            anchors={"top": "top", "top_target": self.reward_box},
        )

    def _metric_html(self, label: str, value: str, color: str) -> str:
        return (
            f"<font pixel_size='{self._label_size}'>{label}</font><br>"
            f"<font pixel_size='{self._metric_size}' color='{color}'>{value}</font>"
        )

    def _mission_html(
        self,
        rescued=0,
        remaining=0,
        steps=0,
        max_steps=0,
        inv_text="None",
        inv_color=_COLOR_NEUTRAL,
    ) -> str:
        rem_color = _COLOR_NEGATIVE if remaining > 0 else _COLOR_POSITIVE
        return (
            f"<font pixel_size='{self._body_size}'>"
            f"Rescued: <font color='#32CD32'>{rescued}</font>"
            f"&nbsp;|&nbsp;"
            f"Remaining: <font color='{rem_color}'>{remaining}</font><br>"
            f"Steps: {steps}/{max_steps}"
            f"&nbsp;|&nbsp;"
            f"Inventory: <font color='{inv_color}'>{inv_text}</font>"
            f"</font><hr>"
        )

    def render(self, obs, env, user=None):
        rescued = obs.get("saved_victims", 0)
        remaining = obs.get("remaining_victims", 0)
        steps = getattr(env, "step_count", 0)
        max_steps = getattr(env, "max_steps", 0)
        carrying = getattr(env, "carrying", None)

        if carrying:
            inv_text = f"{carrying.color.capitalize()} {carrying.type.capitalize()}"
            inv_color = _KEY_COLORS.get(carrying.color.lower(), _COLOR_NEUTRAL)
        else:
            inv_text, inv_color = "None", _COLOR_NEUTRAL

        reward = getattr(user, "last_reward", 0.0) if user else 0.0
        total_reward = getattr(user, "total_reward", 0.0) if user else 0.0
        time_left = getattr(user, "remaining_time", 0.0) if user else 0.0

        mins = int(time_left) // 60
        secs = int(time_left) % 60
        time_color = _COLOR_POSITIVE if time_left > 60 else _COLOR_NEGATIVE

        self.info_box.set_text(
            self._mission_html(
                rescued=rescued,
                remaining=remaining,
                steps=steps,
                max_steps=max_steps,
                inv_text=inv_text,
                inv_color=inv_color,
            )
        )
        self.reward_box.set_text(
            self._metric_html("REWARD", f"{reward:+.1f}", _reward_color(reward))
        )
        self.total_box.set_text(
            self._metric_html(
                "TOTAL REWARD", f"{total_reward:+.1f}", _reward_color(total_reward)
            )
        )
        self.time_box.set_text(
            self._metric_html("TIME LEFT", f"{mins}:{secs:02d}", time_color)
        )
