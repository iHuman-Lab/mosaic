"""Play any pygame program live in a Jupyter notebook.

``play_live(run)`` runs your normal pygame loop in a thread. Pygame draws off-screen (SDL's dummy driver), the picture is shown
in the cell, and the keys and the mouse you use in the browser reach the program as ordinary pygame events: ``KEYDOWN`` / ``KEYUP``,
``MOUSEBUTTONDOWN`` / ``MOUSEBUTTONUP`` / ``MOUSEMOTION`` and ``MOUSEWHEEL``. ``pygame.mouse.get_pos``, ``pygame.mouse.get_pressed`` and
``pygame.key.get_mods`` report the browser's state while a session runs.
Needs ipywidgets and ipyevents. Keep this file in the same folder as the notebook.

    def game():                                   # any pygame program that opens a window and loops
        screen = pygame.display.set_mode((400, 300))
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
            ...
            pygame.display.flip()

    play_live(game)
"""
import io
import os
import threading
import time
import traceback
from types import SimpleNamespace

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")      # must be set before pygame is imported
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import numpy as np
import pygame
import ipywidgets as widgets
from ipyevents import Event
from IPython.display import display
from PIL import Image

# browser key name (lower case) -> pygame key. A single character maps to itself: "a" is pygame.K_a.
NAMED_KEYS = {"arrowup": pygame.K_UP, "arrowdown": pygame.K_DOWN, "arrowleft": pygame.K_LEFT, "arrowright": pygame.K_RIGHT,
              " ": pygame.K_SPACE, "enter": pygame.K_RETURN, "escape": pygame.K_ESCAPE, "backspace": pygame.K_BACKSPACE,
              "shift": pygame.K_LSHIFT, "control": pygame.K_LCTRL, "tab": pygame.K_TAB, "alt": pygame.K_LALT}

# Tab and Alt work as well, but some browsers keep them for themselves, so E and Q do the same: MOSAIC's "pick up / rescue" and "ask the teammate".
MOSAIC_KEYS = {"e": pygame.K_TAB, "q": pygame.K_LALT}

_ORIGINAL_FLIP, _ORIGINAL_UPDATE = pygame.display.flip, pygame.display.update      # to put back when a session ends
_ORIGINAL_MOUSE = (pygame.mouse.get_pos, pygame.mouse.get_pressed, pygame.key.get_mods)
_BUTTONS = {0: 1, 1: 2, 2: 3}                           # browser button (left, middle, right) -> pygame button
_session = None                                         # the running session, so that a new one stops it


def _pygame_key(name, keys):
    name = name.lower()
    if name in keys:
        return keys[name]
    if name in NAMED_KEYS:
        return NAMED_KEYS[name]
    return ord(name) if len(name) == 1 else None


class _Session:
    def __init__(self, run, keys, fps, quality, width, summary, on_stop):
        self.keys = {**(keys or {})}
        self.fps, self.quality, self.width, self.summary, self.on_stop = fps, quality, width, summary, on_stop
        self.stopped = threading.Event()
        self._finish_lock = threading.Lock()
        self._finished = False
        self.screen = widgets.Image(format="jpeg")
        self.out = widgets.Output()
        self.game = threading.Thread(target=self._run, args=(run,), daemon=True)
        self._frame = None                                  # the last finished frame, taken when the program calls flip / update
        self._frame_time = 0.0
        self.mouse_pos, self.mouse_buttons, self.mods = (0, 0), [False, False, False], 0

    def _snapshot(self):
        """Copy the window right after the program has finished drawing it, so a half-drawn frame is never shown."""
        now = time.time()
        surface = pygame.display.get_surface()
        if surface is not None and now - self._frame_time >= 0.5 / self.fps:
            self._frame = np.asarray(pygame.surfarray.array3d(surface)).swapaxes(0, 1)
            self._frame_time = now

    def _hook_display(self):
        def flip(*args, **kwargs):
            result = _ORIGINAL_FLIP(*args, **kwargs)
            self._snapshot()
            return result

        def update(*args, **kwargs):
            result = _ORIGINAL_UPDATE(*args, **kwargs)
            self._snapshot()
            return result
        def get_pos():
            return self.mouse_pos

        def get_pressed(num_buttons=3, *args, **kwargs):
            return tuple(self.mouse_buttons[:num_buttons]) + (False,) * max(0, num_buttons - 3)

        def get_mods():
            return self.mods
        pygame.display.flip, pygame.display.update = flip, update
        pygame.mouse.get_pos, pygame.mouse.get_pressed, pygame.key.get_mods = get_pos, get_pressed, get_mods

    def _unhook_display(self):
        pygame.display.flip, pygame.display.update = _ORIGINAL_FLIP, _ORIGINAL_UPDATE
        pygame.mouse.get_pos, pygame.mouse.get_pressed, pygame.key.get_mods = _ORIGINAL_MOUSE

    def _run(self, run):
        try:
            run()
        except Exception:
            self.out.append_stderr(traceback.format_exc())
        finally:
            self.stopped.set()
            self._finish()                                  # the program ended by itself, for example the clock ran out

    def _update_mods(self, event):
        self.mods = ((pygame.KMOD_LSHIFT if event.get("shiftKey") else 0) | (pygame.KMOD_LCTRL if event.get("ctrlKey") else 0)
                     | (pygame.KMOD_LALT if event.get("altKey") else 0) | (pygame.KMOD_LMETA if event.get("metaKey") else 0))

    def on_key(self, event):
        self._update_mods(event)
        name = event.get("key", "")
        key = _pygame_key(name, self.keys)
        if key is None:
            return
        kind = pygame.KEYDOWN if event.get("event") == "keydown" else pygame.KEYUP
        pygame.event.post(pygame.event.Event(kind, key=key, mod=self.mods, scancode=0, unicode=name if len(name) == 1 else ""))

    def _native(self, event):
        """The mouse position of a browser event, in pixels of the pygame window (the picture may be shown smaller)."""
        surface = pygame.display.get_surface()
        width, height = surface.get_size() if surface is not None else (1, 1)
        shown_w = event.get("boundingRectWidth") or self.width
        shown_h = event.get("boundingRectHeight") or self.width * height / width
        x = event.get("relativeX", event.get("offsetX", 0)) * width / shown_w
        y = event.get("relativeY", event.get("offsetY", 0)) * height / shown_h
        return max(0, min(width - 1, int(round(x)))), max(0, min(height - 1, int(round(y))))

    def on_mouse(self, event):
        self._update_mods(event)
        kind = event.get("event")
        if kind == "contextmenu":                           # right button: the program gets it as a button, not a menu
            return
        previous, pos = self.mouse_pos, self._native(event)
        self.mouse_pos = pos
        if kind in ("mousedown", "mouseup"):
            button = _BUTTONS.get(event.get("button", 0), 1)
            self.mouse_buttons[button - 1] = kind == "mousedown"
            pygame.event.post(pygame.event.Event(pygame.MOUSEBUTTONDOWN if kind == "mousedown" else pygame.MOUSEBUTTONUP,
                                                 pos=pos, button=button, touch=False))
        elif kind == "mousemove":
            pygame.event.post(pygame.event.Event(pygame.MOUSEMOTION, pos=pos, rel=(pos[0] - previous[0], pos[1] - previous[1]),
                                                 buttons=tuple(self.mouse_buttons), touch=False))
        elif kind == "wheel":
            step = -1 if event.get("deltaY", 0) > 0 else 1   # the browser's down is pygame's negative
            if event.get("deltaY", 0):
                pygame.event.post(pygame.event.Event(pygame.MOUSEWHEEL, x=0, y=step, flipped=False, precise_x=0.0,
                                                     precise_y=float(step), which=0, touch=False))
        elif kind == "mouseleave":                          # a button released outside the picture would stay pressed
            for index, pressed in enumerate(self.mouse_buttons):
                if pressed:
                    self.mouse_buttons[index] = False
                    pygame.event.post(pygame.event.Event(pygame.MOUSEBUTTONUP, pos=pos, button=index + 1, touch=False))

    def show_frames(self):
        sent = None
        while not self.stopped.is_set():
            try:
                frame = self._frame
                if frame is not None and frame is sent:             # nothing new to send
                    time.sleep(1 / self.fps)
                    continue
                sent = frame
                if frame is None and pygame.display.get_surface() is not None:      # a program that never calls flip / update
                    frame = np.asarray(pygame.surfarray.array3d(pygame.display.get_surface())).swapaxes(0, 1)
                if frame is not None:
                    shown_height = round(self.width * frame.shape[0] / frame.shape[1])
                    if self.screen.layout.width != f"{self.width}px":       # show it at exactly this size, whatever the page layout does
                        self.screen.layout.width, self.screen.layout.height = f"{self.width}px", f"{shown_height}px"
                        self.screen.width, self.screen.height = str(self.width), str(shown_height)
                    buffer = io.BytesIO()
                    Image.fromarray(frame).save(buffer, "JPEG", quality=self.quality)
                    self.screen.value = buffer.getvalue()
            except pygame.error:                        # the program closed its window between two frames
                pass
            time.sleep(1 / self.fps)

    def stop(self, _=None):
        if not self.stopped.is_set():
            pygame.event.post(pygame.event.Event(pygame.QUIT))     # ask the program's loop to finish
            self.game.join(timeout=3)
            self.stopped.set()
        self._finish()

    def _finish(self):
        """Run ``on_stop`` and ``summary`` once, however the program ended."""
        with self._finish_lock:
            if self._finished:
                return
            self._finished = True
        self._unhook_display()
        for callback in (self.on_stop, self.summary):
            if callback is None:
                continue
            try:
                text = callback()
            except Exception:
                self.out.append_stderr(traceback.format_exc())
                continue
            if text:
                self.out.append_stdout(str(text) + "\n")


def play_live(run, keys=None, fps=15, quality=70, width=480, summary=None, on_stop=None, side=None,
              hint="Click the picture, then use the keys."):
    """Run ``run()``, a function with a normal pygame loop, and play it in this cell. Click the picture first.

    ``keys`` maps browser key names to pygame keys, e.g. ``{"e": pygame.K_TAB}``. ``fps`` and ``quality`` set how often and how
    sharp the picture is sent (lower them on a slow connection), and ``width`` is the width in pixels it is shown at. ``on_stop`` and then ``summary`` are called once when the program
    ends, whether you pressed Stop or it finished by itself, and any text they return is printed. The program should end its
    loop on ``pygame.QUIT``, which is what Stop sends. ``hint`` is the line shown above the picture.
    """
    global _session
    if _session is not None:
        _session.stop()                                 # a new session replaces the previous one
    session = _session = _Session(run, keys, fps, quality, width, summary, on_stop)

    keyboard = Event(source=session.screen, watched_events=["keydown", "keyup"], prevent_default_action=True)
    keyboard.on_dom_event(session.on_key)
    mouse = Event(source=session.screen, watched_events=["mousedown", "mouseup", "mouseleave", "wheel", "contextmenu"],
                  prevent_default_action=True)          # no page scrolling on the wheel, no menu on a right click
    mouse.on_dom_event(session.on_mouse)
    try:                                                # movement is frequent: send at most one event every 40 ms
        motion = Event(source=session.screen, watched_events=["mousemove"], throttle_or_debounce="throttle", wait=40)
    except Exception:
        motion = Event(source=session.screen, watched_events=["mousemove"])
    motion.on_dom_event(session.on_mouse)
    session._events = (keyboard, mouse, motion)         # keep them alive as long as the session
    stop_button = widgets.Button(description="Stop")
    stop_button.on_click(session.stop)
    top = widgets.Layout(align_items="flex-start")           # do not stretch the children to the width of the page
    picture = session.screen if side is None else widgets.HBox([session.screen, side], layout=top)
    display(widgets.VBox([widgets.HTML(hint), picture, stop_button, session.out], layout=top))

    session._hook_display()
    session.game.start()
    threading.Thread(target=session.show_frames, daemon=True).start()


def play_mosaic(gui, fps=15, quality=70, width=480, on_stop=None):
    """Play a MOSAIC ``SAREnvGUI`` live. Arrows turn and walk, Space opens a door, E picks up or rescues, Q asks the teammate."""
    def score():
        status = gui.user.env.get_mission_status()
        return (f"rescued {status['saved_victims']} of {status['saved_victims'] + status['remaining_victims']}, "
                f"reward {gui.user.total_reward:+.1f}, {gui.user.total_steps} steps")
    play_live(gui.run, keys=MOSAIC_KEYS, fps=fps, quality=quality, width=width, summary=score, on_stop=on_stop)


def play_view(player, fps=15, quality=70, width=320):
    """Play a MOSAIC ``User`` with only the game view: no information or chat panel. A box beside it shows the agent's observation,
    updated after every action. Arrows turn and walk, Space opens a door, E picks up or rescues. ``player.obs`` keeps the last observation."""
    readout = widgets.HTML()

    def show_observation():
        lines = []
        for key, value in (player.obs or {}).items():
            if key == "grid":
                value = "(the grid: see below)"
            elif isinstance(value, (list, dict)):
                value = f"{type(value).__name__} of length {len(value)}"
            lines.append(f"{key:18s} {value}")
        readout.value = "<pre>" + "\n".join(lines) + "</pre>"

    def run():
        pygame.init()
        player.reset()
        player.on_step = lambda info: show_observation()
        show_observation()
        window, clock = None, pygame.time.Clock()
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
                if event.type == pygame.KEYDOWN:
                    player.handle_key(SimpleNamespace(key=pygame.key.name(event.key)))
            frame = player.get_frame()
            if window is None:
                window = pygame.display.set_mode((frame.shape[1], frame.shape[0]))
            window.blit(pygame.surfarray.make_surface(frame.swapaxes(0, 1)), (0, 0))
            pygame.display.flip()
            clock.tick(30)

    play_live(run, keys=MOSAIC_KEYS, fps=fps, quality=quality, width=width, side=readout)


def play_shasta(gui, fps=15, quality=70, width=800, on_stop=None):
    """Play a SHaSTA ``ShastaGUI`` live. Click a group's marker (or press 1-9), click a street node, then Enter or the Send button.
    The wheel zooms, a right-drag pans, Space pauses, ``+`` / ``-`` change the speed, V shows the 3D view."""
    def score():
        mission = gui.commander.mission
        return f"targets reached {mission.score} of {len(mission.targets)}, {gui.commander.step_count} steps" if mission else ""
    play_live(gui.run, fps=fps, quality=quality, width=width, summary=score, on_stop=on_stop,
              hint="Click the picture, then use the mouse and the keys.")
