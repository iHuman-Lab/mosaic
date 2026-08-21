"""Tests for the GUI/LLM dependency-injection boundary: User, SAREnvGUI, and
ChatPanel receive an already-constructed LLMClient instead of building one
from provider/model config. No monkeypatching — all seams are exercised
through fakes and real (injected) collaborators.
"""

import os
import subprocess
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pygame
import pytest
from minigrid.core.actions import Actions

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

REPO_ROOT = Path(__file__).resolve().parent.parent

from mosaic.gui.chat import ChatPanel
from mosaic.gui.info import InfoPanel
from mosaic.gui.main import SAREnvGUI
from mosaic.gui.user import User
from mosaic.llm.client import DummyLLMClient, LLMClient
from mosaic.sar.env import PickupVictimEnv


def _make_env():
    """A fast, deterministic SAR env for GUI/User construction tests."""
    env = PickupVictimEnv(
        room_size=6, num_rows=2, num_cols=2, num_dists=4, render_mode="rgb_array"
    )
    env.reset(seed=1)
    return env


class FakeLLMClient(LLMClient):
    def __init__(self, response="advice"):
        self.response = response
        self.prompts = []

    def query(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.response


class ExplodingLLMClient(LLMClient):
    def query(self, prompt: str) -> str:
        raise RuntimeError("boom")


# --- User ------------------------------------------------------------------


def test_user_defaults_to_dummy_llm_client_when_none_given():
    user = User(_make_env(), llm_client=None)
    assert isinstance(user.llm_client, DummyLLMClient)


def test_user_rejects_non_llmclient():
    with pytest.raises(TypeError):
        User(_make_env(), llm_client=object())


def test_user_ask_llm_uses_injected_client():
    user = User(_make_env(), llm_client=FakeLLMClient(response="go north"))
    user.obs = user.env.reset(seed=2)[0]
    user.steps_since_last_llm = 7

    result = user.ask_llm()

    assert result == "go north"
    assert user.steps_since_last_llm == 0


def test_user_ask_llm_async_snapshot_is_isolated_from_later_mutation():
    """The async path must deep-copy obs before handing it to the prompt
    builder, so mutating the live obs mid-flight cannot affect the request
    already in progress."""
    captured = {}
    saw_obs_event = threading.Event()
    release_event = threading.Event()

    def blocking_builder(obs):
        captured["obs"] = obs
        captured["nested_before_release"] = dict(obs["nested"])
        saw_obs_event.set()
        release_event.wait(timeout=5)
        return "prompt"

    user = User(
        _make_env(),
        llm_client=FakeLLMClient(response="advice"),
        prompt_builder=blocking_builder,
    )
    user.obs = {"nested": {"value": 1}}

    user.ask_llm_async()
    assert saw_obs_event.wait(timeout=5)

    # Mutate the *original* obs while the background thread is mid-flight.
    user.obs["nested"]["value"] = 999

    release_event.set()
    user.llm_thread.join(timeout=5)

    assert captured["nested_before_release"] == {"value": 1}
    assert captured["obs"]["nested"]["value"] == 1
    assert user.llm_result == ("reply", "advice")


def test_user_ask_llm_async_error_path():
    user = User(_make_env(), llm_client=ExplodingLLMClient())
    user.obs = user.env.reset(seed=3)[0]

    user.ask_llm_async()
    user.llm_thread.join(timeout=5)

    kind, value = user.llm_result
    assert kind == "error"
    assert "boom" in value


def test_user_ask_llm_no_observation_yet():
    user = User(_make_env(), llm_client=FakeLLMClient())
    assert user.obs is None
    assert user.ask_llm() == "No observation available yet."


# --- SAREnvGUI ---------------------------------------------------------


def test_sar_env_gui_constructs_with_injected_client_and_no_provider_config():
    env = _make_env()
    fake = FakeLLMClient()
    config = {"prompt_type": "detailed", "max_time": 1.0}  # no provider/model keys

    gui = SAREnvGUI(env, config=config, llm_client=fake)

    assert gui.user.llm_client is fake
    assert not hasattr(gui, "provider")
    assert not hasattr(gui, "model")


def test_sar_env_gui_defaults_to_dummy_client_with_no_llm_client_given():
    gui = SAREnvGUI(_make_env())
    assert isinstance(gui.user.llm_client, DummyLLMClient)


def test_sar_env_gui_set_advice_callbacks_applies_immediately_and_survives_recreate():
    gui = SAREnvGUI(_make_env(), llm_client=FakeLLMClient())
    calls = []

    gui.set_advice_callbacks(on_reply=lambda: calls.append("reply"), on_end=lambda: calls.append("end"))
    assert gui.chat_panel.on_llm_reply is not None
    assert gui.chat_panel.on_blink_end is not None

    gui._create_panels()  # simulates what toggle_fullscreen() triggers
    assert gui.chat_panel.on_llm_reply is not None
    gui.chat_panel.on_llm_reply()
    assert calls == ["reply"]


class _RecordingUser(User):
    instantiated = False

    def __init__(self, *args, **kwargs):
        type(self).instantiated = True
        super().__init__(*args, **kwargs)


class _RecordingInfoPanel(InfoPanel):
    instantiated = False

    def __init__(self, *args, **kwargs):
        type(self).instantiated = True
        super().__init__(*args, **kwargs)


class _RecordingChatPanel(ChatPanel):
    instantiated = False

    def __init__(self, *args, **kwargs):
        type(self).instantiated = True
        super().__init__(*args, **kwargs)


def test_sar_env_gui_uses_injected_component_classes():
    gui = SAREnvGUI(
        _make_env(),
        llm_client=FakeLLMClient(),
        user_class=_RecordingUser,
        info_panel_class=_RecordingInfoPanel,
        chat_panel_class=_RecordingChatPanel,
    )

    assert isinstance(gui.user, _RecordingUser)
    assert isinstance(gui.info_panel, _RecordingInfoPanel)
    assert isinstance(gui.chat_panel, _RecordingChatPanel)
    assert _RecordingUser.instantiated
    assert _RecordingInfoPanel.instantiated
    assert _RecordingChatPanel.instantiated

    # Injected classes must survive panel recreation (what toggle_fullscreen()
    # triggers), since that's the whole reason they're stored on the instance.
    gui._create_panels()
    assert type(gui.info_panel) is _RecordingInfoPanel
    assert type(gui.chat_panel) is _RecordingChatPanel


def test_sar_env_gui_default_component_classes_when_not_injected():
    gui = SAREnvGUI(_make_env(), llm_client=FakeLLMClient())

    assert type(gui.user) is User
    assert type(gui.info_panel) is InfoPanel
    assert type(gui.chat_panel) is ChatPanel


def test_gui_handle_user_input_steps_the_environment():
    """Exercises the real SAREnvGUI -> User -> env.step() path a key press
    takes, not just User.handle_key() in isolation."""
    gui = SAREnvGUI(_make_env(), llm_client=FakeLLMClient())
    try:
        gui.reset()
        initial_steps = gui.user.total_steps

        gui.handle_user_input(SimpleNamespace(type=pygame.KEYDOWN, key=pygame.K_UP))

        assert gui.user.total_steps == initial_steps + 1
        assert gui.user.last_action == Actions.forward
    finally:
        gui.close()


def test_importing_mosaic_gui_does_not_initialize_pygame():
    """Run in a subprocess so this can't be order-dependent on some other
    test in the same session having already called pygame.init()."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import pygame, mosaic.gui.main; "
            "assert not pygame.get_init(); "
            "assert not pygame.display.get_init(); "
            "assert pygame.display.get_surface() is None",
        ],
        cwd=str(REPO_ROOT / "src"),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


# --- ChatPanel ---------------------------------------------------------


def _panel_and_manager():
    import pygame_gui

    manager = pygame_gui.UIManager((400, 400))
    panel = ChatPanel(manager, 0, 0, 200, 400)
    return panel


class _FakeUser:
    def __init__(self):
        self.llm_thread = threading.Thread(target=lambda: None)
        self.llm_thread.start()
        self.llm_thread.join()
        self.llm_result = None


def test_chat_panel_fires_on_reply_only_for_successful_result():
    panel = _panel_and_manager()
    calls = []
    panel.on_llm_reply = lambda: calls.append("fired")

    user = _FakeUser()
    user.llm_result = ("reply", "go east")
    panel.poll_llm(user)

    assert calls == ["fired"]


def test_chat_panel_does_not_fire_on_error_result():
    panel = _panel_and_manager()
    calls = []
    panel.on_llm_reply = lambda: calls.append("fired")

    user = _FakeUser()
    user.llm_result = ("error", "network down")
    panel.poll_llm(user)

    assert calls == []
