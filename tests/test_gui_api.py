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
from mosaic.gui.feedback import EdgeVignette
from mosaic.gui.info import InfoPanel
from mosaic.gui.main import SAREnvGUI
from mosaic.gui.user import User
from mosaic.llm.client import DummyLLMClient, LLMClient
from mosaic.sar.env import PickupVictimEnv
from mosaic.sar.objects import Victim
from mosaic.sar.placers import VictimPlacer


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


def _single_victim_env():
    """A 2x2 env with exactly one manually-placed real victim directly in
    front of the agent, for deterministic rescue+mission_complete testing."""
    env = PickupVictimEnv(
        room_size=6,
        num_rows=2,
        num_cols=2,
        num_dists=4,
        victim_placer=VictimPlacer(num_real_victims=0),
        render_mode="rgb_array",
    )
    env.reset(seed=1)
    env.max_steps = 100  # see tests/test_actions.py::_reset for why
    room = next(
        env.get_room(i, j)
        for i in range(env.num_rows)
        for j in range(env.num_cols)
        if not getattr(env.get_room(i, j), "locked", False)
    )
    top_x, top_y = room.top
    agent_pos = (top_x + 1, top_y + 1)
    env.grid.set(*agent_pos, None)
    env.agent_pos = agent_pos
    env.agent_dir = 0  # facing right (+x)
    env.put_obj(Victim("up", color="red"), top_x + 2, top_y + 1)
    env._victims = env.find_objects_by_type((Victim,))
    env.total_victims = len(env._victims)
    return env


def test_user_step_sets_last_info_from_env():
    user = User(_make_env(), llm_client=FakeLLMClient())
    user.reset()

    user.step(Actions.left)

    assert user.last_info == {"events": []}


def test_user_reset_clears_last_info():
    user = User(_make_env(), llm_client=FakeLLMClient())
    user.reset()
    user.step(Actions.left)
    assert user.last_info != {}

    user.reset()

    assert user.last_info == {}


def test_user_on_step_fires_with_mission_complete_before_terminal_reset_clears_last_info():
    """User.step() auto-resets on terminated=True, which would otherwise
    silently wipe last_info before the GUI ever sees a mission_complete
    event. on_step must fire with the real info first."""
    # Not user.reset() — the env is already carefully hand-placed, and
    # reset() would re-randomize it via env.reset(). step() only needs
    # total_reward to exist first (normally set by reset()).
    user = User(_single_victim_env(), llm_client=FakeLLMClient())
    user.total_reward = 0.0
    captured = []
    user.on_step = lambda info: captured.append(info)

    user.step(Actions.pickup)

    assert len(captured) == 1
    assert any(e["type"] == "mission_complete" for e in captured[0]["events"])
    assert user.last_info == {}


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
    attached_count = 0

    def attach(self, manager):
        type(self).attached_count += 1
        super().attach(manager)


class _RecordingChatPanel(ChatPanel):
    attached_count = 0

    def attach(self, manager):
        type(self).attached_count += 1
        super().attach(manager)


class _RecordingEdgeVignette(EdgeVignette):
    instantiated = False

    def __init__(self, *args, **kwargs):
        type(self).instantiated = True
        super().__init__(*args, **kwargs)


def test_sar_env_gui_uses_injected_component_instances():
    env = _make_env()
    injected_user = _RecordingUser(env, llm_client=FakeLLMClient())
    injected_vignette = _RecordingEdgeVignette(env.screen_size)
    injected_info_panel = _RecordingInfoPanel(env.screen_size, env.screen_size // 2)
    injected_chat_panel = _RecordingChatPanel(
        env.screen_size, 0, env.screen_size // 2, env.screen_size
    )

    gui = SAREnvGUI(
        env,
        llm_client=FakeLLMClient(),
        user=injected_user,
        info_panel=injected_info_panel,
        chat_panel=injected_chat_panel,
        vignette=injected_vignette,
    )

    assert gui.user is injected_user
    assert gui.info_panel is injected_info_panel
    assert gui.chat_panel is injected_chat_panel
    assert gui.vignette is injected_vignette
    assert _RecordingUser.instantiated
    assert _RecordingEdgeVignette.instantiated
    assert _RecordingInfoPanel.attached_count == 1
    assert _RecordingChatPanel.attached_count == 1

    # info_panel/chat_panel are the same instances across panel recreation
    # (what toggle_fullscreen() triggers) — only their pygame_gui widget tree
    # is rebuilt, via a second attach() call to the fresh manager.
    gui._create_panels()
    assert gui.info_panel is injected_info_panel
    assert gui.chat_panel is injected_chat_panel
    assert _RecordingInfoPanel.attached_count == 2
    assert _RecordingChatPanel.attached_count == 2


def test_sar_env_gui_defaults_when_not_injected():
    gui = SAREnvGUI(_make_env(), llm_client=FakeLLMClient())

    assert type(gui.user) is User
    assert type(gui.info_panel) is InfoPanel
    assert type(gui.chat_panel) is ChatPanel
    assert type(gui.vignette) is EdgeVignette


def test_sar_env_gui_components_survive_create_panels():
    """user/vignette/info_panel/chat_panel are all built once and never
    replaced — a fullscreen toggle only rebuilds pygame_gui widgets via
    attach(), it never swaps in new objects. An in-flight vignette animation,
    and any external references to these instances, survive it."""
    gui = SAREnvGUI(_make_env(), llm_client=FakeLLMClient())
    original_vignette = gui.vignette
    original_info_panel = gui.info_panel
    original_chat_panel = gui.chat_panel

    gui._create_panels()

    assert gui.vignette is original_vignette
    assert gui.info_panel is original_info_panel
    assert gui.chat_panel is original_chat_panel


def test_sar_env_gui_wires_user_on_step_to_vignette_trigger():
    gui = SAREnvGUI(_make_env(), llm_client=FakeLLMClient())
    triggered = []
    gui.vignette.trigger = lambda events: triggered.append(events)

    gui.user.on_step({"events": [{"type": "victim_rescued", "reward": 1.0}]})

    assert triggered == [[{"type": "victim_rescued", "reward": 1.0}]]


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
    panel = ChatPanel(0, 0, 200, 400)
    panel.attach(manager)
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
