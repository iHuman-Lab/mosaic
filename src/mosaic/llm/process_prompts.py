from pathlib import Path

import pandas as pd
import yaml
from game.sar.observations import (
    FAKE_VICTIM,
    LAVA,
    VICTIM,
    WALL,
    cam_bounds,
    decode_door,
    decode_key,
    is_door,
    is_key,
)

from .pathfinding import query_all_objects

_DIR_NAMES = {0: "East", 1: "South", 2: "West", 3: "North"}
_EMPTY = 0
_REWARD = {"victim": "+1.0", "key": "0", "door": "0"}
_SORT_PRIORITY = {
    ("Yes", "Yes"): 0,
    ("Yes", "No"): 1,
    ("No", "Yes"): 2,
    ("No", "No"): 3,
}


def _room(x: int, y: int, room_size: int) -> str:
    rs = max(room_size - 1, 1)
    return f"({y // rs},{x // rs})"


def _compass(ax: int, ay: int, ox: int, oy: int) -> str:
    """Return dominant compass direction from agent (ax,ay) to object (ox,oy).
    y increases southward in grid coordinates."""
    dx, dy = ox - ax, oy - ay
    if abs(dx) >= abs(dy):
        return "East" if dx > 0 else "West"
    return "South" if dy > 0 else "North"


def _build_table(obs: dict) -> str:
    grid = obs["grid"]
    ax, ay = obs["agent_x"], obs["agent_y"]
    room_size = obs["room_size"]
    cx0, cy0, cx1, cy1 = cam_bounds(obs)
    bfs_results = query_all_objects(obs)
    victim_health_map = obs.get("victim_health", {})
    # Health depletes at 30 total over max_steps, only while visible
    deplete_per_step = 20.0 / max(obs["max_steps"], 1)

    rows = []
    counters: dict[str, int] = {}

    for y, row_cells in enumerate(grid):
        for x, cell in enumerate(row_cells):
            cell = int(cell)
            if cell in (_EMPTY, WALL, LAVA, FAKE_VICTIM) or (x == ax and y == ay):
                continue

            if cell == VICTIM:
                kind, label_base = "victim", "Victim"
            elif cell == FAKE_VICTIM:
                kind, label_base = "fake_victim", "FakeVictim"
            elif is_key(cell):
                kind, label_base = "key", f"{decode_key(cell)['color'].capitalize()}Key"
            elif is_door(cell):
                kind, label_base = (
                    "door",
                    f"{decode_door(cell)['color'].capitalize()}Door",
                )
            else:
                continue

            counters[label_base] = counters.get(label_base, 0) + 1
            pr = bfs_results.get((x, y))
            reachable = "Yes" if (pr and pr.reachable) else "No"
            path_length = (
                pr.path_length if (pr and pr.path_length is not None) else None
            )
            direction = _compass(ax, ay, x, y)

            if kind == "door":
                d = decode_door(cell)
                status = (
                    "Open"
                    if d["is_open"]
                    else ("Locked" if d["is_locked"] else "Closed")
                )
                tool = f"{d['color']} key" if d["is_locked"] else "None"
                health = "—"
                margin = "—"
                saveable = "—"
                sort_urgency = 99999
            elif kind == "victim":
                health_val = victim_health_map.get(f"{x},{y}", 1.0)
                health_pct = int(health_val * 100)
                tool = "None"
                is_visible = cx0 <= x < cx1 and cy0 <= y < cy1
                if health_val == 0.0:
                    status = "Dead"
                    health = "0% (Dead)"
                    margin = "—"
                    saveable = "No"
                    sort_urgency = 99999  # never prioritise dead victims
                elif is_visible and path_length is not None:
                    status = "Alive"
                    health = f"{health_pct}%"
                    steps_until_death = health_val / deplete_per_step
                    m = int(steps_until_death - path_length)
                    margin = str(m)
                    saveable = "Yes" if m >= 0 else "No"
                    sort_urgency = m
                else:
                    status = "Alive"
                    health = f"{health_pct}%"
                    margin = "—"  # not depleting while off-screen
                    saveable = "—"
                    sort_urgency = 99999
            elif kind == "fake_victim":
                status, tool, health, margin, saveable, sort_urgency = (
                    "Decoy",
                    "None",
                    "—",
                    "—",
                    "—",
                    99999,
                )
            else:
                status, tool, health, margin, saveable, sort_urgency = (
                    "Available",
                    "None",
                    "—",
                    "—",
                    "—",
                    99999,
                )

            visible = "Yes" if cx0 <= x < cx1 and cy0 <= y < cy1 else "No"
            rows.append(
                {
                    "Object": f"{label_base}{counters[label_base]}",
                    "Direction": direction,
                    "Room": _room(x, y, room_size),
                    "Visible": visible,
                    "Reachable": reachable,
                    "Health": health,
                    "Saveable": saveable,
                    "Margin": margin,
                    "Reward": _REWARD.get(kind, "0"),
                    "Status": status,
                    "ToolRequired": tool,
                    "PathLength": path_length if path_length is not None else "N/A",
                    "_sort_priority": _SORT_PRIORITY.get((reachable, visible), 3),
                    "_sort_urgency": sort_urgency,
                    "_sort_path": path_length if path_length is not None else 9999,
                }
            )

    if not rows:
        return "(no objects on map)"

    df = pd.DataFrame(rows)
    df = df.sort_values(["_sort_priority", "_sort_urgency", "_sort_path"]).drop(
        columns=["_sort_priority", "_sort_urgency", "_sort_path"]
    )
    return df.to_markdown(index=False)


def build_obs(obs: dict) -> str:
    inv = f"{obs['carrying']} key" if obs["carrying"] else "nothing"
    ax, ay = obs["agent_x"], obs["agent_y"]
    room_size = obs["room_size"]
    agent_room = _room(ax, ay, room_size)
    return "\n".join(
        [
            f"Agent: room {agent_room} facing {_DIR_NAMES.get(obs['agent_dir'], '?')} | Carrying: {inv} | Victims remaining: {obs['remaining_victims']} | Steps left: {obs['max_steps'] - obs['step_count']}",
            "",
            "Map objects:",
            _build_table(obs),
        ]
    )


def _build_grid_info(obs: dict) -> str:
    n_rows = obs["num_rows"]
    n_cols = obs["num_cols"]
    mid_r, mid_c = n_rows // 2, n_cols // 2
    return (
        f"There are {n_rows * n_cols} rooms arranged in a {n_rows}×{n_cols} grid. "
        f"({0},{mid_c}) is North, ({n_rows - 1},{mid_c}) is South, "
        f"({mid_r},{0}) is West, ({mid_r},{n_cols - 1}) is East."
    )


def _load_prompts() -> dict:
    path = Path(__file__).parent / "prompts.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


_PROMPTS: dict = _load_prompts()


def build_prompt(obs: dict, prompt_type: str = "sparse") -> str:
    suffix_key = "detailed_suffix" if prompt_type == "detailed" else "sparse_suffix"
    template = _PROMPTS["preamble"] + _PROMPTS[suffix_key]
    return template.format(obs=build_obs(obs), grid_info=_build_grid_info(obs))
