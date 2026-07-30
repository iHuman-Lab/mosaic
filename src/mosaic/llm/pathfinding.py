"""BFS-based pathfinding utilities for LLM prompt generation.

All queries operate on the encoded integer grid from obs["grid"].
Lava and walls are impassable. Locked doors are impassable unless the
agent is carrying the matching colour key.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from game.sar.observations import (
    _COLORS,
    DOOR_BASE,
    KEY_BASE,
    LAVA,
    WALL,
    decode_door,
    is_door,
    is_key,
)

_EMPTY = 0
_DIRS = ((1, 0), (-1, 0), (0, 1), (0, -1))


@dataclass
class PathResult:
    reachable: bool
    path_length: int | None  # steps to stand adjacent and interact
    blocking_doors: list[str]  # colours of locked doors on shortest path


def _is_passable(cell: int, carrying: str | None) -> bool:
    """Return True if the agent can move through this cell."""
    if cell == _EMPTY:
        return True
    if cell == WALL or cell == LAVA:
        return False
    if is_door(cell):
        offset = cell - DOOR_BASE
        state = offset % 3  # 0=open, 1=closed, 2=locked
        if state == 0:
            return True
        if state == 2:  # locked
            return carrying is not None and _COLORS[offset // 3] == carrying
        return True  # closed but unlocked
    if is_key(cell):
        return True
    return False  # victims / fake victims block movement


def _bfs_full(
    grid: list[list[int]],
    start: tuple[int, int],
    carrying: str | None,
) -> tuple[dict, dict]:
    """Single BFS from start returning (dist, parent) for all reachable cells."""
    rows = len(grid)
    cols = len(grid[0]) if rows else 0
    dist: dict[tuple[int, int], int] = {start: 0}
    parent: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    queue: deque[tuple[int, int]] = deque([start])

    while queue:
        cx, cy = queue.popleft()
        for dx, dy in _DIRS:
            nx, ny = cx + dx, cy + dy
            if not (0 <= nx < cols and 0 <= ny < rows):
                continue
            if (nx, ny) in dist:
                continue
            if not _is_passable(int(grid[ny][nx]), carrying):
                continue
            dist[(nx, ny)] = dist[(cx, cy)] + 1
            parent[(nx, ny)] = (cx, cy)
            queue.append((nx, ny))

    return dist, parent


def _blocking_doors(
    grid: list[list[int]],
    parent: dict,
    end: tuple[int, int],
) -> list[str]:
    """Trace path from end to start via parent links, collect locked door colours."""
    blocking = []
    node = end
    while node is not None:
        x, y = node
        cell = int(grid[y][x])
        if is_door(cell):
            d = decode_door(cell)
            if d["is_locked"]:
                blocking.append(d["color"])
        node = parent.get(node)
    return blocking


def query_all_objects(obs: dict) -> dict[tuple[int, int], PathResult]:
    """Single BFS from the agent to find path info for every object of interest."""
    grid = obs["grid"]
    ax, ay = obs["agent_x"], obs["agent_y"]
    carrying = obs.get("carrying")
    start = (ax, ay)
    rows = len(grid)
    cols = len(grid[0]) if rows else 0

    dist, parent = _bfs_full(grid, start, carrying)

    results: dict[tuple[int, int], PathResult] = {}
    for y, row in enumerate(grid):
        for x, cell in enumerate(row):
            cell = int(cell)
            if cell in (_EMPTY, WALL) or (x, y) == start:
                continue

            # Find the closest reachable cell adjacent to this object
            best_dist: int | None = None
            best_neighbor: tuple[int, int] | None = None
            for dx, dy in _DIRS:
                nx, ny = x + dx, y + dy
                if not (0 <= nx < cols and 0 <= ny < rows):
                    continue
                d = dist.get((nx, ny))
                if d is not None and (best_dist is None or d < best_dist):
                    best_dist = d
                    best_neighbor = (nx, ny)

            if best_neighbor is None:
                results[(x, y)] = PathResult(
                    reachable=False, path_length=None, blocking_doors=[]
                )
            else:
                results[(x, y)] = PathResult(
                    reachable=True,
                    path_length=best_dist,
                    blocking_doors=_blocking_doors(grid, parent, best_neighbor),
                )

    return results
