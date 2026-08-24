# Core Systems

## Environment

**Class:** `PickupVictimEnv` — `src/mosaic/sar/env.py`

The main Gymnasium-compatible environment. Inherits from `SARLevelGen`.

**Key configuration parameters:**

| Parameter | Default | Description |
| --- | --- | --- |
| `room_size` | 14 | Tiles per room (interior) |
| `num_rows` | 3 | Rows of rooms |
| `num_cols` | 3 | Columns of rooms |
| `victim_placer` | `VictimPlacer()` | Placer controlling victim placement |
| `lava_placer` | `LavaPlacer()` | Placer controlling lava placement |
| `locked_room_placer` | `LockedRoomPlacer()` | Placer controlling locked doors/keys |
| `agent_pov` | True | Enable first-person view camera |

**Key methods:**

| Method | Description |
| --- | --- |
| `gen_mission()` | Generates a full level: rooms, doors, keys, lava, victims |
| `reset()` | Initializes an episode; computes `max_steps` |
| `step(action)` | Processes one action; updates victim health, checks end conditions |
| `get_mission_status()` | Returns `{status, saved_victims, remaining_victims}` |
| `switch_camera(strategy)` | Swaps camera strategy at runtime |

Use the `build_sar_env()` factory (`src/mosaic/sar/env.py`) to construct a fully configured
instance from already-built placer instances instead of calling `PickupVictimEnv(...)` directly.

## Level generation

**Class:** `SARLevelGen` — `src/mosaic/core/level.py`

Inherits from MiniGrid's `MiniGridEnv`. Provides:

- Multi-room grid construction
- Pluggable camera injection
- Base rendering with camera-aware crop

Subclassed by `PickupVictimEnv` and `TutorialEnv`.

## Victim system

**File:** `src/mosaic/sar/objects.py`

### Real victims (`Victim`)

- Rendered as a symmetric **cross (✚)**
- Directional variants: `victim_up`, `victim_down`, `victim_left`, `victim_right`
- Have a `health` property (0.0 → 1.0) with a rendered health bar
- Reward on rescue: **+1.0**

### Fake victims (`FakeVictim`)

- Rendered as an asymmetric **T-shape (⊤)**
- Left/right shift variants to vary appearance
- Same health depletion system as real victims
- Penalty on accidental rescue: **-0.5**

### Placement (`VictimPlacer`) — `src/mosaic/sar/placers.py`

- Real victims placed preferentially in locked rooms (harder to reach)
- Fake victims distributed across all accessible rooms
- Reachability verified before placement (no inaccessible victims)

## Hazards & obstacles

**Lava** (`LavaPlacer`) — `src/mosaic/sar/placers.py`

- Placed randomly within rooms, avoiding doorways and agent spawn
- Stepping on lava immediately terminates the episode with failure
- Density controlled by the placer's configuration

**Locked doors & keys**

- A subset of room connections are randomly locked, controlled by `LockedRoomPlacer`
- A matching colored key is placed in a reachable (unlocked) room
- The agent must carry the key to toggle the locked door open

## Mission system

**Class:** `PickupAllVictimsInstr` — `src/mosaic/sar/instructions.py`

Manages mission completion logic:

- Tracks rescued vs. remaining real victims
- Reports `success` when all real victims rescued
- Reports `failure` on lava contact or timeout

## Camera system

**File:** `src/mosaic/core/camera.py`

Interchangeable camera strategies:

- **`FullviewCamera`** — Shows the entire grid at once. Best for small maps or debugging.
- **`AgentCenteredCamera`** — Centers the viewport on the agent. Entire room plus border tiles always visible.
- **`EdgeFollowCamera`** *(default)* — Camera moves only when the agent approaches the edge of the viewport (dead-zone tracking).
- **`AgentFOVCamera`** — Clips the view to the boundaries of the agent's current room. Natural wall framing.
- **`AgentConeCamera`** — Room-bounded view with MiniGrid's line-of-sight cone; tiles outside the agent's forward sightline are blacked out.

Switch cameras at runtime:

```python
env.switch_camera("edge_follow")   # or "full", "centered", "fov", "cone"
```

## Observation encoding

**Class:** `GameObservation` — `src/mosaic/sar/observations.py`

Returns a dict on each step:

| Key | Type | Description |
| --- | --- | --- |
| `image` | ndarray (H×W×3) | RGB camera frame |
| `direction` | int | Agent facing: 0=E, 1=S, 2=W, 3=N |
| `agent_x`, `agent_y` | int | Agent grid position |
| `grid` | ndarray (H×W) | Full map encoded as integers (see below) |
| `carrying` | str or None | Color of held object |
| `step_count` | int | Steps elapsed this episode |
| `max_steps` | int | Episode step limit |
| `mission_status` | str | `"success"`, `"failure"`, or `"incomplete"` |
| `saved_victims` | int | Count of rescued real victims |
| `remaining_victims` | int | Count of unrescued real victims |
| `num_rows`, `num_cols` | int | Map dimensions in rooms |
| `room_size` | int | Room size in tiles |
| `cam_top_x`, `cam_top_y` | int | Camera viewport top-left tile |
| `cam_view_w`, `cam_view_h` | int | Camera viewport size in tiles |

**Grid integer encoding:**

```text
0       = Empty floor
1       = Wall
4       = Lava
5       = Victim (real)
6       = FakeVictim (decoy)

Doors:  10 + (color_index × 3) + door_state
  door_state: 0=open, 1=closed, 2=locked
  color_index: red=0, green=1, blue=2, purple=3, yellow=4, grey=5
  Range: 10-27

Keys:   30 + color_index
  Range: 30-35
```

## Reward system

**Class:** `RescueAction` — `src/mosaic/sar/actions.py`

All rewards are sparse (returned only on specific events):

| Event | Reward |
| --- | --- |
| Rescue a real victim | +1.0 |
| Pick up a fake victim | -0.5 |
| All victims rescued (mission complete) | +1.0 bonus |
| Step on lava | episode terminated, no reward |
| Timeout | episode terminated, no reward |
