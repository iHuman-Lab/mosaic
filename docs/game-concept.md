# Game Concept & Mechanics

![Agent view of a search-and-rescue building, with lava hazards, a locked-room key, and a decoy victim](assets/images/gui-screenshot.png)

## Objective

Rescue **all real victims** before time runs out. Real victims are displayed as a **cross shape (✚)**. Fake victims (decoys) appear as a **T-shape (⊤)** — picking them up applies a score penalty.

## Building layout

The building is a grid of rooms (default 2x2 or 3x3), each separated by walls with doorways. Some doors are **locked** and require a matching colored key found elsewhere in the building.

## Hazards

| Hazard | Effect |
| --- | --- |
| Lava | Instant mission failure if stepped on |
| Locked doors | Block passage until matching key is collected |
| Time limit | Mission fails if max steps exceeded |

## Victim health

All victims (real and fake) have a health bar that depletes over time while they are visible to the agent. This creates urgency — visible victims that are not rescued will eventually be lost.

## Scoring

| Event | Reward |
| --- | --- |
| Rescue real victim | +1.0 |
| Pick up fake victim (decoy) | -0.5 |
| Complete mission (all rescued) | +1.0 bonus |

## Controls

| Key | Action |
| --- | --- |
| `↑` or `W` | Move forward |
| `←` | Rotate left |
| `→` | Rotate right |
| `Space` | Toggle / interact (open doors) |
| `Tab` or `Page Up` | Pick up / rescue victim |
| `Left Shift` or `Page Down` | Drop held object |
| `Alt` (left or right) | Ask LLM for a suggestion |
| `F11` | Toggle fullscreen |
| `Backspace` | Reset current mission |
| `ESC` | Quit |
