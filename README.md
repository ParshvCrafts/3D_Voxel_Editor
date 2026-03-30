# JARVIS-Style 3D Voxel Editor

A hand gesture controlled 3D voxel editor built with Python, MediaPipe, OpenCV, Pygame, and ModernGL. The project combines real-time hand tracking, a priority-based gesture recognition system, a sparse voxel engine, and a holographic AR-inspired renderer to create an interactive desktop editing experience reminiscent of cinematic sci-fi interfaces.

## Overview

This project turns a webcam into the primary input device for building and editing voxel structures. Instead of relying on a mouse for the core interaction loop, users place, delete, grab, rotate, scatter, restore, recolor, save, and export voxel models through single-hand and two-hand gestures.

The application is organized as a modular real-time system:

- `main.py` runs the application loop and routes gesture events into editor actions.
- `hand_tracker.py` uses MediaPipe to detect hands and extract landmark-derived hand state.
- `gestures.py` interprets those states with a hold-timer and priority-based gesture state machine.
- `voxel_engine.py` stores voxels, applies transforms, handles history, and manages physics-style effects.
- `renderer.py` draws the 3D scene with ModernGL, bloom, wireframes, and AR-friendly transparency.
- `ui_renderer.py` renders the HUD, webcam preview, cursors, overlays, particles, and gesture feedback.

## Highlights

- Real-time webcam-driven hand tracking with up to two hands
- Gesture-based voxel placement, deletion, grab/move, rotation, scatter, restore, and reset
- Sparse dictionary-backed voxel storage for efficient editing
- Group transform workflow for moving and rotating the entire structure without rebuilding per-voxel data every frame
- Holographic visual style with bloom, wireframes, cursors, particles, and HUD overlays
- AR mode with webcam background composited behind the 3D scene
- Undo/redo, save/load to JSON, and export to OBJ
- Debug overlay and clearly documented gesture timing, priorities, and failure modes

## Core Interaction Model

The current gesture system is intentionally strict about gesture priority to avoid conflicts between actions. The main production mapping is:

| Priority | Gesture | Action |
| --- | --- | --- |
| 1 | Both fists | Reset transform |
| 2 | Both palms | Enter continuous rotation |
| 2.5 | Right thumb up only | Full reset after 5 second hold |
| 3 | Right pinch + left point | Delete voxels under the left cursor |
| 4 | Right fist | Grab and move the whole structure |
| 5 | Left pinch only | Place voxels continuously |
| 6 | Left thumb down | Scatter voxels |
| 7 | Left thumb up | Restore voxels |

Additional edge-triggered gestures:

- Left victory sign cycles the active color palette
- Right victory starts disco mode
- Right open palm stops disco mode

Keyboard controls remain available for toggles, debugging, scene management, and fallback control paths.

## Architecture

The runtime follows a clear data flow:

1. OpenCV captures a raw webcam frame.
2. MediaPipe processes the unmirrored frame for correct handedness.
3. The frame is mirrored for display after hand processing.
4. `hand_tracker.py` produces structured hand data.
5. `gestures.py` evaluates that data through a state machine and emits gesture actions when appropriate.
6. `main.py` applies those actions to the voxel engine.
7. `renderer.py` renders the voxel scene and post-processing.
8. `ui_renderer.py` composites the webcam preview, HUD, cursors, loading circles, debug data, and effects.

Key implementation decisions:

- Voxels are stored sparsely as integer grid keys: `Dict[Tuple[int, int, int], Voxel]`.
- Placement uses a YouTube-style 2D mapping model; the working interaction plane keeps `z = 0.0`.
- Group grab and rotation operate through a model matrix instead of rewriting voxel coordinates every frame.
- The delete system intentionally has a direct reference from the gesture recognizer to the voxel engine to avoid cooldown deadlocks during continuous delete painting.
- AR rendering preserves alpha so the webcam can show through the scene correctly.

For deeper technical details, see:

- [architecture.md](./architecture.md)
- [prd.md](./prd.md)
- [memory.md](./memory.md)
- [progress.md](./progress.md)
- [techstack.md](./techstack.md)
- [CLAUDE.md](./CLAUDE.md)

## Tech Stack

- Python 3.8+
- MediaPipe for hand tracking
- OpenCV for webcam capture and image processing
- Pygame for the window, input loop, and OpenGL context setup
- ModernGL for rendering
- Pyrr for matrix and vector math
- NumPy for numerical operations

## Requirements

Hardware and platform expectations:

- A working webcam
- OpenGL 3.3+ support
- A desktop environment capable of opening a hardware-accelerated Pygame window

Recommended runtime assumptions from the current configuration:

- Window: `1280 x 720`
- Target framerate: `60 FPS`
- Webcam request: `1280 x 720` with fallback behavior depending on hardware
- Maximum voxel count: `10,000`

## Installation

```bash
git clone https://github.com/ParshvCrafts/3D_Voxel_Editor.git
cd 3D_Voxel_Editor
python -m venv .venv
```

Activate the virtual environment:

```bash
# Windows PowerShell
.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Running the App

```bash
python main.py
```

On startup, the application initializes:

- the OpenGL context
- webcam capture
- hand tracking
- gesture recognition
- voxel engine
- 3D renderer
- UI renderer

If the webcam cannot be opened, the app attempts to continue in a degraded mode instead of crashing immediately.

## Controls

### Gesture Controls

| Gesture | Action |
| --- | --- |
| Left pinch | Place voxels |
| Right pinch + left index point | Delete voxels in batch on release |
| Right fist | Grab and move the structure |
| Both palms | Rotate the structure on X and Y axes |
| Both fists | Reset transform |
| Right thumb up only | Full reset after hold |
| Left thumb down | Scatter |
| Left thumb up | Restore |
| Left victory | Cycle colors |
| Right victory | Start disco mode |
| Right palm | Stop disco mode |

### Keyboard Controls

| Key | Action |
| --- | --- |
| `ESC` | Quit |
| `H` | Toggle help overlay |
| `D` | Toggle debug overlay |
| `W` | Toggle webcam preview |
| `P` | Toggle particles |
| `A` | Toggle AR mode |
| `M` | Toggle symmetry |
| `Shift + X/Y/Z` | Set symmetry axis |
| `S` | Save scene to JSON |
| `L` | Load scene |
| `E` | Export to OBJ |
| `C` | Clear all voxels |
| `R` | Reset camera |
| `1-8` | Quick color select |
| `Ctrl + Z` | Undo |
| `Ctrl + Shift + Z` | Redo |
| `F1-F5` | Switch editor mode |
| `G` | Gravity burst / restore |
| `T` | Smooth restore |
| `O` | Cycle colors |
| `I` | Toggle disco mode |

## Project Structure

```text
3D_Voxel_Editor/
|-- main.py
|-- config.py
|-- hand_tracker.py
|-- gestures.py
|-- voxel_engine.py
|-- renderer.py
|-- ui_renderer.py
|-- utils.py
|-- requirements.txt
|-- shaders/
|-- assets/
|-- tests/
|-- architecture.md
|-- memory.md
|-- prd.md
|-- progress.md
|-- techstack.md
`-- CLAUDE.md
```

## Testing

The repository currently includes unit tests focused on the voxel engine, covering:

- clear/reset history behavior
- grab/move snapping with undo/redo
- save/load scene round-trips
- disco color freeze and authored-color restoration
- invalid scene loading failures

Run tests with:

```bash
python -m unittest discover -s tests
```

## Current Status

According to the project history documents, the codebase is currently at Phase 15 with all core editing features implemented. Verified working capabilities include:

- continuous placement
- continuous delete collection with batch removal
- grab/move of the full structure
- hologram-style rotation
- transform reset
- full reset with right-thumb-up safety gating
- color cycling
- AR mode
- save/load/export
- undo/redo

Items still worth further validation include some advanced or secondary flows such as grab preview visibility, disco stop/start ergonomics, scatter/restore edge behavior, symmetry mode, and multi-select box selection.

## Known Technical Notes

- The webcam frame is mirrored after MediaPipe processing, not before.
- The working placement model currently assumes `z = 0.0`.
- Grid keys must remain integer tuples for voxel lookup consistency.
- Continuous gesture handlers should avoid emitting repeated events during hold states because action cooldown can block state transitions.
- AR mode intentionally skips grid rendering to avoid flickering issues.

These details are documented more fully in [memory.md](./memory.md) and [architecture.md](./architecture.md).

## Roadmap Opportunities

The implemented foundation is strong enough to support future work such as:

- deeper 3D placement beyond the current interaction plane
- stronger test coverage for gesture recognition and renderer behavior
- improved onboarding and calibration flows for new users
- richer selection and editing tools
- polished export and asset pipeline features

## License

No license file is currently present in the repository. If you plan to open-source or distribute the project broadly, adding an explicit license is recommended.
