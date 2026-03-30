# JARVIS-Style 3D Voxel Editor

Hand gesture controlled 3D voxel editor with holographic visual effects inspired by Iron Man's JARVIS interface.

## Quick Links

| Document | Purpose |
|----------|---------|
| [techstack.md](techstack.md) | Dependencies, tech stack, how to run |
| [architecture.md](architecture.md) | System design, rendering pipeline, coordinate systems |
| [prd.md](prd.md) | Gesture mappings, controls, feature specs |
| [progress.md](progress.md) | Phase history, fixes applied, what's verified |
| [memory.md](memory.md) | Known bugs, gotchas, debugging patterns |

## Project Structure

```
3D Voxel Editor/
├── main.py              # App entry point, event loop, gesture→action routing
├── config.py            # All tunable parameters (dataclasses)
├── hand_tracker.py      # MediaPipe hand tracking, 3D coordinate extraction
├── gestures.py          # Gesture state machine (priority-based detection)
├── voxel_engine.py      # Voxel storage, physics, transforms, undo/redo
├── renderer.py          # ModernGL 3D rendering (instanced voxels, bloom)
├── ui_renderer.py       # 2D UI (cursors, particles, webcam, HUD)
├── utils.py             # Math helpers, filters, ray casting
├── requirements.txt     # Python dependencies
├── shaders/             # GLSL shaders (voxel, wireframe, grid, bloom, etc.)
└── assets/              # Assets directory
```

## Key Architectural Decisions

1. **Sparse voxel storage** — `Dict[Tuple[int,int,int], Voxel]` keyed by grid position
2. **Group transforms via model matrix** — grab/rotate update a uniform, not per-voxel data
3. **Direct block collection for delete** — gesture system holds a ref to voxel_engine to bypass event cooldown deadlock (Phase 13 fix)
4. **Frame-based mirroring** — `cv2.flip` after MediaPipe, shader mirror disabled
5. **2D placement** — Z is always 0 (matches YouTube reference)
6. **Priority-based gesture detection** — strict ordering prevents conflicts
7. **Full reset isolated to the right thumb-up pose** — avoids stealing the left-hand placement gesture
