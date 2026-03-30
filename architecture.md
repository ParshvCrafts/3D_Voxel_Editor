# Architecture

## System Overview

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│   Webcam     │───▶│ HandTracker  │───▶│   Gesture    │
│  (cv2)       │    │ (MediaPipe)  │    │ Recognizer   │
└──────────────┘    └──────────────┘    └──────┬───────┘
                                               │ GestureEvent
                                               ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  Renderer    │◀───│  VoxelEngine │◀───│   main.py    │
│ (ModernGL)   │    │ (dict-based) │    │ (event loop) │
└──────────────┘    └──────────────┘    └──────────────┘
        │
        ▼
┌──────────────┐
│ UIRenderer   │
│ (HUD, webcam)│
└──────────────┘
```

## Data Flow

1. **Webcam** → raw frame → **HandTracker** processes with MediaPipe
2. Frame is `cv2.flip(1)` AFTER MediaPipe (mirror for display)
3. **HandTracker** produces `Hand3D` objects (landmarks, fingers_up, is_pinching)
4. **GestureRecognizer** runs state machine → returns `GestureEvent` or `None`
5. **main.py** routes events to **VoxelEngine** methods
6. **Renderer** reads voxel data + model matrix → draws scene
7. **UIRenderer** draws webcam preview, cursors, particles, HUD

## Coordinate Systems

| System | Range | Usage |
|--------|-------|-------|
| Screen (pixels) | 0–1280, 0–720 | Pygame window |
| MediaPipe (norm) | 0.0–1.0 | Raw hand landmarks |
| World | ~(-9, 9) X, ~(-6, 6) Y | 3D scene |
| Grid (int) | Integers | Voxel dictionary keys |

**Mapping formula** (YouTube-style):
```python
world_x = (0.5 - mediapipe_x) * 18.0  # Inverts X
world_y = (0.5 - mediapipe_y) * 12.0  # Inverts Y
world_z = 0.0                          # Always 2D
```

## Rendering Pipeline

1. **AR background**: Render webcam fullscreen (if AR mode)
2. **Scene FBO**: Clear to transparent (AR) or dark background
3. **Voxels**: Instanced draw call with model matrix (group transform)
4. **Wireframes**: Edge overlay on voxels
5. **Preview cube**: Wireframe at cursor position (same pass, no flicker)
6. **Bloom**: Extract bright → blur (5 passes) → composite
7. **UI pass**: Webcam preview, cursors, particles, text

## Gesture State Machine

```
                    ┌─────────┐
            ┌──────▶│  IDLE   │◀──────┐
            │       └────┬────┘       │
            │            │            │
    cancel/done    detect gesture   cancel/done
            │            │            │
            │    ┌───────┴────────┐   │
            │    ▼                ▼   │
       ┌─────────┐         ┌─────────┐
       │ PLACING  │         │DELETING │
       │(L pinch) │         │(R pinch │
       │          │         │+L index)│
       └─────────┘         └─────────┘
            ...and GRABBING, ROTATING, RESETTING, etc.
```

**Priority order** (checked in `_handle_idle_state()`):
1. RESET — both fists (1s)
2. ROTATE — both palms (1s)
2.5. FULL_RESET — right thumb up only (5s, left hand must stay out of active gestures)
3. DELETE — right pinch + left index pointing
4. GRAB — right fist alone
5. PLACE — left pinch only (NO right pinch)
6. SCATTER/RESTORE — left thumb gestures

## Voxel Engine

- **Storage**: `Dict[Tuple[int,int,int], Voxel]` — sparse, grid-snapped
- **Group transform**: `group_offset` + `group_rotation` + `group_rotation_x`
  - Applied via `get_model_matrix()` → shader uniform `u_model`
  - Grab/rotate update these values without rebuilding buffers
  - `reset_group_transform()` zeros everything
- **delete_voxel(grid_pos)** — removes from dict, records history
- **clear()** — wipes all voxels, selection, and history
- **Physics states**: normal → scattered/gravity_burst → restoring → normal

## Delete System (Phase 13 Architecture)

The gesture system holds a **direct reference** to `voxel_engine` (set via `set_voxel_engine()`).

During delete mode:
1. `_handle_deleting_state()` reads `self._voxel_engine.voxels` directly
2. Blocks under cursor are added to `self.blocks_to_delete` set **without returning events**
3. Only ONE event (`batch_delete`) is returned when pinch is released
4. This bypasses the `ACTION_COOLDOWN` that previously blocked pinch release detection

## Full Reset Isolation

- Full reset now lives on the **right-hand thumbs-up** pose instead of a left-hand open palm.
- `GestureRecognizer._is_full_reset_pose()` explicitly rejects full reset when the left hand is busy with placement, delete pointing, open-palm rotation prep, victory color toggle, or left thumb gestures.
- This keeps the left-hand interaction namespace focused on placement and edit gestures while still preserving the 5-second loading-circle confirmation.

## Mirroring Architecture

```
RAW frame → MediaPipe → cv2.flip(1) → Draw HUD → Upload texture → Display (no shader mirror)
```

- MediaPipe processes **unmirrored** frame for correct handedness
- Frame is flipped AFTER processing for mirror-view display
- HUD coordinates use mirrored X: `mirror_x = img_width - raw_x`
- Shader `u_mirror_x` is **disabled** (set to 0)
- 3D coordinates use formula `(0.5 - x)` which naturally inverts
