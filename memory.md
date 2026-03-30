# Memory — Known Bugs, Gotchas & Debugging

## Critical Gotchas

### 1. ACTION_COOLDOWN blocks state machine (Phase 13 fix)

**Problem**: Returning ANY event from a gesture handler triggers `last_action_time`, which blocks the state machine for `ACTION_COOLDOWN` (0.3s = ~18 frames at 60fps). During this window, NO state transitions are checked — including pinch release.

**Rule**: During continuous gesture states (like delete collecting), return `None` instead of events. Only return ONE event at the end (e.g., on pinch release).

**Affected code**: `gestures.py` → `_handle_deleting_state()`, `update()` lines 329-331.

### 2. Gesture system has a direct ref to voxel_engine

`GestureRecognizer._voxel_engine` is set via `set_voxel_engine()` in `main.py`. This is intentional — the delete system needs to check `grid_pos in self._voxel_engine.voxels` directly without going through events. Don't remove this coupling.

### 3. cv2.flip happens AFTER MediaPipe

Frame is flipped horizontally in `hand_tracker.py` AFTER MediaPipe processes the raw frame. This means:
- MediaPipe landmarks are in RAW (unmirrored) coordinates
- HUD drawing uses `mirror_x = img_width - raw_x`
- Shader `u_mirror_x` is OFF (set to 0.0)
- If you re-enable shader mirroring, you'll get double-mirror (broken)

### 4. Z is always 0

All world coordinates have `z = 0.0`. The YouTube reference uses 2D placement only. If you change this, you'll break block alignment, delete cursor matching, and grid snapping.

### 5. Grid keys are integer tuples

Voxel dict keys are `(int, int, int)`. Watch for float→int conversion issues. `world_to_grid()` does `int(round(snapped / grid_size))`. If you pass world coordinates as keys directly, they won't match.

### 6. NoneType on hand properties

Always null-check before accessing `.fingers_up`, `.is_pinching`, `.landmarks`, etc. Hands can disappear between frames. Pattern:
```python
if left_hand is None or right_hand is None:
    self._transition_to(GestureState.IDLE)
    return None
```

### 7. AR mode skips grid rendering

Grid is NOT rendered in AR mode (`renderer.py`). This prevents flickering caused by grid transparency interactions with the webcam background.

### 8. Bloom shader must preserve alpha

`bloom_combine.frag` uses `frag_color = vec4(final_color, scene_color.a)` — NOT `alpha=1.0`. In AR mode, transparent pixels must stay transparent so the webcam shows through.

### 9. Edge-triggered gestures need cooldown

Color toggle (left victory) uses deliberate close→open edge detection AND a 500ms cooldown. Without both, it triggers on hand entry/exit from frame.

### 10. Full reset must stay out of the left-hand gesture namespace (Phase 15 fix)

Full reset is intentionally bound to **right thumb up only**. The recognizer should reject that pose whenever the left hand is actively pinching, pointing, holding an open palm, showing victory, or using a thumb gesture. This prevents full reset from stealing the left-hand placement flow.

## Debug Logging Patterns

All debug prints follow numbered patterns:

**Deletion**: `[DELETE-1]` through `[DELETE-5]`
- 1: Enter state / mode active
- 2: Block marked for deletion
- 3: Pinch released, batch event generated
- 4: Event received in main.py
- 5: Individual delete (before/after count)

**Reset**: `[RESET-1]` through `[RESET-4]`
- 1: Timer started
- 2: Progress (elapsed/total)
- 3: Timer complete
- 4: Handler called (before/after voxel count)

**Gestures**: `[GESTURE] ...` for state transitions.
**Voxel Engine**: `[VOXEL] ...` for grab/transform events.

## Common Debugging Steps

### "Gesture not triggering"
1. Check console for `[GESTURE]` prints — is the state entering?
2. Check `ACTION_COOLDOWN` — is another event blocking it?
3. Check priority order — is a higher-priority gesture consuming the input?
4. Press D for debug overlay — shows fingers_up, pinch state, mode

### "Blocks not appearing/disappearing"
1. Check voxel count in debug overlay
2. Check `[DELETE-5]` prints — does "exists" show True?
3. Check grid key format — `(int, int, int)` not `(float, float, float)`
4. After delete, renderer auto-rebuilds buffer on next frame

### "Crash on hand property access"
Add null check: `if hand is None: return None` before any `.fingers_up`, `.is_pinching`, `.landmarks` access.

### "Flickering in AR mode"
- Grid rendering must be skipped
- Bloom combine must early-out on transparent pixels
- Preview must render in same pass as voxels (not separate UI pass)

## Gesture Hold Times (config.py)

| Gesture | Hold Time | Config Key |
|---------|-----------|------------|
| Place | 0.5s | PLACE_HOLD_TIME |
| Delete | 0.5s | DELETE_HOLD_TIME |
| Grab | 0.5s | GRAB_HOLD_TIME |
| Rotate | 1.0s | ROTATE_HOLD_TIME |
| Scatter | 0.8s | SCATTER_HOLD_TIME |
| Restore | 0.8s | RESTORE_HOLD_TIME |
| Full Reset | 5.0s | FULL_RESET_HOLD_TIME |

## Rotation Parameters (gestures.py)

```python
NEUTRAL_SPREAD = 0.4   # Expected horizontal distance between hands
DEAD_ZONE = 0.03       # Movement threshold
SENSITIVITY = 0.025    # Speed multiplier
SMOOTH_FACTOR = 0.2    # Exponential moving average weight
MAX_SPEED = 0.04       # Clamp per frame
```
