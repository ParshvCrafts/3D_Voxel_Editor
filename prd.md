# Product Requirements

## Gesture Controls

### Gesture Priority Table (checked in this exact order)

| Priority | Gesture | Hands | Detection | Timer | Action |
|----------|---------|-------|-----------|-------|--------|
| 1 | **RESET** | Both fists | All fingers curled, both hands | 1s | Reset position/rotation to origin |
| 2 | **ROTATE** | Both palms | 4+ fingers extended, both hands | 1s → continuous | Hologram-style rotation |
| 2.5 | **FULL RESET** | R thumb up only | R thumb up (`[1,0,0,0,0]`), L hand must stay out of active gestures (no pinch/point/palm/victory/thumb gesture) | 5s | Clear ALL voxels + reset transform |
| 3 | **DELETE** | R pinch + L point | R thumb+index touch, L index up + middle DOWN | 0.5s → paint | Delete blocks under L index |
| 4 | **GRAB** | R fist alone | R all fingers curled, L not pinching | 0.5s → drag | Move entire structure |
| 5 | **PLACE** | L pinch only | L thumb+index touch, R NOT pinching | 0.5s → continuous | Place blocks |
| 6 | **SCATTER** | L thumb down | Thumb below wrist, others curled | 0.8s (one-shot) | Explode voxels |
| 7 | **RESTORE** | L thumb up | Thumb above wrist, others curled | 0.8s (one-shot) | Restore positions |

### Edge-Triggered Gestures (instant, no timer)

| Gesture | Hand | Detection | Action |
|---------|------|-----------|--------|
| **Color Toggle** | L victory | Index+middle up, ring+pinky down (close→open only) | Cycle color palette |
| **Disco Start** | R victory | Index+middle up, ring+pinky down | Random colors each frame |
| **Disco Stop** | R palm | All fingers extended | Stop disco mode |

### Rotation Control

- **Activate**: Both palms open for 1 second
- **Y-axis (spin)**: Spread hands apart horizontally
- **X-axis (tilt)**: Move one hand higher than the other
- **Stop**: Close one or both palms
- **Parameters**: sensitivity=0.025, smoothing=0.2, max_speed=0.04, dead_zone=0.03

### Full Reset Safety Rule

- Full reset is intentionally isolated to the **right-hand thumbs-up** pose.
- The **left hand must stay neutral** while charging full reset, so left pinch placing and left-point delete guidance keep priority.
- The 5-second confirmation uses the same loading-circle timer system as other hold gestures.

### Delete Flow

1. Right pinch + left index pointing → enter DELETING state
2. Hold 0.5s → DELETE mode active
3. Move left index over blocks → blocks collected in set (directly checked against voxel dict)
4. Release right pinch → batch delete all collected blocks
5. No events during collecting (avoids cooldown deadlock)

## Keyboard Controls

| Key | Action | Key | Action |
|-----|--------|-----|--------|
| ESC | Quit | H | Toggle help overlay |
| W | Toggle webcam preview | D | Toggle debug overlay |
| P | Toggle particles | A | Toggle AR mode |
| M | Toggle symmetry | Shift+X/Y/Z | Set symmetry axis |
| S | Save scene (JSON) | L | Load scene |
| E | Export to OBJ | C | Clear all blocks |
| R | Reset camera | 1-8 | Quick color select |
| Ctrl+Z | Undo | Ctrl+Shift+Z | Redo |
| F1-F5 | Mode: Navigate/Build/Erase/Select/Physics |
| G | Gravity burst / Restore | T | Smooth restore |
| O | Cycle colors | I | Toggle disco mode |

## Finger Detection Logic

```python
# Thumb: X-axis based, handedness-aware
if hand_type == "Right":
    thumb_extended = thumb_tip.x < thumb_ip.x  # Points LEFT
else:
    thumb_extended = thumb_tip.x > thumb_ip.x  # Points RIGHT

# Other fingers: Y-axis based (Y=0 at top)
extended = tip.y < pip.y  # Tip ABOVE pip = extended

# Fist: sum(fingers_up) == 0
# Palm open: sum(fingers_up) >= 4
# Victory: index=1, middle=1, ring=0, pinky=0
# Pointing: index=1, middle=0
```

## Color Palette

8 colors accessible via keys 1-8 or left victory gesture:
1. Cyan `(0.0, 0.831, 1.0)`
2. Blue `(0.0, 0.478, 1.0)`
3. White `(0.9, 0.95, 1.0)`
4. Green `(0.0, 1.0, 0.4)`
5. Purple `(0.6, 0.2, 1.0)`
6. Pink `(1.0, 0.2, 0.6)`
7. Orange `(1.0, 0.4, 0.0)`
8. Gold `(1.0, 0.84, 0.0)`
