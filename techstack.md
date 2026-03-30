# Tech Stack & Setup

## Dependencies

```
mediapipe >= 0.10.0    # Hand tracking
opencv-python >= 4.8.0 # Webcam, image processing
moderngl >= 5.8.0      # OpenGL 3.3 rendering
pygame >= 2.5.0        # Window, input, event loop
pyrr >= 0.10.3         # Matrix/vector math
numpy >= 1.24.0        # Numerical operations
```

## Running

```bash
cd "3D Voxel Editor"
pip install -r requirements.txt
python main.py
```

## System Requirements

- Webcam (1280x720 requested, falls back to 640x480)
- OpenGL 3.3+ support
- Python 3.8+

## Key Config Values (`config.py`)

| Parameter | Value | Notes |
|-----------|-------|-------|
| Window | 1280×720 | 60 FPS target |
| Grid size | 1.0 | World units per voxel |
| Max voxels | 10,000 | Performance limit |
| Pinch threshold | 40px / 60px release | Hysteresis |
| Action cooldown | 0.3s | Between gesture events |
| Full reset hold | 5.0s | `FULL_RESET_HOLD_TIME`, right thumb up only |
| AR mode | Enabled by default | Webcam as background |
| Coordinate mapping | `(0.5 - x) * 18`, `(0.5 - y) * 12` | YouTube-style |
| Camera position | (0, 5, 15) | Looking at origin |
| FOV | 50° | Slightly wide |

## Shader Files

| Shader | Purpose |
|--------|---------|
| `voxel.vert/frag` | Instanced voxel cubes with Fresnel glow |
| `wireframe.vert/frag` | Edge overlay, preview cubes |
| `grid.vert/frag` | Floor grid with distance fade |
| `cursor.vert/frag` | 3D cursor rings |
| `particle.vert/frag` | Billboard ambient particles |
| `hud.vert/frag` | Webcam texture, 2D overlays |
| `bloom_extract/blur/combine.frag` | Post-processing bloom pipeline |

## Instance Data Layout (per voxel)

```
[position.x, position.y, position.z,   // 3 floats
 color.r, color.g, color.b,            // 3 floats
 rotation.x, rotation.y, rotation.z,   // 3 floats
 selected]                             // 1 float (10 floats total)
```
