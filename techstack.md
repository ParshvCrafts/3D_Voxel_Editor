# Tech Stack & Setup

## Dependencies

Pinned in `requirements.txt` (tested on Python 3.11.9):

```
mediapipe==0.10.8                # Hand tracking
opencv-contrib-python==4.11.0.86 # Webcam, image processing (mediapipe's own OpenCV dep)
moderngl==5.12.0                 # OpenGL 3.3 rendering
pygame==2.6.1                    # Window, input, event loop
pyrr==0.10.3                     # Matrix/vector math
numpy==1.26.4                    # Numerical operations
```

Never add `opencv-python` / `opencv-python-headless` alongside: they overwrite the same `cv2` module.

README images: `python tools/capture_readme_assets.py` renders them through the real app with no webcam.

## Running

```bash
cd "3D Voxel Editor"
pip install -r requirements.txt
python main.py
```

## System Requirements

- Webcam (1280x720 requested, falls back to 640x480)
- OpenGL 3.3+ support
- Python 3.11 (mediapipe 0.10.8 ships wheels for 3.8–3.11 only)

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
