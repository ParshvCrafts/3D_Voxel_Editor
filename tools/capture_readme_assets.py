"""Render README screenshots from the real app, without a webcam.

usage: python tools/capture_readme_assets.py [out_dir]

Runs the actual VoxelEditorApp render loop (renderer, bloom, grid, HUD) with the
camera forced off, so the app takes its documented "demo mode" path. A small
scene is placed through VoxelEngine.place_voxel and the group rotation is set
through update_group_rotation, the same calls the gestures make. The OpenGL
back buffer is read after the frame is drawn and saved as PNG.

No hand is in frame, so these images show renderer output only; they are not a
substitute for a recording of the gesture workflow.
"""
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)


class NoCamera:
    def __init__(self, *a): pass
    def set(self, *a): return False
    def get(self, *a): return 0
    def isOpened(self): return False
    def read(self): return False, None
    def release(self): pass


cv2.VideoCapture = NoCamera

import pygame  # noqa: E402
from config import CONFIG  # noqa: E402
import main  # noqa: E402

# A pixel-art rocket on the z = 0 placement plane, plus a second layer behind it
# so the rotated shot shows depth. Rows read top to bottom; letters map to palette.
ROCKET = [
    "....R....",
    "...RWR...",
    "..BWWWB..",
    "..BWCWB..",
    "..BWWWB..",
    "..BWWWB..",
    ".PBWWWBP.",
    "PPBBBBBPP",
    "P..OYO..P",
    "....O....",
]
# Colors from the app's own palette; one WHITE voxel as a lit window.
_c = CONFIG.colors
PALETTE = {
    "R": _c.PINK, "W": _c.CYAN, "B": _c.BLUE, "C": _c.WHITE,
    "P": _c.PURPLE, "O": _c.ORANGE, "Y": _c.GOLD,
}


def build_scene(engine):
    g = CONFIG.voxel.GRID_SIZE
    h = len(ROCKET)
    for depth in (0, -1):
        for row, line in enumerate(ROCKET):
            for col, ch in enumerate(line):
                if ch == ".":
                    continue
                if depth and ch not in "BW":
                    continue
                x = (col - len(line) // 2) * g
                y = (h - row) * g
                engine.place_voxel((x, y, depth * g), color=PALETTE[ch], record_history=False)


def grab(app):
    w, h = CONFIG.window.WIDTH, CONFIG.window.HEIGHT
    raw = app.ctx.screen.read(viewport=(0, 0, w, h), components=3)
    img = np.frombuffer(raw, dtype=np.uint8).reshape(h, w, 3)[::-1]
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def run_frames(app, n):
    for _ in range(n):
        app._handle_events()
        app.voxel_engine.update_physics(1 / 60)
        app.ui_renderer.update(1 / 60)
        app._render(1 / 60)
        frame = grab(app)
        pygame.display.flip()
        app.clock.tick(60)
    return frame


def save(path, img):
    cv2.imwrite(path, img, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    print(f"wrote {path} ({os.path.getsize(path) // 1024} KB)")


def crop(img, x0=380, y0=90, w=520, h=440):
    """The model sits in the middle of a mostly empty 1280x720 frame."""
    return img[y0:y0 + h, x0:x0 + w]


def social_preview(path, shot):
    """1280x640 card for GitHub Settings -> Social preview."""
    from PIL import Image, ImageDraw, ImageFont

    bg = (3, 8, 18)
    card = Image.new("RGB", (1280, 640), bg)
    model = crop(shot, 330, 60, 620, 560).astype(np.float32)
    # Fade the crop's edges into the card background so no hard seam shows.
    h, w = model.shape[:2]
    fx = np.clip(np.minimum(np.arange(w), w - 1 - np.arange(w)) / 120, 0, 1)
    fy = np.clip(np.minimum(np.arange(h), h - 1 - np.arange(h)) / 90, 0, 1)
    a = (fy[:, None] * fx[None, :])[..., None]
    model = (model * a + np.array(bg[::-1], np.float32) * (1 - a)).astype(np.uint8)
    card.paste(Image.fromarray(cv2.cvtColor(model, cv2.COLOR_BGR2RGB)), (640, 40))

    def font(size, bold=False):
        for name in (("segoeuib.ttf" if bold else "segoeui.ttf"), "arial.ttf", "DejaVuSans.ttf"):
            try:
                return ImageFont.truetype(name, size)
            except OSError:
                continue
        return ImageFont.load_default()

    d = ImageDraw.Draw(card)
    d.text((70, 200), "3D Voxel Editor", font=font(64, True), fill=(235, 245, 255))
    d.text((72, 290), "Build voxel models with your hands.", font=font(32), fill=(0, 212, 255))
    d.text((72, 340), "Webcam + MediaPipe hand tracking,", font=font(26), fill=(150, 170, 190))
    d.text((72, 376), "OpenGL rendering with ModernGL.", font=font(26), fill=(150, 170, 190))
    card.save(path, optimize=True)
    print(f"wrote {path} ({os.path.getsize(path) // 1024} KB)")


def write_gif(path, frames, fps=15):
    from PIL import Image

    imgs = [Image.fromarray(cv2.cvtColor(cv2.resize(f, (450, 350), interpolation=cv2.INTER_AREA),
                                         cv2.COLOR_BGR2RGB)) for f in frames]
    pal = [im.quantize(colors=128, method=Image.Quantize.MEDIANCUT) for im in imgs]
    pal[0].save(path, save_all=True, append_images=pal[1:], duration=int(1000 / fps), loop=0, optimize=True)
    print(f"wrote {path} ({os.path.getsize(path) // 1024} KB, {len(frames)} frames)")


def main_capture(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    app = main.VoxelEditorApp()
    # Framing only: pull the camera back so the whole 10-row model fits.
    app.renderer.camera.position = np.array([0.0, 7.0, 24.0], dtype=np.float32)
    app.renderer.camera.target = np.array([0.0, 4.5, 0.0], dtype=np.float32)
    build_scene(app.voxel_engine)
    engine = app.voxel_engine
    print(f"voxels placed: {engine.get_voxel_count()}")

    front = crop(run_frames(app, 45))

    engine.update_group_rotation(0.6, 0.25)
    rotated_full = run_frames(app, 20)
    rotated = crop(rotated_full)

    engine.reset_group_transform(log=False)
    engine.scatter((0.0, 5.0, 0.0))
    scatter = crop(run_frames(app, 10))

    gap = np.full((front.shape[0], 8, 3), 18, np.uint8)
    save(os.path.join(out_dir, "hero-strip.png"), np.hstack([front, gap, rotated, gap, scatter]))
    social_preview(os.path.join(out_dir, "social-preview.png"), rotated_full)

    # GIF: settle, spin one full turn, scatter, restore. 2 render frames per GIF frame.
    engine.clear()
    build_scene(engine)
    run_frames(app, 30)
    gif = []
    wide = dict(x0=280, y0=40, w=720, h=560)
    for _ in range(35):  # one full turn
        engine.update_group_rotation(2 * np.pi / 35, 0.0)
        gif.append(crop(run_frames(app, 2), **wide))
    engine.reset_group_transform(log=False)
    engine.scatter((0.0, 5.0, 0.0))
    for _ in range(18):
        gif.append(crop(run_frames(app, 2), **wide))
    engine.restore()
    for _ in range(40):
        gif.append(crop(run_frames(app, 2), **wide))
    write_gif(os.path.join(out_dir, "demo-render.gif"), gif)

    app._cleanup()


if __name__ == "__main__":
    main_capture(sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "docs", "assets"))
