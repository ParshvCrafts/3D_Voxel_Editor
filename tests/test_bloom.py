import os
import sys
import unittest

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import moderngl  # noqa: E402

from config import CONFIG  # noqa: E402
from voxel_engine import VoxelEngine  # noqa: E402

try:
    CTX = moderngl.create_standalone_context()
except Exception as exc:  # no GPU / driver on this machine
    CTX = None
    NO_GL_REASON = str(exc)


@unittest.skipIf(CTX is None, "no OpenGL 3.3 context available")
class BloomBufferTests(unittest.TestCase):
    """Bloom must reflect the current frame only.

    Regression: alpha blending stayed on during the bloom passes, so dark
    pixels (written as vec4(0)) never replaced old bright ones. A white voxel
    that moved left a permanent, growing glow trail behind it.
    """

    def setUp(self):
        from renderer import Renderer

        self.renderer = Renderer(CTX, 640, 360)
        self.renderer._render_final_composite = lambda ar_mode=False: None  # no window in tests
        self.engine = VoxelEngine()

    def render(self, frames=3):
        for _ in range(frames):
            self.renderer.render(self.engine, 1 / 60)

    def bright_columns(self):
        """Columns (in half-res bloom space) that hold any extracted bright pixel."""
        raw = self.renderer.bloom_texture.read()
        img = np.frombuffer(raw, np.uint8).reshape(180, 320, 4)[:, :, :3]
        return np.flatnonzero(img.max(axis=(0, 2)) > 0)

    def test_bright_voxel_leaves_no_trail_after_moving(self):
        white = CONFIG.colors.WHITE
        self.engine.place_voxel((-6.0, 5.0, 0.0), color=white, record_history=False)
        self.render()
        before = self.bright_columns()
        self.assertTrue(len(before) and before.max() < 160, "white voxel should bloom on the left half")

        self.engine.delete_voxel(self.engine.world_to_grid((-6.0, 5.0, 0.0)), record_history=False)
        self.engine.place_voxel((6.0, 5.0, 0.0), color=white, record_history=False)
        self.render()
        after = self.bright_columns()

        self.assertTrue(len(after) and after.min() >= 160, "moved voxel should bloom on the right half")
        self.assertEqual(int((after < 160).sum()), 0, "old position still glows: bloom buffer kept a stale frame")


if __name__ == "__main__":
    unittest.main()
