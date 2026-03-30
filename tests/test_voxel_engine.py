import json
import os
import sys
import tempfile
import unittest


PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


from voxel_engine import VoxelEngine  # noqa: E402


class VoxelEngineTests(unittest.TestCase):
    def test_clear_records_history_and_resets_runtime_state(self):
        engine = VoxelEngine()
        engine.place_voxel((0.0, 0.0, 0.0))
        engine.select_voxel((0, 0, 0))
        engine.scatter_state = "scattered"
        engine.disco_mode = True
        engine.group_offset = (2.0, 0.0, 0.0)
        engine.group_rotation = 1.25

        engine.clear(record_history=True)

        self.assertEqual(engine.get_voxel_count(), 0)
        self.assertEqual(engine.scatter_state, "normal")
        self.assertFalse(engine.disco_mode)
        self.assertEqual(engine.group_offset, (0.0, 0.0, 0.0))
        self.assertEqual(engine.group_rotation, 0.0)
        self.assertEqual(engine.group_rotation_x, 0.0)
        self.assertEqual(engine.undo_stack[-1].action_type, "delete")

        self.assertTrue(engine.undo())
        self.assertIn((0, 0, 0), engine.voxels)

    def test_grab_move_snaps_to_grid_and_supports_undo_redo(self):
        engine = VoxelEngine()
        engine.place_voxel((0.0, 0.0, 0.0))
        engine.place_voxel((1.0, 0.0, 0.0))

        engine.start_grab((0.0, 0.0, 0.0))
        engine.group_offset = (0.6, 0.4, 0.0)
        engine.end_grab()

        self.assertEqual(sorted(engine.voxels.keys()), [(1, 0, 0), (2, 0, 0)])

        self.assertTrue(engine.undo())
        self.assertEqual(sorted(engine.voxels.keys()), [(0, 0, 0), (1, 0, 0)])

        self.assertTrue(engine.redo())
        self.assertEqual(sorted(engine.voxels.keys()), [(1, 0, 0), (2, 0, 0)])

    def test_save_and_load_scene_round_trip(self):
        engine = VoxelEngine()
        engine.set_color((0.1, 0.2, 0.3))
        engine.place_voxel((0.0, 0.0, 0.0))
        engine.place_voxel((1.0, 0.0, 0.0), color=(0.9, 0.4, 0.2))
        engine.voxels[(1, 0, 0)].color = (0.2, 0.8, 0.5)

        with tempfile.TemporaryDirectory() as tmpdir:
            scene_path = os.path.join(tmpdir, "scene.json")
            self.assertTrue(engine.save_to_file(scene_path))

            loaded = VoxelEngine()
            self.assertTrue(loaded.load_from_file(scene_path))
            self.assertEqual(loaded.get_voxel_count(), 2)
            self.assertEqual(loaded.current_color, (0.1, 0.2, 0.3))
            self.assertEqual(loaded.group_offset, (0.0, 0.0, 0.0))
            self.assertIn((0, 0, 0), loaded.voxels)
            self.assertEqual(loaded.voxels[(1, 0, 0)].color, (0.2, 0.8, 0.5))
            self.assertEqual(loaded.voxels[(1, 0, 0)].authored_color, (0.9, 0.4, 0.2))

    def test_disco_can_freeze_visible_colors_and_restore_authored_colors(self):
        engine = VoxelEngine()
        engine.place_voxel((0.0, 0.0, 0.0), color=(0.2, 0.3, 0.4))
        engine.place_voxel((1.0, 0.0, 0.0), color=(0.7, 0.1, 0.2))

        engine.start_disco_mode()
        engine.voxels[(0, 0, 0)].color = (0.9, 0.8, 0.1)
        engine.voxels[(1, 0, 0)].color = (0.1, 0.9, 0.8)

        self.assertTrue(engine.freeze_disco_colors())
        self.assertFalse(engine.disco_mode)
        self.assertEqual(engine.voxels[(0, 0, 0)].color, (0.9, 0.8, 0.1))
        self.assertEqual(engine.voxels[(1, 0, 0)].color, (0.1, 0.9, 0.8))

        engine.start_disco_mode()
        self.assertTrue(engine.disco_mode)
        self.assertTrue(engine.restore_original_colors())
        self.assertFalse(engine.disco_mode)
        self.assertEqual(engine.voxels[(0, 0, 0)].color, (0.2, 0.3, 0.4))
        self.assertEqual(engine.voxels[(1, 0, 0)].color, (0.7, 0.1, 0.2))

    def test_load_invalid_scene_returns_false(self):
        engine = VoxelEngine()

        with tempfile.TemporaryDirectory() as tmpdir:
            scene_path = os.path.join(tmpdir, "invalid.json")
            with open(scene_path, "w", encoding="utf-8") as handle:
                json.dump({"voxels": "not-a-list"}, handle)

            self.assertFalse(engine.load_from_file(scene_path))


if __name__ == "__main__":
    unittest.main()
