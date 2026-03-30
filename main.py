"""
JARVIS-Style 3D Voxel Editor
============================
Hand gesture controlled 3D voxel editor with holographic visual effects.

SINGLE HAND GESTURES:
- Pinch (index + thumb): Place block (hold 0.5s)
- Peace sign -> Fist: Delete block (hold 1s)
- Pinch + drag: Extend/extrude blocks
- Open palm hold: Open color menu
- Fist hold: Recombine scattered blocks (hold 2s)
- Fast hand spread: Scatter blocks
- Swipe left/right: Undo/Redo

TWO-HAND GESTURES (Phase 2):
- Left any + Right pinch: Place block (more precise)
- Left point + Right pinch: Delete at pointed location
- Left palm + Right fist drag: Pan camera
- Both palms rotate: Rotate selection
- Both pinch + spread: Zoom camera in/out

Keyboard:
- ESC: Quit
- H: Toggle help overlay
- W: Toggle webcam preview
- P: Toggle particles
- M: Toggle symmetry mode
- Shift+X/Y/Z: Set symmetry axis
- S: Save scene (JSON)
- E: Export to OBJ
- L: Load scene
- C: Clear all blocks
- R: Reset camera
- 1-8: Quick color select
- Ctrl+Z: Undo / Ctrl+Shift+Z: Redo
"""

import pygame
from pygame.locals import *
import moderngl
import cv2
import numpy as np
import time
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

from config import CONFIG, EditorMode
from hand_tracker import HandTracker3D
from gestures import GestureRecognizer, GestureState, GestureEvent
from voxel_engine import VoxelEngine
from renderer import Renderer
from ui_renderer import UIRenderer, HelpOverlay, DebugOverlay


class VoxelEditorApp:
    """
    Main application class for the JARVIS Voxel Editor.

    Manages:
    - Pygame/OpenGL window
    - Webcam capture
    - Hand tracking
    - Gesture recognition
    - Voxel engine
    - 3D and UI rendering
    """

    def __init__(self):
        """Initialize the application."""
        print("=" * 60)
        print("JARVIS-STYLE 3D VOXEL EDITOR")
        print("=" * 60)
        print("Initializing...")

        # Initialize Pygame
        pygame.init()
        pygame.display.set_caption(CONFIG.window.TITLE)

        # Create OpenGL window
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 3)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK,
                                       pygame.GL_CONTEXT_PROFILE_CORE)

        flags = DOUBLEBUF | OPENGL
        if CONFIG.window.FULLSCREEN:
            flags |= FULLSCREEN

        self.screen = pygame.display.set_mode(
            (CONFIG.window.WIDTH, CONFIG.window.HEIGHT),
            flags
        )

        # Create ModernGL context
        self.ctx = moderngl.create_context()
        print(f"OpenGL Version: {self.ctx.info['GL_VERSION']}")

        # Initialize webcam
        print("Initializing webcam...")
        self.cap = cv2.VideoCapture(CONFIG.hand_tracking.WEBCAM_INDEX)
        # Request higher resolution (YouTube uses 1280x720)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CONFIG.hand_tracking.WEBCAM_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CONFIG.hand_tracking.WEBCAM_HEIGHT)
        
        # Verify actual resolution
        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"  Requested: {CONFIG.hand_tracking.WEBCAM_WIDTH}x{CONFIG.hand_tracking.WEBCAM_HEIGHT}")
        print(f"  Actual: {actual_width}x{actual_height}")

        if not self.cap.isOpened():
            print("Warning: Failed to open webcam. Running in demo mode.")
            self.webcam_available = False
        else:
            self.webcam_available = True

        # Initialize hand tracker
        print("Initializing hand tracking...")
        self.hand_tracker = HandTracker3D()

        # Initialize gesture recognizer
        self.gesture_recognizer = GestureRecognizer()
        self._setup_gesture_callbacks()

        # Initialize voxel engine
        print("Initializing voxel engine...")
        self.voxel_engine = VoxelEngine()

        # PHASE 13: Connect voxel engine to gesture recognizer for direct delete
        self.gesture_recognizer.set_voxel_engine(self.voxel_engine)

        # DISABLED: Demo scene - start with empty canvas like YouTube reference
        # self._create_demo_scene()

        # Initialize renderer
        print("Initializing 3D renderer...")
        self.renderer = Renderer(self.ctx, CONFIG.window.WIDTH, CONFIG.window.HEIGHT)

        # Initialize UI renderer
        print("Initializing UI renderer...")
        self.ui_renderer = UIRenderer(self.ctx, CONFIG.window.WIDTH, CONFIG.window.HEIGHT)

        # Help overlay
        self.help_overlay = HelpOverlay()

        # Debug overlay (Phase 3)
        self.debug_overlay = DebugOverlay()

        # UI state
        self.show_help = CONFIG.ui.SHOW_HELP_OVERLAY
        self.show_webcam = True
        self.show_particles = True
        self.show_debug = False
        self.running = True
        self.presentation_mode = CONFIG.ui.PRESENTATION_MODE

        # Phase 4: AR Mode (webcam as fullscreen background)
        self.ar_mode = CONFIG.ar.ENABLED

        # Phase 3: Current editor mode
        self.current_mode = EditorMode.BUILD

        # Current cursor state
        self.cursor_world_pos = (0.0, 5.0, 0.0)
        self.left_cursor_world_pos = (0.0, 5.0, 0.0)
        self.current_gesture_progress = 0.0
        
        # PHASE 10: Preview state for renderer-based wireframe (no flickering)
        self.preview_position = None
        self.preview_color = None

        # Phase 2: Two-hand state
        self.two_hands_active = False
        self.symmetry_enabled = False

        # Timing
        self.clock = pygame.time.Clock()
        self.last_time = time.time()
        self.fps = 0
        self.frame_count = 0

        # Webcam frame for display
        self.webcam_frame = None

        # Current hands data
        self.current_hands = []

        # Error tracking for graceful degradation
        self.webcam_error_count = 0
        self.max_webcam_errors = 30  # Allow some frame drops before warning

        print("Initialization complete!")
        print("=" * 60)
        self._print_controls()
        self._set_presentation_mode(self.presentation_mode)

    def _print_controls(self):
        """Print control instructions."""
        print("GESTURE CONTROLS:")
        print("  - Pinch (index + thumb) → Place block (hold 0.5s)")
        print("  - Peace sign → Fist → Delete block (hold 1.0s)")
        print("  - Pinch + drag → Extend/extrude")
        print("  - Open palm (hold) → Color menu")
        print("  - Fast hand spread → SCATTER!")
        print("  - Fist (hold 2s) → Recombine")
        print("")
        print("KEYBOARD:")
        print("  H - Help overlay | W - Webcam | P - Particles")
        print("  D - Debug overlay | F9 - Presentation mode | M - Symmetry mode")
        print("  A - Toggle AR mode (webcam background)")
        print("  S - Save | E - Export OBJ | L - Load | C - Clear")
        print("  R - Reset camera | 1-8 - Colors | ESC - Quit")
        print("")
        print("PHASE 5 FEATURES:")
        print("  G - Gravity burst (explode upward)")
        print("  T - Smooth restore animation")
        print("  O - Cycle colors | I - Disco mode")
        print("")
        print("MODES (Phase 3):")
        print("  F1 - Navigate | F2 - Build | F3 - Erase")
        print("  F4 - Select | F5 - Physics")
        print("=" * 60)

    def _set_mode(self, mode: EditorMode):
        """Set the current editor mode (Phase 3)."""
        self.current_mode = mode
        self.gesture_recognizer.set_mode(mode)

        # Update cursor color based on mode
        mode_colors = {
            EditorMode.NAVIGATE: CONFIG.colors.WHITE,
            EditorMode.BUILD: CONFIG.colors.CYAN,
            EditorMode.ERASE: CONFIG.colors.RED,
            EditorMode.SELECT: CONFIG.colors.YELLOW,
            EditorMode.PHYSICS: CONFIG.colors.ORANGE,
        }
        self.ui_renderer.cursor_color = mode_colors.get(mode, CONFIG.colors.CYAN)

        print(f"Mode: {mode.name}")

    def _set_presentation_mode(self, enabled: bool):
        """Toggle clean demo-friendly UI defaults without affecting core behavior."""
        self.presentation_mode = enabled
        self.hand_tracker.draw_hand_overlay = CONFIG.ui.SHOW_HAND_TRACKING_OVERLAY
        self.hand_tracker.show_hand_labels = CONFIG.ui.SHOW_HAND_LABELS and not enabled

        if enabled:
            self.show_help = False
            self.help_overlay.visible = False
            self.show_debug = False
            self.debug_overlay.visible = False

    def _setup_gesture_callbacks(self):
        """Set up callbacks for gesture events."""
        self.gesture_recognizer.on_gesture_start = self._on_gesture_start
        self.gesture_recognizer.on_gesture_progress = self._on_gesture_progress
        self.gesture_recognizer.on_gesture_complete = self._on_gesture_complete
        self.gesture_recognizer.on_gesture_cancel = self._on_gesture_cancel

    def _on_gesture_start(self, state: GestureState):
        """Called when a gesture starts."""
        # Update cursor color based on gesture
        if state == GestureState.PLACING or state == GestureState.TWO_HAND_PLACING:
            self.ui_renderer.cursor_color = CONFIG.colors.GREEN
            # Show ghost block preview at snapped position
            snapped_pos = self.voxel_engine.snap_to_grid(self.cursor_world_pos)
            self.ui_renderer.show_ghost_block(snapped_pos, self.voxel_engine.current_color)
        elif state == GestureState.DELETING or state == GestureState.TWO_HAND_DELETING:
            self.ui_renderer.cursor_color = CONFIG.colors.RED
            self.ui_renderer.loading_circle_color = CONFIG.colors.RED
        elif state == GestureState.COLOR_MENU:
            self.ui_renderer.cursor_color = CONFIG.colors.PURPLE
        elif state == GestureState.PANNING:
            self.ui_renderer.cursor_color = CONFIG.colors.ORANGE
            self.ui_renderer.show_hand_connection_line(True)
        elif state == GestureState.ZOOMING:
            self.ui_renderer.cursor_color = CONFIG.colors.YELLOW
            self.ui_renderer.show_hand_connection_line(True)
        elif state == GestureState.ROTATING:
            self.ui_renderer.cursor_color = CONFIG.colors.PURPLE
            self.ui_renderer.show_hand_connection_line(True)
        elif state == GestureState.FULL_RESETTING:
            self.ui_renderer.cursor_color = CONFIG.colors.RED
            self.ui_renderer.loading_circle_color = CONFIG.colors.RED
        else:
            self.ui_renderer.cursor_color = CONFIG.colors.CYAN

    def _on_gesture_progress(self, state: GestureState, progress: float):
        """Called during gesture hold."""
        self.current_gesture_progress = progress
        self.ui_renderer.cursor_progress = progress

        # Show loading circle for timed gestures
        if state in (GestureState.PLACING, GestureState.TWO_HAND_PLACING,
                    GestureState.DELETING, GestureState.TWO_HAND_DELETING,
                    GestureState.RECOMBINING, GestureState.FULL_RESETTING):
            loading_pos = self.cursor_world_pos
            if state in (GestureState.DELETING, GestureState.TWO_HAND_DELETING):
                left_hand = self.hand_tracker.get_left_hand()
                if left_hand:
                    loading_pos = left_hand.index_tip_world
            elif state == GestureState.FULL_RESETTING:
                right_hand = self.hand_tracker.get_right_hand()
                if right_hand:
                    loading_pos = right_hand.thumb_tip_world

            self.ui_renderer.show_loading_circle(loading_pos, progress)

            # Update ghost block position during placement
            if state in (GestureState.PLACING, GestureState.TWO_HAND_PLACING):
                snapped_pos = self.voxel_engine.snap_to_grid(self.cursor_world_pos)
                self.ui_renderer.show_ghost_block(snapped_pos, self.voxel_engine.current_color)

    def _on_gesture_complete(self, event: GestureEvent):
        """Called when a gesture is confirmed."""
        # Reset cursor and UI elements
        self.ui_renderer.cursor_progress = 0.0
        self.ui_renderer.cursor_color = CONFIG.colors.CYAN
        self.current_gesture_progress = 0.0
        self.ui_renderer.hide_ghost_block()
        self.ui_renderer.hide_loading_circle()
        self.ui_renderer.show_hand_connection_line(False)
        self.ui_renderer.hide_axis_lock()

        if event.gesture_type == GestureState.PLACING:
            success = self.voxel_engine.place_voxel(event.position)
            if success:
                print(f"Placed voxel at {event.position}")
                # Place symmetry mirror if enabled
                if self.symmetry_enabled:
                    mirror_pos = self.ui_renderer.get_symmetry_position(event.position)
                    self.voxel_engine.place_voxel(mirror_pos)
                    print(f"  + Mirror at {mirror_pos}")

        elif event.gesture_type == GestureState.DELETING:
            # PHASE 13: Simplified - only batch_delete events come through now
            print(f"[DELETE-4] Received DELETING event, extra_data={event.extra_data}")
            if event.extra_data and event.extra_data.get('batch_delete'):
                positions = event.extra_data.get('positions', [])
                print(f"[DELETE-4] BATCH DELETE with {len(positions)} positions: {positions}")
                deleted_count = 0
                voxels_before = len(self.voxel_engine.voxels)
                for grid_pos in positions:
                    # Convert list to tuple if needed (JSON serialization might change it)
                    if isinstance(grid_pos, list):
                        grid_pos = tuple(grid_pos)
                    exists = grid_pos in self.voxel_engine.voxels
                    print(f"[DELETE-5] Deleting at {grid_pos}, exists={exists}")
                    if self.voxel_engine.delete_voxel(grid_pos):
                        deleted_count += 1
                voxels_after = len(self.voxel_engine.voxels)
                print(f"[DELETE-5] Deleted {deleted_count}, voxels: {voxels_before} -> {voxels_after}")
            else:
                # Legacy single delete (fallback)
                grid_pos = self.voxel_engine.world_to_grid(event.position)
                if self.voxel_engine.delete_voxel(grid_pos):
                    print(f"Deleted voxel at {grid_pos}")

        elif event.gesture_type == GestureState.EXTENDING:
            if event.direction:
                length = np.sqrt(sum(d**2 for d in event.direction))
                count = max(1, int(length / CONFIG.voxel.GRID_SIZE))
                created = self.voxel_engine.extend_voxels(event.position, event.direction, count)
                print(f"Extended {created} voxels")

        elif event.gesture_type == GestureState.ROTATING:
            # PHASE 11: Hologram-style continuous rotation with X and Y axes
            is_continuous = event.extra_data and event.extra_data.get('continuous', False)
            
            if is_continuous:
                # Get both rotation deltas
                delta_y = event.extra_data.get('rotation_y', event.rotation_angle)
                delta_x = event.extra_data.get('rotation_x', 0.0)
                # Apply delta rotation to group transform (visual rotation)
                self.voxel_engine.update_group_rotation(delta_y, delta_x)
            else:
                # Legacy: Snap rotation for selected voxels
                if self.voxel_engine.get_selected_count() > 0:
                    self.voxel_engine.rotate_selected(event.rotation_angle)
                    print(f"Rotated selection by {np.degrees(event.rotation_angle):.0f} degrees")
                else:
                    self.voxel_engine.rotate_all(event.rotation_angle)
                    print(f"Rotated all voxels by {np.degrees(event.rotation_angle):.0f} degrees")

        elif event.gesture_type == GestureState.RESETTING:
            # PHASE 9.3: Both fists = reset position and rotation
            self.voxel_engine.reset_group_transform()
        
        elif event.gesture_type == GestureState.FULL_RESETTING:
            # Phase 15: Right thumb up only = clear ALL voxels and reset transform
            voxels_before = len(self.voxel_engine.voxels)
            print(f"[RESET-4] clear_all() called, voxels before: {voxels_before}")
            self.voxel_engine.clear()
            self.voxel_engine.reset_group_transform()
            voxels_after = len(self.voxel_engine.voxels)
            print(f"[RESET-4] FULL RESET complete, voxels after: {voxels_after}")

        elif event.gesture_type == GestureState.COLOR_MENU:
            self.ui_renderer.hide_color_menu()
            palette = CONFIG.colors.get_palette()
            if 0 <= event.selected_color_index < len(palette):
                new_color = palette[event.selected_color_index]
                if self.voxel_engine.get_selected_count() > 0:
                    self.voxel_engine.change_selected_color(new_color)
                else:
                    self.voxel_engine.set_color(new_color)
                print(f"Selected color {event.selected_color_index + 1}")

        elif event.gesture_type == GestureState.SCATTERING:
            self.voxel_engine.scatter(event.position)
            print("SCATTER!")

        elif event.gesture_type == GestureState.RECOMBINING:
            self.voxel_engine.recombine()
            print("Recombining...")

        elif event.gesture_type == GestureState.SWIPING_LEFT:
            if self.voxel_engine.undo():
                print("Undo")

        elif event.gesture_type == GestureState.SWIPING_RIGHT:
            if self.voxel_engine.redo():
                print("Redo")

        elif event.gesture_type == GestureState.SELECTING:
            if event.selection_box:
                min_pos = event.selection_box[:3]
                max_pos = event.selection_box[3:]
                self.voxel_engine.select_in_box(min_pos, max_pos)
                print(f"Selected {self.voxel_engine.get_selected_count()} voxels")

        # YouTube-style continuous building (Phase 6 fix)
        elif event.gesture_type == GestureState.CONTINUOUS_BUILDING:
            extra = event.extra_data
            if extra.get('commit', False):
                # Commit all sketch voxels to permanent
                sketch_positions = extra.get('sketch_positions', [])
                placed_count = 0
                for pos in sketch_positions:
                    if self.voxel_engine.place_voxel(pos, record_history=False):
                        placed_count += 1
                        # Place symmetry mirror if enabled
                        if self.symmetry_enabled:
                            mirror_pos = self.ui_renderer.get_symmetry_position(pos)
                            self.voxel_engine.place_voxel(mirror_pos, record_history=False)
                if placed_count > 0:
                    print(f"Committed {placed_count} voxels")
                # Clear sketch visualization
                self.ui_renderer.clear_sketch_voxels()
            else:
                # Add voxel to sketch (preview) at current position
                snapped_pos = self.voxel_engine.snap_to_grid(event.position)
                grid_key = self.voxel_engine.world_to_grid(snapped_pos)
                
                # Add to gesture recognizer's sketch tracking
                self.gesture_recognizer.sketch_keys.add(snapped_pos)
                
                # Show sketch voxel visualization
                self.ui_renderer.add_sketch_voxel(snapped_pos, self.voxel_engine.current_color)
                
                # Show axis lock indicator if active
                axis = extra.get('active_axis')
                if axis:
                    self.ui_renderer.show_axis_lock(axis)

        # Phase 7: Color toggle and disco mode gestures
        elif event.gesture_type == GestureState.COLOR_TOGGLE:
            # Change to the selected color
            palette = CONFIG.colors.get_palette()
            if 0 <= event.selected_color_index < len(palette):
                new_color = palette[event.selected_color_index]
                self.voxel_engine.set_color(new_color)
                print(f"Color changed to index {event.selected_color_index}")
        
        elif event.gesture_type == GestureState.DISCO_MODE:
            action = (event.extra_data or {}).get("action", "start")

            if action == "start":
                self.voxel_engine.start_disco_mode()
                print("Disco mode started")
            elif action == "freeze":
                self.voxel_engine.freeze_disco_colors()
                print("Disco mode stopped and frozen")
            elif action == "restore":
                self.voxel_engine.restore_original_colors()
                print("Disco mode stopped and original colors restored")
        
        # Phase 8: Grab gesture - move entire voxel structure
        elif event.gesture_type == GestureState.GRABBING:
            # PHASE 9.2: Actually move the voxel group
            if not self.voxel_engine.is_grabbed:
                # Start grab
                self.voxel_engine.start_grab(event.position)
            else:
                # Update grab position
                self.voxel_engine.update_grab(event.position)

        # Phase 2: Camera control gestures
        elif event.gesture_type == GestureState.PANNING:
            # Apply pan to camera
            if event.pan_delta != (0.0, 0.0):
                pan_x, pan_y = event.pan_delta
                # Move camera position and target together
                right = np.cross(
                    self.renderer.camera.target - self.renderer.camera.position,
                    np.array([0, 1, 0])
                )
                right = right / np.linalg.norm(right)
                up = np.array([0, 1, 0])

                move = right * pan_x + up * pan_y
                self.renderer.camera.position += move
                self.renderer.camera.target += move

        elif event.gesture_type == GestureState.ZOOMING:
            # Apply zoom to camera
            if event.zoom_delta != 0.0:
                direction = self.renderer.camera.target - self.renderer.camera.position
                direction = direction / np.linalg.norm(direction)
                zoom_amount = event.zoom_delta * 5.0  # Scale factor

                # Move camera along view direction
                new_pos = self.renderer.camera.position + direction * zoom_amount

                # Limit zoom distance
                dist_to_target = np.linalg.norm(self.renderer.camera.target - new_pos)
                if 2.0 < dist_to_target < 50.0:
                    self.renderer.camera.position = new_pos

    def _on_gesture_cancel(self, state: GestureState):
        """Called when a gesture is cancelled."""
        self.ui_renderer.cursor_progress = 0.0
        self.ui_renderer.cursor_color = CONFIG.colors.CYAN
        self.current_gesture_progress = 0.0
        self.ui_renderer.hide_color_menu()
        self.ui_renderer.hide_ghost_block()
        self.ui_renderer.hide_loading_circle()
        self.ui_renderer.show_hand_connection_line(False)
        self.ui_renderer.hide_axis_lock()
        self.ui_renderer.clear_sketch_voxels()
        
        # PHASE 9.2: End grab when gesture is cancelled
        if state == GestureState.GRABBING and self.voxel_engine.is_grabbed:
            self.voxel_engine.end_grab()

    def _create_demo_scene(self):
        """Create some demo voxels to start with."""
        colors = CONFIG.colors

        # Simple platform
        for x in range(-2, 3):
            for z in range(-2, 3):
                self.voxel_engine.place_voxel(
                    (x * CONFIG.voxel.GRID_SIZE, 0, z * CONFIG.voxel.GRID_SIZE),
                    color=colors.CYAN,
                    record_history=False
                )

        # Some pillars
        for y in range(1, 4):
            self.voxel_engine.place_voxel(
                (-2 * CONFIG.voxel.GRID_SIZE, y * CONFIG.voxel.GRID_SIZE, -2 * CONFIG.voxel.GRID_SIZE),
                color=colors.BLUE,
                record_history=False
            )
            self.voxel_engine.place_voxel(
                (2 * CONFIG.voxel.GRID_SIZE, y * CONFIG.voxel.GRID_SIZE, -2 * CONFIG.voxel.GRID_SIZE),
                color=colors.BLUE,
                record_history=False
            )
            self.voxel_engine.place_voxel(
                (-2 * CONFIG.voxel.GRID_SIZE, y * CONFIG.voxel.GRID_SIZE, 2 * CONFIG.voxel.GRID_SIZE),
                color=colors.BLUE,
                record_history=False
            )
            self.voxel_engine.place_voxel(
                (2 * CONFIG.voxel.GRID_SIZE, y * CONFIG.voxel.GRID_SIZE, 2 * CONFIG.voxel.GRID_SIZE),
                color=colors.BLUE,
                record_history=False
            )

        # Top connector
        for x in range(-2, 3):
            self.voxel_engine.place_voxel(
                (x * CONFIG.voxel.GRID_SIZE, 4 * CONFIG.voxel.GRID_SIZE, -2 * CONFIG.voxel.GRID_SIZE),
                color=colors.PURPLE,
                record_history=False
            )
            self.voxel_engine.place_voxel(
                (x * CONFIG.voxel.GRID_SIZE, 4 * CONFIG.voxel.GRID_SIZE, 2 * CONFIG.voxel.GRID_SIZE),
                color=colors.PURPLE,
                record_history=False
            )

    def run(self):
        """Main application loop."""
        while self.running:
            # Calculate delta time
            current_time = time.time()
            delta_time = current_time - self.last_time
            self.last_time = current_time
            self.frame_count += 1

            # Handle events
            self._handle_events()

            # Capture and process webcam
            if self.webcam_available:
                self._process_webcam()

            # Update systems
            self.voxel_engine.update_physics(delta_time)
            self.ui_renderer.update(delta_time)

            # Update cursor position from hand tracking
            self._update_cursor()

            # Update color menu if active
            if self.gesture_recognizer.is_color_menu_open():
                self.ui_renderer.show_color_menu(
                    self.cursor_world_pos,
                    self.gesture_recognizer.selected_color_index
                )

            # Render
            self._render(delta_time)

            # Swap buffers
            pygame.display.flip()

            # Cap framerate
            self.clock.tick(CONFIG.window.TARGET_FPS)
            self.fps = self.clock.get_fps()

        self._cleanup()

    def _handle_events(self):
        """Handle Pygame events."""
        for event in pygame.event.get():
            if event.type == QUIT:
                self.running = False

            elif event.type == KEYDOWN:
                self._handle_key(event.key)

            elif event.type == VIDEORESIZE:
                self.renderer.resize(event.w, event.h)
                self.ui_renderer.resize(event.w, event.h)

    def _handle_key(self, key):
        """Handle keyboard input."""
        if key == K_ESCAPE:
            self.running = False

        elif key == K_h:
            self.show_help = not self.show_help
            self.help_overlay.visible = self.show_help
            print(f"Help overlay: {'ON' if self.show_help else 'OFF'}")

        elif key == K_F9:
            self._set_presentation_mode(not self.presentation_mode)
            print(f"Presentation mode: {'ON' if self.presentation_mode else 'OFF'}")

        elif key == K_d:
            # Phase 3: Toggle debug overlay
            self.show_debug = self.debug_overlay.toggle()
            print(f"Debug overlay: {'ON' if self.show_debug else 'OFF'}")

        elif key == K_w:
            self.show_webcam = not self.show_webcam
            print(f"Webcam preview: {'ON' if self.show_webcam else 'OFF'}")

        elif key == K_p:
            self.show_particles = not self.show_particles
            print(f"Particles: {'ON' if self.show_particles else 'OFF'}")

        elif key == K_a:
            # Phase 4: Toggle AR mode
            self.ar_mode = not self.ar_mode
            print(f"AR Mode: {'ON - Webcam as background' if self.ar_mode else 'OFF - Dark background'}")

        # Phase 3: Mode switching with F1-F5
        elif key == K_F1:
            self._set_mode(EditorMode.NAVIGATE)
        elif key == K_F2:
            self._set_mode(EditorMode.BUILD)
        elif key == K_F3:
            self._set_mode(EditorMode.ERASE)
        elif key == K_F4:
            self._set_mode(EditorMode.SELECT)
        elif key == K_F5:
            self._set_mode(EditorMode.PHYSICS)

        elif key == K_s:
            if self.voxel_engine.save_to_file("voxel_scene.json"):
                print("Scene saved to voxel_scene.json")
            else:
                print("Failed to save scene")

        elif key == K_e:
            if self.voxel_engine.export_to_obj("voxel_export.obj"):
                print("Exported to voxel_export.obj (and .mtl)")
            else:
                print("Export failed (no voxels?)")

        elif key == K_l:
            if self.voxel_engine.load_from_file("voxel_scene.json"):
                self.voxel_engine.reset_group_transform(log=False)
                print("Scene loaded from voxel_scene.json")
            else:
                print("Failed to load scene")

        elif key == K_c:
            self.voxel_engine.clear(record_history=True)
            self.voxel_engine.reset_group_transform(log=False)
            print("Scene cleared")

        elif key == K_r:
            cam_cfg = CONFIG.camera
            self.renderer.camera.position = np.array(cam_cfg.INITIAL_POSITION, dtype=np.float32)
            self.renderer.camera.target = np.array(cam_cfg.LOOK_AT, dtype=np.float32)
            print("Camera reset")

        elif key == K_m:
            # Toggle symmetry mode (Phase 2)
            self.symmetry_enabled = self.ui_renderer.toggle_symmetry()
            print(f"Symmetry mode: {'ON (' + self.ui_renderer.symmetry_axis.upper() + ' axis)' if self.symmetry_enabled else 'OFF'}")

        elif key == K_x and pygame.key.get_mods() & KMOD_SHIFT:
            # Change symmetry axis to X
            self.ui_renderer.set_symmetry(True, 'x')
            self.symmetry_enabled = True
            print("Symmetry axis: X")

        elif key == K_y and pygame.key.get_mods() & KMOD_SHIFT:
            # Change symmetry axis to Y
            self.ui_renderer.set_symmetry(True, 'y')
            self.symmetry_enabled = True
            print("Symmetry axis: Y")

        elif key == K_z and pygame.key.get_mods() & KMOD_SHIFT and not (pygame.key.get_mods() & KMOD_CTRL):
            # Change symmetry axis to Z (Shift+Z without Ctrl)
            self.ui_renderer.set_symmetry(True, 'z')
            self.symmetry_enabled = True
            print("Symmetry axis: Z")

        elif key == K_z and pygame.key.get_mods() & KMOD_CTRL:
            if pygame.key.get_mods() & KMOD_SHIFT:
                if self.voxel_engine.redo():
                    print("Redo")
            else:
                if self.voxel_engine.undo():
                    print("Undo")

        # Phase 5: Gravity burst (G key)
        elif key == K_g:
            if self.voxel_engine.scatter_state == "normal":
                self.voxel_engine.gravity_burst()
                print("GRAVITY BURST!")
            elif self.voxel_engine.scatter_state in ("gravity_burst", "scattered"):
                self.voxel_engine.restore()
                print("Restoring voxels...")

        # Phase 5: YouTube-style restore (T key)
        elif key == K_t:
            if self.voxel_engine.scatter_state in ("scattered", "gravity_burst"):
                self.voxel_engine.restore()
                print("Smooth restore activated")

        # Phase 5: Cycle colors (O key)
        elif key == K_o:
            new_color = self.voxel_engine.cycle_colors()
            idx = self.voxel_engine.color_cycle_index + 1
            print(f"Color cycled to {idx}")

        # Phase 5: Toggle disco mode (I key)
        elif key == K_i:
            disco_on = self.voxel_engine.toggle_disco_mode()
            print(f"Disco mode: {'ON - Party time!' if disco_on else 'OFF - Frozen colors'}")

        # Quick color selection (1-8)
        elif K_1 <= key <= K_8:
            palette = CONFIG.colors.get_palette()
            color_index = key - K_1
            if color_index < len(palette):
                self.voxel_engine.set_color(palette[color_index])
                print(f"Selected color {color_index + 1}")

    def _process_webcam(self):
        """Capture and process webcam frame."""
        success, frame = self.cap.read()
        if not success:
            self.webcam_error_count += 1
            if self.webcam_error_count >= self.max_webcam_errors:
                self.webcam_available = False
                print("Warning: Webcam feed became unavailable. Falling back to dark background.")
            return
        self.webcam_error_count = 0

        # Process with hand tracker
        frame, hands = self.hand_tracker.process_frame(
            frame,
            draw=self.hand_tracker.draw_hand_overlay
        )
        self.current_hands = hands

        # Update gesture recognizer
        event = self.gesture_recognizer.update(self.hand_tracker, hands)
        if event:
            self._on_gesture_complete(event)

        # Draw gesture state on frame
        state = self.gesture_recognizer.get_state()
        progress = self.gesture_recognizer.get_progress()

        if state != GestureState.IDLE:
            self._draw_loading_circle_cv(frame, hands, progress, state)

        # Draw status info
        self._draw_status_cv(frame)

        # Phase 3: Update debug overlay data
        if self.show_debug:
            gesture_info = self.gesture_recognizer.get_debug_info()
            primary_hand = self.hand_tracker.get_primary_hand()
            snapped = self.voxel_engine.snap_to_grid(self.cursor_world_pos)

            self.debug_overlay.update({
                'mode': self.current_mode.name,
                'gesture_state': gesture_info.get('state', 'IDLE'),
                'gesture_progress': gesture_info.get('progress', 0.0),
                'pinch_distance': gesture_info.get('pinch_distance', 0.0),
                'fingers_up': gesture_info.get('fingers_up', [0, 0, 0, 0, 0]),
                'is_pinching': primary_hand.is_pinching if primary_hand else False,
                'edge_triggered': gesture_info.get('edge_triggered', False),
                'cursor_x': self.cursor_world_pos[0],
                'cursor_y': self.cursor_world_pos[1],
                'cursor_z': self.cursor_world_pos[2],
                'grid_pos': snapped,
                'fps': self.fps,
                'voxel_count': self.voxel_engine.get_voxel_count(),
                'visible_voxels': self.voxel_engine.get_voxel_count(),
                'symmetry': f"{self.ui_renderer.symmetry_axis.upper()}" if self.ui_renderer.symmetry_enabled else "OFF",
                'scatter_state': self.voxel_engine.scatter_state,
                'ar_mode': 'ON' if self.ar_mode else 'OFF',
            })

            # Draw debug info on CV frame as well
            self._draw_debug_cv(frame)

        # Store frame and update texture
        self.webcam_frame = frame
        self.ui_renderer.update_webcam_texture(frame)

    def _update_cursor(self):
        """Update 3D cursor position from hand tracking."""
        # Update right hand cursor (primary)
        right_hand = self.hand_tracker.get_right_hand()
        hand_detected = False
        
        if right_hand:
            self.cursor_world_pos = right_hand.index_tip_world
            self.ui_renderer.set_cursor(
                self.cursor_world_pos,
                progress=self.current_gesture_progress
            )
            self.ui_renderer.right_hand_confidence = right_hand.confidence
            hand_detected = True
        else:
            # Fall back to primary hand if right not available
            primary_hand = self.hand_tracker.get_primary_hand()
            if primary_hand:
                self.cursor_world_pos = primary_hand.index_tip_world
                self.ui_renderer.set_cursor(
                    self.cursor_world_pos,
                    progress=self.current_gesture_progress
                )
                hand_detected = True
        
        # PHASE 12: Preview state for renderer-based wireframe (no flickering)
        # Preview is now rendered in same pass as voxels via renderer.render()
        gesture_state = self.gesture_recognizer.state
        show_preview_states = (
            GestureState.PLACING,
            GestureState.DELETING,
            GestureState.TWO_HAND_PLACING,
            GestureState.TWO_HAND_DELETING,
            GestureState.CONTINUOUS_BUILDING,
            GestureState.GRABBING,  # PHASE 12: Add grab preview
        )
        
        if hand_detected and gesture_state in show_preview_states:
            # GRABBING: Yellow preview showing where structure CENTER will move to
            if gesture_state == GestureState.GRABBING:
                if self.voxel_engine.is_grabbed:
                    # Get bounding box of structure
                    bounds = self.voxel_engine.get_bounding_box()
                    if bounds:
                        # Calculate where structure center will be after grab
                        # Current center + current grab offset = target center
                        current_center = bounds['center']
                        grab_offset = self.voxel_engine.get_snapped_group_offset()
                        target_center = (
                            current_center[0] + grab_offset[0],
                            current_center[1] + grab_offset[1],
                            current_center[2] + grab_offset[2]
                        )
                        self.preview_position = target_center
                        self.preview_color = (1.0, 0.8, 0.0)  # Yellow/Orange for grab
                    else:
                        self.preview_position = None
                        self.preview_color = None
                else:
                    self.preview_position = None
                    self.preview_color = None
            # DELETE states: RED preview at LEFT index position
            elif gesture_state in (GestureState.DELETING, GestureState.TWO_HAND_DELETING):
                # PHASE 12: Use LEFT index for delete cursor
                left_hand = self.hand_tracker.get_left_hand()
                if left_hand:
                    delete_cursor_pos = left_hand.index_tip_world
                    snapped_pos = self.voxel_engine.snap_to_grid(delete_cursor_pos)
                    self.preview_position = snapped_pos
                    self.preview_color = (1.0, 0.2, 0.2)  # Red for delete
                else:
                    self.preview_position = None
                    self.preview_color = None
            # BUILD states: Use current selected color
            else:
                snapped_pos = self.voxel_engine.snap_to_grid(self.cursor_world_pos)
                self.preview_position = snapped_pos
                self.preview_color = self.voxel_engine.current_color
        else:
            self.preview_position = None
            self.preview_color = None
        
        # PHASE 10: Disable old ghost block (now using renderer preview)
        self.ui_renderer.hide_ghost_block()

        # Update left hand cursor (Phase 2)
        left_hand = self.hand_tracker.get_left_hand()
        if left_hand:
            self.left_cursor_world_pos = left_hand.index_tip_world
            self.ui_renderer.set_left_cursor(
                self.left_cursor_world_pos,
                confidence=left_hand.confidence
            )
            self.two_hands_active = True
        else:
            self.ui_renderer.hide_left_cursor()
            self.two_hands_active = False

    def _draw_loading_circle_cv(self, frame, hands, progress: float, state: GestureState):
        """Draw JARVIS-style HUD progress circle (Phase 5).

        Matches YouTube reference implementation with:
        - Progress arc starting from top (-90 degrees)
        - Dashed inner circle
        - State label
        
        CRITICAL: Frame is already flipped (cv2.flip in hand_tracker.py),
        but landmark coordinates are RAW from MediaPipe.
        Must mirror X coordinate to match flipped frame.
        """
        if not hands:
            return

        right_hand = self.hand_tracker.get_right_hand()
        left_hand = self.hand_tracker.get_left_hand()
        primary_hand = self.hand_tracker.get_primary_hand()
        target_hand = primary_hand
        target_landmark_id = 8

        if state == GestureState.FULL_RESETTING and right_hand:
            target_hand = right_hand
            target_landmark_id = 4
        elif state in (GestureState.DELETING, GestureState.TWO_HAND_DELETING) and left_hand:
            target_hand = left_hand

        if not target_hand:
            return

        h, w = frame.shape[:2]
        # Mirror X coordinate: frame is flipped, but landmarks are raw
        center_x = w - int(target_hand.landmarks[target_landmark_id].x)
        center_y = int(target_hand.landmarks[target_landmark_id].y)

        # Determine color based on state (Phase 8: definitive gesture colors)
        state_colors = {
            GestureState.PLACING: (255, 240, 0),       # Cyan (BGR)
            GestureState.DELETING: (51, 51, 255),      # Red (BGR)
            GestureState.GRABBING: (0, 165, 255),      # Orange (BGR)
            GestureState.ROTATING: (255, 240, 0),      # Cyan (BGR)
            GestureState.SCATTER_CHARGING: (255, 0, 255),  # Magenta (BGR)
            GestureState.RESTORE_CHARGING: (0, 255, 0),    # Green (BGR)
            GestureState.RECOMBINING: (0, 255, 0),     # Green (BGR)
            GestureState.TWO_HAND_PLACING: (255, 240, 0),
            GestureState.TWO_HAND_DELETING: (51, 51, 255),
            GestureState.FULL_RESETTING: (51, 51, 255),
            GestureState.COLOR_MENU: (255, 100, 255),  # Purple (disabled)
        }
        color = state_colors.get(state, CONFIG.colors.UI_CYAN)

        # Smaller loading circles for professional appearance
        outer_radius = 22
        inner_radius = 18
        thickness = 2

        # Draw progress arc (from -90 degrees, clockwise)
        start_angle = -90
        end_angle = start_angle + int(360 * progress)
        cv2.ellipse(frame, (center_x, center_y), (outer_radius, outer_radius),
                   0, start_angle, end_angle, color, thickness)

        # Draw dashed inner circle (YouTube reference uses setLineDash([3, 5]))
        # OpenCV doesn't have native dashed lines, so we draw segmented arcs
        num_dashes = 24
        dash_angle = 360 / num_dashes
        for i in range(0, num_dashes, 2):  # Draw every other segment
            dash_start = i * dash_angle
            dash_end = dash_start + (dash_angle * 0.6)  # 60% fill for dash
            cv2.ellipse(frame, (center_x, center_y), (inner_radius, inner_radius),
                       0, dash_start, dash_end, color, 1)

        if (not self.presentation_mode) and CONFIG.ui.SHOW_GESTURE_LABELS:
            cv2.putText(frame, state.name, (center_x - 40, center_y - outer_radius - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    def _get_status_mode_display(self):
        """Return the most useful live mode label and color for the status HUD."""
        gesture_state = self.gesture_recognizer.get_state()
        gesture_labels = {
            GestureState.PLACING: "Build",
            GestureState.CONTINUOUS_BUILDING: "Build",
            GestureState.TWO_HAND_PLACING: "Build",
            GestureState.DELETING: "Delete",
            GestureState.TWO_HAND_DELETING: "Delete",
            GestureState.GRABBING: "Grab",
            GestureState.ROTATING: "Rotate",
            GestureState.PANNING: "Pan",
            GestureState.ZOOMING: "Zoom",
            GestureState.SCATTER_CHARGING: "Scatter",
            GestureState.RESTORE_CHARGING: "Restore",
            GestureState.RECOMBINING: "Recombine",
            GestureState.SELECTING: "Select",
            GestureState.COLOR_MENU: "Color",
            GestureState.FULL_RESETTING: "Reset",
        }
        gesture_colors = {
            GestureState.PLACING: CONFIG.colors.UI_CYAN,
            GestureState.CONTINUOUS_BUILDING: CONFIG.colors.UI_CYAN,
            GestureState.TWO_HAND_PLACING: CONFIG.colors.UI_CYAN,
            GestureState.DELETING: CONFIG.colors.UI_RED,
            GestureState.TWO_HAND_DELETING: CONFIG.colors.UI_RED,
            GestureState.GRABBING: CONFIG.colors.UI_ORANGE,
            GestureState.ROTATING: CONFIG.colors.UI_CYAN,
            GestureState.PANNING: CONFIG.colors.UI_ORANGE,
            GestureState.ZOOMING: (0, 255, 255),
            GestureState.SCATTER_CHARGING: (255, 0, 255),
            GestureState.RESTORE_CHARGING: CONFIG.colors.UI_GREEN,
            GestureState.RECOMBINING: CONFIG.colors.UI_GREEN,
            GestureState.SELECTING: (0, 255, 255),
            GestureState.COLOR_MENU: (255, 100, 255),
            GestureState.FULL_RESETTING: CONFIG.colors.UI_RED,
        }
        if gesture_state != GestureState.IDLE:
            return (
                gesture_labels.get(gesture_state, gesture_state.name.replace("_", " ").title()),
                gesture_colors.get(gesture_state, CONFIG.colors.UI_WHITE),
            )

        scatter_state = self.voxel_engine.scatter_state
        if scatter_state != "normal":
            scatter_labels = {
                "scattered": "Scatter",
                "gravity_burst": "Scatter",
                "restoring": "Restore",
                "recombining": "Recombine",
            }
            scatter_colors = {
                "scattered": CONFIG.colors.UI_ORANGE,
                "gravity_burst": CONFIG.colors.UI_ORANGE,
                "restoring": CONFIG.colors.UI_GREEN,
                "recombining": CONFIG.colors.UI_GREEN,
            }
            return (
                scatter_labels.get(scatter_state, scatter_state.replace("_", " ").title()),
                scatter_colors.get(scatter_state, CONFIG.colors.UI_WHITE),
            )

        if self.voxel_engine.disco_mode:
            return ("Disco", (255, 100, 255))

        mode_labels = {
            EditorMode.NAVIGATE: "Navigate",
            EditorMode.BUILD: "Build",
            EditorMode.ERASE: "Erase",
            EditorMode.SELECT: "Select",
            EditorMode.PHYSICS: "Physics",
        }
        mode_colors = {
            EditorMode.NAVIGATE: CONFIG.colors.UI_WHITE,
            EditorMode.BUILD: CONFIG.colors.UI_CYAN,
            EditorMode.ERASE: CONFIG.colors.UI_RED,
            EditorMode.SELECT: (0, 255, 255),
            EditorMode.PHYSICS: CONFIG.colors.UI_ORANGE,
        }
        return (
            mode_labels.get(self.current_mode, self.current_mode.name.title()),
            mode_colors.get(self.current_mode, CONFIG.colors.UI_WHITE),
        )

    def _draw_status_cv(self, frame):
        """Draw status info on webcam frame."""
        if not CONFIG.ui.SHOW_WEBCAM_STATUS:
            return

        h, w = frame.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        counter_text_thickness = CONFIG.ui.VOXEL_COUNTER_TEXT_THICKNESS
        mode_text_thickness = CONFIG.ui.STATUS_TEXT_THICKNESS
        voxel_text = f"Voxels {self.voxel_engine.get_voxel_count()}"
        mode_text, mode_color = self._get_status_mode_display()

        voxel_text_width = 0
        if CONFIG.ui.SHOW_STATUS_VOXEL_COUNT:
            voxel_text_width = cv2.getTextSize(voxel_text, font, 0.52, counter_text_thickness)[0][0]

        mode_text_width = 0
        if CONFIG.ui.SHOW_STATUS_MODE:
            mode_text_width = cv2.getTextSize(mode_text, font, 0.56, mode_text_thickness)[0][0]

        overlay = frame.copy()
        left_panel_top = h - 46
        left_panel_bottom = h - 10
        left_panel_right = 64
        if CONFIG.ui.SHOW_STATUS_COLOR_SWATCH:
            left_panel_right += 34
        if CONFIG.ui.SHOW_STATUS_VOXEL_COUNT:
            left_panel_right += voxel_text_width + 24
        cv2.rectangle(overlay, (10, left_panel_top), (left_panel_right, left_panel_bottom), (8, 16, 28), -1)

        if CONFIG.ui.SHOW_STATUS_MODE:
            mode_panel_width = max(166, mode_text_width + 36)
            cv2.rectangle(
                overlay,
                (w - 10 - mode_panel_width, left_panel_top),
                (w - 10, left_panel_bottom),
                (8, 16, 28),
                -1,
            )

        cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)

        if CONFIG.ui.SHOW_STATUS_COLOR_SWATCH:
            current_color = self.voxel_engine.current_color
            color_bgr = (
                int(current_color[2] * 255),
                int(current_color[1] * 255),
                int(current_color[0] * 255),
            )
            cv2.rectangle(frame, (18, h - 36), (44, h - 18), color_bgr, -1)
            cv2.rectangle(frame, (18, h - 36), (44, h - 18), CONFIG.colors.UI_WHITE, 1)

        if CONFIG.ui.SHOW_STATUS_VOXEL_COUNT:
            cv2.putText(
                frame,
                voxel_text,
                (56, h - 22),
                font,
                0.52,
                CONFIG.colors.UI_WHITE,
                counter_text_thickness,
            )

        if CONFIG.ui.SHOW_STATUS_SCATTER and self.voxel_engine.scatter_state != "normal":
            cv2.putText(frame, f"[{self.voxel_engine.scatter_state.upper()}]",
                       (w - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                       CONFIG.colors.UI_ORANGE, 2)

        if CONFIG.ui.SHOW_STATUS_MODE:
            mode_panel_width = max(166, mode_text_width + 36)
            cv2.putText(
                frame,
                mode_text,
                (w - 10 - mode_panel_width + 18, h - 22),
                font,
                0.56,
                mode_color,
                mode_text_thickness,
            )

    def _draw_debug_cv(self, frame):
        """Draw debug overlay on webcam frame (Phase 3)."""
        h, w = frame.shape[:2]
        debug_lines = self.debug_overlay.get_lines()

        if not debug_lines:
            return

        # Draw semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (w - 280, 10), (w - 10, 10 + len(debug_lines) * 18 + 10),
                     (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # Draw debug lines
        y = 25
        for line in debug_lines:
            if line.startswith("===") or line.startswith("---"):
                color = CONFIG.colors.UI_CYAN
            elif "True" in line:
                color = CONFIG.colors.UI_GREEN
            elif "False" in line:
                color = CONFIG.colors.UI_RED
            else:
                color = CONFIG.colors.UI_WHITE

            cv2.putText(frame, line, (w - 270, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            y += 18

    def _render(self, delta_time: float):
        """Render the scene."""
        # Get camera matrices
        view = self.renderer.camera.get_view_matrix()
        projection = self.renderer.camera.get_projection_matrix()
        use_ar_background = self.ar_mode and self.webcam_available

        # Phase 4: AR Mode - Render webcam as fullscreen background FIRST
        if use_ar_background:
            self.ui_renderer.render_ar_background(CONFIG.ar.BACKGROUND_DIM)

        # Render 3D scene (voxels, grid, etc.)
        # PHASE 10: Pass preview position/color to renderer (no flickering)
        if use_ar_background:
            self.renderer.render(
                self.voxel_engine, delta_time,
                ar_mode=True,
                voxel_opacity=CONFIG.ar.VOXEL_OPACITY,
                grid_opacity=CONFIG.ar.GRID_OPACITY,
                preview_position=self.preview_position,
                preview_color=self.preview_color
            )
        else:
            self.renderer.render(
                self.voxel_engine, delta_time,
                preview_position=self.preview_position,
                preview_color=self.preview_color
            )

        # Render all UI elements using the unified render method (Phase 2)
        # In AR mode, don't show the small webcam preview (it's fullscreen)
        self.ui_renderer.render(
            view, projection,
            self.renderer.camera.position,
            show_webcam=self.show_webcam and not use_ar_background and self.webcam_available,
            show_particles=self.show_particles,
            ar_mode=use_ar_background,
            ar_dim=CONFIG.ar.BACKGROUND_DIM
        )

    def _cleanup(self):
        """Clean up resources."""
        print("\nCleaning up...")
        self.hand_tracker.release()
        if self.webcam_available:
            self.cap.release()
        self.renderer.cleanup()
        pygame.quit()
        print("Goodbye!")


def main():
    """Main entry point."""
    try:
        app = VoxelEditorApp()
        app.run()
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        pygame.quit()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
