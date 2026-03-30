"""
Voxel Engine Module for JARVIS Voxel Editor
============================================
Handles voxel data structures, operations, and physics simulation.
"""

import numpy as np
import math
import time
import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Set
from enum import Enum, auto
from utils import (
    snap_to_grid_3d, lerp3, ease_in_out, distance_3d,
    rotate_point_around_center, Ray, ray_box_intersection
)
from config import CONFIG


class VoxelState(Enum):
    """State of a voxel."""
    STATIC = auto()      # Normal, in-place
    DYNAMIC = auto()     # Physics-enabled (scattered)
    ANIMATING = auto()   # Animating back to position


@dataclass
class Voxel:
    """A single voxel in the world."""
    position: Tuple[float, float, float]  # Current position
    original_position: Tuple[float, float, float]  # Position before scatter
    color: Tuple[float, float, float]  # RGB color (0-1)
    authored_color: Tuple[float, float, float]  # Intended user-chosen color
    state: VoxelState = VoxelState.STATIC

    # Physics properties (used when state is DYNAMIC)
    velocity: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotation: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    angular_velocity: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    # Animation properties (used when state is ANIMATING)
    animation_start_pos: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    animation_progress: float = 0.0
    animation_delay: float = 0.0  # Staggered start time

    # Selection
    is_selected: bool = False


@dataclass
class HistoryAction:
    """An action that can be undone/redone."""
    action_type: str  # "place", "delete", "move", "color_change"
    voxels_data: List[Dict] = field(default_factory=list)  # Voxel state before action
    new_voxels_data: List[Dict] = field(default_factory=list)  # Voxel state after action


class VoxelEngine:
    """
    Core voxel engine with physics simulation.

    Features:
    - Sparse voxel storage using dictionary
    - Grid snapping
    - Selection system
    - Undo/redo history
    - Scatter/recombine physics
    - Ray casting for cursor interaction
    """

    def __init__(self):
        """Initialize the voxel engine."""
        # Voxel storage: key is (grid_x, grid_y, grid_z) tuple
        self.voxels: Dict[Tuple[int, int, int], Voxel] = {}

        # Selection
        self.selected_voxels: Set[Tuple[int, int, int]] = set()

        # Current color for new voxels
        self.current_color = CONFIG.colors.CYAN

        # History for undo/redo
        self.undo_stack: List[HistoryAction] = []
        self.redo_stack: List[HistoryAction] = []
        self.max_history = CONFIG.history.MAX_UNDO_STEPS

        # Scatter/recombine state
        self.scatter_state = "normal"  # "normal", "scattered", "recombining", "gravity_burst", "restoring"
        self.recombine_start_time = 0.0

        # Phase 5: YouTube-style restore mode
        self.restore_mode = False  # Use smooth lerp-based restore

        # Phase 5: Color cycling and disco mode
        self.color_cycle_index = 0
        self.disco_mode = False

        # Grid configuration
        self.grid_size = CONFIG.voxel.GRID_SIZE

        # Physics config
        self.physics_cfg = CONFIG.physics
        
        # Phase 9.2: Group transform for grab/rotate (matches YouTube approach)
        # Instead of moving individual voxels, we move the entire group
        self.group_offset = (0.0, 0.0, 0.0)  # Translation offset from grab
        self.group_rotation = 0.0  # Y-axis rotation in radians
        self.grab_start_pos = None  # Hand position when grab started
        self.is_grabbed = False  # Currently being grabbed

    def world_to_grid(self, world_pos: Tuple[float, float, float]) -> Tuple[int, int, int]:
        """Convert world coordinates to grid coordinates."""
        snapped = snap_to_grid_3d(world_pos, self.grid_size)
        return (
            int(round(snapped[0] / self.grid_size)),
            int(round(snapped[1] / self.grid_size)),
            int(round(snapped[2] / self.grid_size))
        )

    def grid_to_world(self, grid_pos: Tuple[int, int, int]) -> Tuple[float, float, float]:
        """Convert grid coordinates to world coordinates."""
        return (
            grid_pos[0] * self.grid_size,
            grid_pos[1] * self.grid_size,
            grid_pos[2] * self.grid_size
        )

    def snap_to_grid(self, world_pos: Tuple[float, float, float]) -> Tuple[float, float, float]:
        """Snap world coordinates to the nearest grid position."""
        return snap_to_grid_3d(world_pos, self.grid_size)

    def place_voxel(self, world_pos: Tuple[float, float, float],
                   color: Optional[Tuple[float, float, float]] = None,
                   record_history: bool = True) -> bool:
        """
        Place a voxel at the given world position.

        Returns True if voxel was placed, False if position was occupied.
        """
        grid_pos = self.world_to_grid(world_pos)

        # Check if position is occupied
        if grid_pos in self.voxels:
            return False

        # Check max voxels limit
        if len(self.voxels) >= CONFIG.voxel.MAX_VOXELS:
            return False

        # Use current color if none specified
        if color is None:
            color = self.current_color

        # Create and store voxel
        actual_world_pos = self.grid_to_world(grid_pos)
        voxel = Voxel(
            position=actual_world_pos,
            original_position=actual_world_pos,
            color=color,
            authored_color=color
        )
        self.voxels[grid_pos] = voxel

        # Record history
        if record_history:
            action = HistoryAction(
                action_type="place",
                new_voxels_data=[{
                    "grid_pos": grid_pos,
                    "color": color,
                    "authored_color": color,
                    "selected": False,
                }]
            )
            self._push_undo(action)

        return True

    def delete_voxel(self, grid_pos: Tuple[int, int, int],
                    record_history: bool = True) -> bool:
        """Delete a voxel at the given grid position."""
        if grid_pos not in self.voxels:
            return False

        voxel = self.voxels[grid_pos]

        # Record history before deleting
        if record_history:
            action = HistoryAction(
                action_type="delete",
                voxels_data=[{
                    "grid_pos": grid_pos,
                    "color": voxel.color,
                    "authored_color": voxel.authored_color,
                    "selected": voxel.is_selected,
                }]
            )
            self._push_undo(action)

        # Remove from selection if selected
        self.selected_voxels.discard(grid_pos)

        # Delete
        del self.voxels[grid_pos]

        return True

    def delete_at_world_pos(self, world_pos: Tuple[float, float, float]) -> bool:
        """Delete a voxel at the given world position."""
        grid_pos = self.world_to_grid(world_pos)
        return self.delete_voxel(grid_pos)

    def delete_selected(self) -> int:
        """Delete all selected voxels. Returns count of deleted voxels."""
        if not self.selected_voxels:
            return 0

        # Record history
        voxels_data = []
        for grid_pos in self.selected_voxels:
            if grid_pos in self.voxels:
                voxels_data.append({
                    "grid_pos": grid_pos,
                    "color": self.voxels[grid_pos].color,
                    "authored_color": self.voxels[grid_pos].authored_color,
                    "selected": self.voxels[grid_pos].is_selected,
                })

        if voxels_data:
            action = HistoryAction(action_type="delete", voxels_data=voxels_data)
            self._push_undo(action)

        # Delete voxels
        count = 0
        for grid_pos in list(self.selected_voxels):
            if grid_pos in self.voxels:
                del self.voxels[grid_pos]
                count += 1

        self.selected_voxels.clear()
        return count

    def extend_voxels(self, start_pos: Tuple[float, float, float],
                     direction: Tuple[float, float, float],
                     count: int = 1) -> int:
        """
        Extend/extrude voxels in a direction from start position.
        Returns number of voxels created.
        """
        # Quantize direction to nearest axis
        abs_dir = (abs(direction[0]), abs(direction[1]), abs(direction[2]))
        max_axis = max(abs_dir)

        if max_axis < 0.1:
            return 0

        if abs_dir[0] == max_axis:
            step = (1 if direction[0] > 0 else -1, 0, 0)
        elif abs_dir[1] == max_axis:
            step = (0, 1 if direction[1] > 0 else -1, 0)
        else:
            step = (0, 0, 1 if direction[2] > 0 else -1)

        # Get starting grid position
        start_grid = self.world_to_grid(start_pos)

        # Get color from source voxel if it exists
        color = self.current_color
        if start_grid in self.voxels:
            color = self.voxels[start_grid].color

        # Create voxels along direction
        created = 0
        voxels_data = []

        for i in range(1, count + 1):
            new_grid = (
                start_grid[0] + step[0] * i,
                start_grid[1] + step[1] * i,
                start_grid[2] + step[2] * i
            )

            if new_grid not in self.voxels and len(self.voxels) < CONFIG.voxel.MAX_VOXELS:
                world_pos = self.grid_to_world(new_grid)
                voxel = Voxel(
                    position=world_pos,
                    original_position=world_pos,
                    color=color,
                    authored_color=color
                )
                self.voxels[new_grid] = voxel
                voxels_data.append({
                    "grid_pos": new_grid,
                    "color": color,
                    "authored_color": color,
                    "selected": False,
                })
                created += 1

        # Record history
        if voxels_data:
            action = HistoryAction(action_type="place", new_voxels_data=voxels_data)
            self._push_undo(action)

        return created

    def select_voxel(self, grid_pos: Tuple[int, int, int], add_to_selection: bool = False):
        """Select a voxel."""
        if not add_to_selection:
            self.clear_selection()

        if grid_pos in self.voxels:
            self.selected_voxels.add(grid_pos)
            self.voxels[grid_pos].is_selected = True

    def select_in_box(self, min_pos: Tuple[float, float, float],
                     max_pos: Tuple[float, float, float]):
        """Select all voxels within a box."""
        self.clear_selection()

        min_grid = self.world_to_grid(min_pos)
        max_grid = self.world_to_grid(max_pos)

        for grid_pos, voxel in self.voxels.items():
            if (min_grid[0] <= grid_pos[0] <= max_grid[0] and
                min_grid[1] <= grid_pos[1] <= max_grid[1] and
                min_grid[2] <= grid_pos[2] <= max_grid[2]):
                self.selected_voxels.add(grid_pos)
                voxel.is_selected = True

    def clear_selection(self):
        """Clear all selections."""
        for grid_pos in self.selected_voxels:
            if grid_pos in self.voxels:
                self.voxels[grid_pos].is_selected = False
        self.selected_voxels.clear()

    def set_color(self, color: Tuple[float, float, float]):
        """Set the current color for new voxels."""
        self.current_color = color

    def start_disco_mode(self) -> bool:
        """Begin continuously randomizing displayed voxel colors."""
        self.disco_mode = True
        return self.disco_mode

    def freeze_disco_colors(self) -> bool:
        """Stop disco mode and keep the currently visible colors."""
        was_active = self.disco_mode
        self.disco_mode = False
        return was_active

    def restore_original_colors(self) -> bool:
        """Stop disco mode and restore each voxel's authored placement color."""
        self.disco_mode = False
        restored = False
        for voxel in self.voxels.values():
            voxel.color = voxel.authored_color
            restored = True
        return restored

    def change_selected_color(self, new_color: Tuple[float, float, float]):
        """Change the color of selected voxels."""
        if not self.selected_voxels:
            return

        # Record history
        voxels_data = []
        new_voxels_data = []

        for grid_pos in self.selected_voxels:
            if grid_pos in self.voxels:
                voxel = self.voxels[grid_pos]
                voxels_data.append({
                    "grid_pos": grid_pos,
                    "color": voxel.color,
                    "authored_color": voxel.authored_color,
                    "selected": voxel.is_selected,
                })
                new_voxels_data.append({
                    "grid_pos": grid_pos,
                    "color": new_color,
                    "authored_color": new_color,
                    "selected": voxel.is_selected,
                })
                voxel.color = new_color
                voxel.authored_color = new_color

        if voxels_data:
            action = HistoryAction(
                action_type="color_change",
                voxels_data=voxels_data,
                new_voxels_data=new_voxels_data
            )
            self._push_undo(action)

    def rotate_all(self, angle: float, axis: str = 'y'):
        """Rotate ALL voxels around their centroid."""
        if not self.voxels:
            return
        
        # Calculate centroid of all voxels
        positions = [self.grid_to_world(gp) for gp in self.voxels.keys()]
        centroid = (
            sum(p[0] for p in positions) / len(positions),
            sum(p[1] for p in positions) / len(positions),
            sum(p[2] for p in positions) / len(positions)
        )
        
        # Remove old voxels and store them
        voxels_to_move = [(gp, self.voxels[gp]) for gp in list(self.voxels.keys())]
        self.voxels.clear()
        
        # Add at new positions
        for old_grid_pos, voxel in voxels_to_move:
            old_world = self.grid_to_world(old_grid_pos)
            new_world = rotate_point_around_center(old_world, centroid, angle, axis)
            new_grid = self.world_to_grid(new_world)
            
            # Avoid collisions
            if new_grid not in self.voxels:
                actual_world = self.grid_to_world(new_grid)
                voxel.position = actual_world
                voxel.original_position = actual_world
                self.voxels[new_grid] = voxel
        
        # Buffer will be rebuilt on next render automatically

    # ============ PHASE 9.2: GROUP TRANSFORM METHODS ============
    
    def start_grab(self, hand_pos: Tuple[float, float, float]):
        """Start grabbing the voxel group at the given hand position."""
        self.grab_start_pos = hand_pos
        self.is_grabbed = True
        print(f"[VOXEL] Grab started at {hand_pos}")
    
    def update_grab(self, hand_pos: Tuple[float, float, float]):
        """Update the group offset based on hand movement during grab.
        
        PHASE 11: Increased sensitivity (2.5x) for responsive feel.
        """
        if not self.is_grabbed or self.grab_start_pos is None:
            return
        
        # PHASE 12: Sensitivity multiplier for responsive grab (reduced from 2.5)
        GRAB_SENSITIVITY = 2.0
        
        # Calculate delta from grab start with sensitivity multiplier
        delta = (
            (hand_pos[0] - self.grab_start_pos[0]) * GRAB_SENSITIVITY,
            (hand_pos[1] - self.grab_start_pos[1]) * GRAB_SENSITIVITY,
            (hand_pos[2] - self.grab_start_pos[2]) * GRAB_SENSITIVITY
        )
        self.group_offset = delta
        # Note: No buffer rebuild needed - group transform applied in shader

    def get_snapped_group_offset(self) -> Tuple[float, float, float]:
        """Snap live grab translation to the voxel grid for predictable whole-structure moves."""
        return self.snap_to_grid(self.group_offset)
    
    def end_grab(self, record_history: bool = True):
        """End the grab and apply the offset to all voxels permanently."""
        if not self.is_grabbed:
            return
        
        voxels_before = self._snapshot_voxels() if record_history else []

        # Apply the offset to all voxels permanently
        if self.get_snapped_group_offset() != (0.0, 0.0, 0.0):
            self._apply_group_offset()
            if record_history:
                voxels_after = self._snapshot_voxels()
                if voxels_before != voxels_after:
                    self._push_undo(HistoryAction(
                        action_type="move",
                        voxels_data=voxels_before,
                        new_voxels_data=voxels_after
                    ))
        
        # Reset grab state
        self.grab_start_pos = None
        self.is_grabbed = False
        self.group_offset = (0.0, 0.0, 0.0)
        print("[VOXEL] Grab ended, offset applied")
    
    def _apply_group_offset(self):
        """Apply the current group offset to all voxels permanently."""
        if not self.voxels:
            return
        
        offset = self.get_snapped_group_offset()
        delta_grid = tuple(int(round(component / self.grid_size)) for component in offset)
        if delta_grid == (0, 0, 0):
            return

        voxels_to_move = [(gp, self.voxels[gp]) for gp in list(self.voxels.keys())]
        translated_voxels: Dict[Tuple[int, int, int], Voxel] = {}
        
        for old_grid_pos, voxel in voxels_to_move:
            new_grid = (
                old_grid_pos[0] + delta_grid[0],
                old_grid_pos[1] + delta_grid[1],
                old_grid_pos[2] + delta_grid[2]
            )
            actual_world = self.grid_to_world(new_grid)
            voxel.position = actual_world
            voxel.original_position = actual_world
            translated_voxels[new_grid] = voxel

        self.voxels = translated_voxels
        if self.selected_voxels:
            self.selected_voxels = {
                (
                    grid_pos[0] + delta_grid[0],
                    grid_pos[1] + delta_grid[1],
                    grid_pos[2] + delta_grid[2],
                )
                for grid_pos in self.selected_voxels
            }
        
        # Buffer will be rebuilt on next render automatically
    
    def update_group_rotation(self, delta_y: float, delta_x: float = 0.0):
        """Update the group rotation by delta angles (continuous rotation).
        
        Args:
            delta_y: Y-axis rotation delta (horizontal spin)
            delta_x: X-axis rotation delta (tilt up/down)
        """
        self.group_rotation += delta_y
        # Store X rotation separately for full 3D control
        if not hasattr(self, 'group_rotation_x'):
            self.group_rotation_x = 0.0
        self.group_rotation_x += delta_x
        # Note: No buffer rebuild needed - group transform applied in shader
    
    def get_group_transform(self) -> Tuple[Tuple[float, float, float], float]:
        """Get the current group transform (offset, rotation)."""
        return (self.group_offset, self.group_rotation)

    def get_model_matrix(self) -> np.ndarray:
        """Create a 4x4 model matrix from group offset and rotation.
        
        YouTube approach: Apply group transform as a model matrix in shader.
        This avoids rebuilding instance buffers on every frame during grab/rotate.
        
        Returns:
            4x4 numpy array (float32) representing the model transformation.
        """
        # Get X rotation (tilt)
        rot_x = getattr(self, 'group_rotation_x', 0.0)
        rot_y = self.group_rotation
        
        # Y-axis rotation matrix
        cos_y = math.cos(rot_y)
        sin_y = math.sin(rot_y)
        rotation_y = np.array([
            [cos_y,  0, sin_y, 0],
            [0,      1, 0,     0],
            [-sin_y, 0, cos_y, 0],
            [0,      0, 0,     1]
        ], dtype=np.float32)
        
        # X-axis rotation matrix (tilt)
        cos_x = math.cos(rot_x)
        sin_x = math.sin(rot_x)
        rotation_x = np.array([
            [1, 0,      0,       0],
            [0, cos_x, -sin_x,   0],
            [0, sin_x,  cos_x,   0],
            [0, 0,      0,       1]
        ], dtype=np.float32)
        
        # Apply translation (group_offset)
        snapped_offset = self.get_snapped_group_offset()
        translation = np.array([
            [1, 0, 0, snapped_offset[0]],
            [0, 1, 0, snapped_offset[1]],
            [0, 0, 1, snapped_offset[2]],
            [0, 0, 0, 1]
        ], dtype=np.float32)
        
        # Combined: translation * rotation_y * rotation_x
        model = translation @ rotation_y @ rotation_x
        
        return model

    def reset_group_transform(self, log: bool = True):
        """Reset group offset and rotation to origin (both fists gesture)."""
        self.group_offset = (0.0, 0.0, 0.0)
        self.group_rotation = 0.0
        self.group_rotation_x = 0.0
        self.is_grabbed = False
        self.grab_start_pos = None
        if log:
            print("[VOXEL] Group transform reset to origin")

    def _snapshot_voxels(self, grid_positions: Optional[List[Tuple[int, int, int]]] = None) -> List[Dict]:
        """Capture a minimal, serializable snapshot for undo/redo and file I/O."""
        snapshot = []
        positions = grid_positions if grid_positions is not None else list(self.voxels.keys())

        for grid_pos in positions:
            voxel = self.voxels.get(grid_pos)
            if voxel is None:
                continue
            snapshot.append({
                "grid_pos": tuple(grid_pos),
                "color": tuple(voxel.color),
                "authored_color": tuple(voxel.authored_color),
                "selected": voxel.is_selected,
            })

        snapshot.sort(key=lambda item: item["grid_pos"])
        return snapshot

    def _build_voxel_from_snapshot(self, data: Dict) -> Tuple[Tuple[int, int, int], Voxel]:
        """Recreate a voxel from serialized history/file state."""
        grid_pos = tuple(data["grid_pos"])
        world_pos = self.grid_to_world(grid_pos)
        voxel = Voxel(
            position=world_pos,
            original_position=world_pos,
            color=tuple(data["color"]),
            authored_color=tuple(data.get("authored_color", data["color"])),
            is_selected=data.get("selected", False)
        )
        return grid_pos, voxel

    def _reset_runtime_state(self, clear_history: bool = False):
        """Reset transient simulation/editor state after destructive scene changes."""
        self.clear_selection()
        self.scatter_state = "normal"
        self.recombine_start_time = 0.0
        self.restore_mode = False
        self.disco_mode = False
        self.reset_group_transform(log=False)

        if clear_history:
            self.undo_stack.clear()
            self.redo_stack.clear()

    def get_bounding_box(self) -> Optional[Dict[str, Tuple[float, float, float]]]:
        """Get the bounding box of all voxels.
        
        PHASE 12: Used for grab preview to show where structure will move.
        
        Returns:
            Dictionary with 'min' and 'max' tuples, or None if no voxels.
        """
        if not self.voxels:
            return None
        
        positions = list(self.voxels.keys())
        grid_size = CONFIG.voxel.GRID_SIZE
        
        min_x = min(p[0] for p in positions) * grid_size
        max_x = (max(p[0] for p in positions) + 1) * grid_size
        min_y = min(p[1] for p in positions) * grid_size
        max_y = (max(p[1] for p in positions) + 1) * grid_size
        min_z = min(p[2] for p in positions) * grid_size
        max_z = (max(p[2] for p in positions) + 1) * grid_size
        
        return {
            'min': (min_x, min_y, min_z),
            'max': (max_x, max_y, max_z),
            'center': ((min_x + max_x) / 2, (min_y + max_y) / 2, (min_z + max_z) / 2)
        }

    def clear(self, record_history: bool = False):
        """Clear all voxels from the scene.
        
        Phase 15: Used by FULL_RESET gesture (right thumb up only for 5s).
        """
        snapshot = self._snapshot_voxels() if record_history else []
        count = len(self.voxels)
        self.voxels.clear()
        self._reset_runtime_state(clear_history=not record_history)
        if record_history and snapshot:
            self._push_undo(HistoryAction(action_type="delete", voxels_data=snapshot))
        print(f"[VOXEL] Cleared all {count} voxels")

    def rotate_selected(self, angle: float, axis: str = 'y'):
        """Rotate selected voxels around their centroid."""
        if not self.selected_voxels:
            return

        # Calculate centroid
        positions = [self.grid_to_world(gp) for gp in self.selected_voxels]
        centroid = (
            sum(p[0] for p in positions) / len(positions),
            sum(p[1] for p in positions) / len(positions),
            sum(p[2] for p in positions) / len(positions)
        )

        # Record history
        voxels_data = []
        for grid_pos in self.selected_voxels:
            if grid_pos in self.voxels:
                voxels_data.append({
                    "grid_pos": grid_pos,
                    "color": self.voxels[grid_pos].color,
                    "authored_color": self.voxels[grid_pos].authored_color,
                    "selected": self.voxels[grid_pos].is_selected,
                })

        # Remove old voxels
        voxels_to_move = [(gp, self.voxels[gp]) for gp in self.selected_voxels if gp in self.voxels]
        for gp, _ in voxels_to_move:
            del self.voxels[gp]

        # Add at new positions
        new_selection = set()
        new_voxels_data = []

        for old_grid_pos, voxel in voxels_to_move:
            old_world = self.grid_to_world(old_grid_pos)
            new_world = rotate_point_around_center(old_world, centroid, angle, axis)
            new_grid = self.world_to_grid(new_world)

            # Avoid collisions
            if new_grid not in self.voxels:
                actual_world = self.grid_to_world(new_grid)
                voxel.position = actual_world
                voxel.original_position = actual_world
                self.voxels[new_grid] = voxel
                new_selection.add(new_grid)
                new_voxels_data.append({
                    "grid_pos": new_grid,
                    "color": voxel.color,
                    "authored_color": voxel.authored_color,
                    "selected": voxel.is_selected,
                })

        self.selected_voxels = new_selection

        # Record history
        if voxels_data:
            action = HistoryAction(
                action_type="move",
                voxels_data=voxels_data,
                new_voxels_data=new_voxels_data
            )
            self._push_undo(action)

    def scatter(self, explosion_center: Tuple[float, float, float]):
        """Scatter all voxels with physics."""
        if self.scatter_state != "normal" or not self.voxels:
            return

        self.scatter_state = "scattered"
        cfg = self.physics_cfg

        for grid_pos, voxel in self.voxels.items():
            # Save original position
            voxel.original_position = voxel.position

            # Calculate direction from explosion center
            dx = voxel.position[0] - explosion_center[0]
            dy = voxel.position[1] - explosion_center[1]
            dz = voxel.position[2] - explosion_center[2]

            distance = max(0.1, math.sqrt(dx**2 + dy**2 + dz**2))

            # Normalize direction
            dir_x = dx / distance
            dir_y = dy / distance
            dir_z = dz / distance

            # Calculate force (inverse square with random variation)
            force = cfg.EXPLOSION_FORCE * (1.0 / (1.0 + distance * 0.1))
            variation = 1.0 + (np.random.random() - 0.5) * 2 * cfg.EXPLOSION_VARIATION

            # Set velocity
            voxel.velocity = (
                dir_x * force * variation,
                dir_y * force * variation + 3.0,  # Add upward component
                dir_z * force * variation
            )

            # Set angular velocity (random tumbling)
            voxel.angular_velocity = (
                (np.random.random() - 0.5) * cfg.ANGULAR_VELOCITY_MAX,
                (np.random.random() - 0.5) * cfg.ANGULAR_VELOCITY_MAX,
                (np.random.random() - 0.5) * cfg.ANGULAR_VELOCITY_MAX
            )

            voxel.state = VoxelState.DYNAMIC

    def recombine(self):
        """Start recombining scattered voxels."""
        if self.scatter_state not in ("scattered", "gravity_burst"):
            return

        self.scatter_state = "recombining"
        self.recombine_start_time = time.time()

        # Calculate maximum distance for staggered animation
        max_distance = 0.0
        for voxel in self.voxels.values():
            dist = distance_3d(voxel.position, voxel.original_position)
            max_distance = max(max_distance, dist)

        # Set up animation for each voxel
        for voxel in self.voxels.values():
            voxel.animation_start_pos = voxel.position
            voxel.animation_progress = 0.0

            # Stagger based on distance
            dist = distance_3d(voxel.position, voxel.original_position)
            voxel.animation_delay = (dist / max(max_distance, 1.0)) * 0.5  # 0-0.5 second stagger

            voxel.state = VoxelState.ANIMATING
            voxel.velocity = (0.0, 0.0, 0.0)

    def gravity_burst(self):
        """Phase 5: YouTube-style gravity burst - explode upward then fall.

        YouTube reference: initiateGravityFall()
        - Random X/Z velocity (-1.5 to 1.5)
        - Upward Y velocity (4 to 12)
        - Gravity pulls down
        """
        if self.scatter_state != "normal" or not self.voxels:
            return

        self.scatter_state = "gravity_burst"

        for voxel in self.voxels.values():
            # Save original position
            voxel.original_position = voxel.position

            # YouTube-style random upward burst
            voxel.velocity = (
                (np.random.random() - 0.5) * 3.0,    # Random X (-1.5 to 1.5)
                np.random.random() * 8.0 + 4.0,       # Upward Y (4 to 12)
                (np.random.random() - 0.5) * 3.0     # Random Z (-1.5 to 1.5)
            )

            # Random tumbling
            voxel.angular_velocity = (
                (np.random.random() - 0.5) * 3.0,
                (np.random.random() - 0.5) * 3.0,
                (np.random.random() - 0.5) * 3.0
            )

            voxel.state = VoxelState.DYNAMIC

    def restore(self):
        """Phase 5: YouTube-style smooth restore - lerp back to origin.

        YouTube reference: v.position.lerp(v.origin, 0.1)
        Smooth 10% per frame movement toward original position.
        """
        if self.scatter_state not in ("scattered", "gravity_burst"):
            return

        self.scatter_state = "restoring"
        self.restore_mode = True

        for voxel in self.voxels.values():
            voxel.state = VoxelState.ANIMATING
            voxel.velocity = (0.0, 0.0, 0.0)

    def cycle_colors(self):
        """Phase 5: Cycle through color palette (YouTube peace sign feature)."""
        palette = CONFIG.colors.get_palette()
        self.color_cycle_index = (self.color_cycle_index + 1) % len(palette)
        self.current_color = palette[self.color_cycle_index]

        # Also update all existing voxels
        for voxel in self.voxels.values():
            voxel.color = self.current_color
            voxel.authored_color = self.current_color

        return self.current_color

    def toggle_disco_mode(self) -> bool:
        """Phase 5: Toggle disco mode while preserving visible colors when switching off."""
        if self.disco_mode:
            self.freeze_disco_colors()
        else:
            self.start_disco_mode()
        return self.disco_mode

    def update_disco(self):
        """Phase 5: Update disco mode - random HSL colors for each voxel."""
        if not self.disco_mode:
            return

        import colorsys
        for voxel in self.voxels.values():
            # Random HSL color
            h = np.random.random()
            s = 0.8 + np.random.random() * 0.2  # 80-100% saturation
            l = 0.5 + np.random.random() * 0.3  # 50-80% lightness

            # Convert HSL to RGB
            r, g, b = colorsys.hls_to_rgb(h, l, s)
            voxel.color = (r, g, b)

    def update_physics(self, delta_time: float):
        """Update physics simulation."""
        # Phase 5: Update disco mode colors
        self.update_disco()

        if self.scatter_state == "scattered":
            self._update_scattered_physics(delta_time)
        elif self.scatter_state == "gravity_burst":
            self._update_gravity_burst_physics(delta_time)
        elif self.scatter_state == "recombining":
            self._update_recombine_animation(delta_time)
        elif self.scatter_state == "restoring":
            self._update_restore_animation(delta_time)

    def _update_scattered_physics(self, delta_time: float):
        """Update physics for scattered voxels."""
        cfg = self.physics_cfg

        for voxel in self.voxels.values():
            if voxel.state != VoxelState.DYNAMIC:
                continue

            # Apply gravity
            voxel.velocity = (
                voxel.velocity[0],
                voxel.velocity[1] + cfg.GRAVITY[1] * delta_time,
                voxel.velocity[2]
            )

            # Apply drag
            voxel.velocity = (
                voxel.velocity[0] * cfg.DRAG,
                voxel.velocity[1] * cfg.DRAG,
                voxel.velocity[2] * cfg.DRAG
            )

            # Update position
            voxel.position = (
                voxel.position[0] + voxel.velocity[0] * delta_time,
                voxel.position[1] + voxel.velocity[1] * delta_time,
                voxel.position[2] + voxel.velocity[2] * delta_time
            )

            # Update rotation
            voxel.rotation = (
                voxel.rotation[0] + voxel.angular_velocity[0] * delta_time,
                voxel.rotation[1] + voxel.angular_velocity[1] * delta_time,
                voxel.rotation[2] + voxel.angular_velocity[2] * delta_time
            )

            # Floor collision (simple)
            if voxel.position[1] < -5.0:
                voxel.position = (voxel.position[0], -5.0, voxel.position[2])
                voxel.velocity = (
                    voxel.velocity[0] * 0.5,
                    abs(voxel.velocity[1]) * 0.3,  # Bounce
                    voxel.velocity[2] * 0.5
                )

    def _update_recombine_animation(self, delta_time: float):
        """Update recombine animation."""
        cfg = self.physics_cfg
        elapsed = time.time() - self.recombine_start_time
        all_complete = True

        for voxel in self.voxels.values():
            if voxel.state != VoxelState.ANIMATING:
                continue

            # Account for stagger delay
            actual_elapsed = max(0, elapsed - voxel.animation_delay)
            voxel.animation_progress = min(1.0, actual_elapsed / cfg.RECOMBINE_DURATION)

            if voxel.animation_progress < 1.0:
                all_complete = False

                # Ease-in-out interpolation
                t = ease_in_out(voxel.animation_progress, cfg.RECOMBINE_EASE_POWER)
                voxel.position = lerp3(voxel.animation_start_pos, voxel.original_position, t)

                # Reduce rotation as it settles
                rot_factor = 1.0 - voxel.animation_progress
                voxel.rotation = (
                    voxel.rotation[0] * rot_factor,
                    voxel.rotation[1] * rot_factor,
                    voxel.rotation[2] * rot_factor
                )
            else:
                # Animation complete
                voxel.position = voxel.original_position
                voxel.rotation = (0.0, 0.0, 0.0)
                voxel.state = VoxelState.STATIC

        if all_complete:
            self.scatter_state = "normal"

    def _update_gravity_burst_physics(self, delta_time: float):
        """Phase 5: Update gravity burst physics (YouTube-style)."""
        cfg = self.physics_cfg

        for voxel in self.voxels.values():
            if voxel.state != VoxelState.DYNAMIC:
                continue

            # Apply gravity (stronger for dramatic effect)
            gravity_strength = -15.0  # Stronger than normal scatter
            voxel.velocity = (
                voxel.velocity[0],
                voxel.velocity[1] + gravity_strength * delta_time,
                voxel.velocity[2]
            )

            # Light air resistance
            drag = 0.99
            voxel.velocity = (
                voxel.velocity[0] * drag,
                voxel.velocity[1] * drag,
                voxel.velocity[2] * drag
            )

            # Update position
            voxel.position = (
                voxel.position[0] + voxel.velocity[0] * delta_time,
                voxel.position[1] + voxel.velocity[1] * delta_time,
                voxel.position[2] + voxel.velocity[2] * delta_time
            )

            # Update rotation
            voxel.rotation = (
                voxel.rotation[0] + voxel.angular_velocity[0] * delta_time,
                voxel.rotation[1] + voxel.angular_velocity[1] * delta_time,
                voxel.rotation[2] + voxel.angular_velocity[2] * delta_time
            )

            # Floor collision with bounce
            if voxel.position[1] < -5.0:
                voxel.position = (voxel.position[0], -5.0, voxel.position[2])
                voxel.velocity = (
                    voxel.velocity[0] * 0.7,
                    abs(voxel.velocity[1]) * 0.4,  # Bounce
                    voxel.velocity[2] * 0.7
                )

    def _update_restore_animation(self, delta_time: float):
        """Phase 5: YouTube-style smooth restore using lerp.

        YouTube reference: v.position.lerp(v.origin, 0.1)
        Each frame moves 10% closer to origin.
        """
        lerp_factor = 0.1  # 10% per frame (YouTube default)
        threshold = 0.05   # Distance threshold to snap to origin
        all_complete = True

        for voxel in self.voxels.values():
            if voxel.state != VoxelState.ANIMATING:
                continue

            # Calculate distance to origin
            dist = distance_3d(voxel.position, voxel.original_position)

            if dist > threshold:
                all_complete = False

                # Lerp toward origin (YouTube style)
                voxel.position = (
                    voxel.position[0] + (voxel.original_position[0] - voxel.position[0]) * lerp_factor,
                    voxel.position[1] + (voxel.original_position[1] - voxel.position[1]) * lerp_factor,
                    voxel.position[2] + (voxel.original_position[2] - voxel.position[2]) * lerp_factor
                )

                # Reduce rotation
                voxel.rotation = (
                    voxel.rotation[0] * (1.0 - lerp_factor),
                    voxel.rotation[1] * (1.0 - lerp_factor),
                    voxel.rotation[2] * (1.0 - lerp_factor)
                )
            else:
                # Snap to origin
                voxel.position = voxel.original_position
                voxel.rotation = (0.0, 0.0, 0.0)
                voxel.state = VoxelState.STATIC

        if all_complete:
            self.scatter_state = "normal"
            self.restore_mode = False

    def raycast(self, ray: Ray) -> Optional[Tuple[int, int, int]]:
        """
        Find the first voxel intersected by a ray.
        Returns grid position of hit voxel, or None.
        """
        closest_hit = None
        closest_dist = float('inf')
        half_size = self.grid_size / 2.0

        for grid_pos, voxel in self.voxels.items():
            # Calculate voxel bounds
            center = voxel.position
            box_min = (center[0] - half_size, center[1] - half_size, center[2] - half_size)
            box_max = (center[0] + half_size, center[1] + half_size, center[2] + half_size)

            # Test intersection
            dist = ray_box_intersection(ray, box_min, box_max)
            if dist is not None and dist < closest_dist:
                closest_dist = dist
                closest_hit = grid_pos

        return closest_hit

    def _push_undo(self, action: HistoryAction):
        """Push an action to the undo stack."""
        self.undo_stack.append(action)
        self.redo_stack.clear()

        # Limit stack size
        while len(self.undo_stack) > self.max_history:
            self.undo_stack.pop(0)

    def undo(self) -> bool:
        """Undo the last action. Returns True if successful."""
        if not self.undo_stack:
            return False

        action = self.undo_stack.pop()
        self.redo_stack.append(action)

        if action.action_type == "place":
            # Remove placed voxels
            for data in action.new_voxels_data:
                grid_pos = data["grid_pos"]
                if grid_pos in self.voxels:
                    del self.voxels[grid_pos]
                    self.selected_voxels.discard(grid_pos)

        elif action.action_type == "delete":
            # Restore deleted voxels
            for data in action.voxels_data:
                grid_pos, voxel = self._build_voxel_from_snapshot(data)
                self.voxels[grid_pos] = voxel
                if voxel.is_selected:
                    self.selected_voxels.add(grid_pos)

        elif action.action_type == "move":
            # Restore original positions
            for data in action.new_voxels_data:
                if data["grid_pos"] in self.voxels:
                    del self.voxels[data["grid_pos"]]
                    self.selected_voxels.discard(data["grid_pos"])

            for data in action.voxels_data:
                grid_pos, voxel = self._build_voxel_from_snapshot(data)
                self.voxels[grid_pos] = voxel
                if voxel.is_selected:
                    self.selected_voxels.add(grid_pos)

        elif action.action_type == "color_change":
            # Restore original colors
            for data in action.voxels_data:
                grid_pos = data["grid_pos"]
                if grid_pos in self.voxels:
                    self.voxels[grid_pos].color = data["color"]
                    self.voxels[grid_pos].authored_color = data.get("authored_color", data["color"])

        return True

    def redo(self) -> bool:
        """Redo the last undone action. Returns True if successful."""
        if not self.redo_stack:
            return False

        action = self.redo_stack.pop()
        self.undo_stack.append(action)

        if action.action_type == "place":
            # Re-place voxels
            for data in action.new_voxels_data:
                grid_pos, voxel = self._build_voxel_from_snapshot(data)
                self.voxels[grid_pos] = voxel
                if voxel.is_selected:
                    self.selected_voxels.add(grid_pos)

        elif action.action_type == "delete":
            # Re-delete voxels
            for data in action.voxels_data:
                grid_pos = data["grid_pos"]
                if grid_pos in self.voxels:
                    del self.voxels[grid_pos]
                    self.selected_voxels.discard(grid_pos)

        elif action.action_type == "move":
            # Apply move again
            for data in action.voxels_data:
                if data["grid_pos"] in self.voxels:
                    del self.voxels[data["grid_pos"]]
                    self.selected_voxels.discard(data["grid_pos"])

            for data in action.new_voxels_data:
                grid_pos, voxel = self._build_voxel_from_snapshot(data)
                self.voxels[grid_pos] = voxel
                if voxel.is_selected:
                    self.selected_voxels.add(grid_pos)

        elif action.action_type == "color_change":
            # Apply new colors
            for data in action.new_voxels_data:
                grid_pos = data["grid_pos"]
                if grid_pos in self.voxels:
                    self.voxels[grid_pos].color = data["color"]
                    self.voxels[grid_pos].authored_color = data.get("authored_color", data["color"])

        return True

    def save_to_file(self, filepath: str) -> bool:
        """Save voxel data to JSON file."""
        data = {
            "version": "1.2",
            "grid_size": self.grid_size,
            "current_color": list(self.current_color),
            "voxels": []
        }

        for grid_pos, voxel in sorted(self.voxels.items()):
            data["voxels"].append({
                "x": grid_pos[0],
                "y": grid_pos[1],
                "z": grid_pos[2],
                "color": list(voxel.color),
                "authored_color": list(voxel.authored_color),
            })

        try:
            directory = os.path.dirname(filepath)
            if directory:
                os.makedirs(directory, exist_ok=True)

            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
            return True
        except (OSError, TypeError, ValueError) as exc:
            print(f"Error saving file: {exc}")
            return False

    def load_from_file(self, filepath: str) -> bool:
        """Load voxel data from JSON file."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)

            if not isinstance(data, dict):
                raise ValueError("Scene file must contain a JSON object")

            voxels_data = data.get("voxels", [])
            if not isinstance(voxels_data, list):
                raise ValueError("Scene file 'voxels' field must be a list")

            self.voxels.clear()
            self._reset_runtime_state(clear_history=True)

            for voxel_data in voxels_data:
                if not all(key in voxel_data for key in ("x", "y", "z", "color")):
                    raise ValueError("Scene voxel is missing required fields")

                grid_pos = (
                    int(voxel_data["x"]),
                    int(voxel_data["y"]),
                    int(voxel_data["z"]),
                )
                color = tuple(float(channel) for channel in voxel_data["color"])
                if len(color) != 3:
                    raise ValueError("Voxel color must contain exactly 3 channels")
                authored_color_data = voxel_data.get("authored_color", voxel_data["color"])
                authored_color = tuple(float(channel) for channel in authored_color_data)
                if len(authored_color) != 3:
                    raise ValueError("Voxel authored_color must contain exactly 3 channels")
                world_pos = self.grid_to_world(grid_pos)
                voxel = Voxel(
                    position=world_pos,
                    original_position=world_pos,
                    color=color,
                    authored_color=authored_color
                )
                self.voxels[grid_pos] = voxel

            current_color = data.get("current_color")
            if isinstance(current_color, list) and len(current_color) == 3:
                self.current_color = tuple(float(channel) for channel in current_color)

            return True

        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            print(f"Error loading file: {exc}")
            return False

    def get_voxel_count(self) -> int:
        """Get total number of voxels."""
        return len(self.voxels)

    def get_selected_count(self) -> int:
        """Get number of selected voxels."""
        return len(self.selected_voxels)

    def export_to_obj(self, filepath: str, include_colors: bool = True) -> bool:
        """
        Export voxel model to Wavefront OBJ format.

        Args:
            filepath: Path to save OBJ file
            include_colors: If True, also creates a .mtl material file

        Returns:
            True if export successful
        """
        if not self.voxels:
            return False

        try:
            # Group voxels by color for materials
            color_groups: Dict[Tuple[float, float, float], List[Tuple[int, int, int]]] = {}
            for grid_pos, voxel in self.voxels.items():
                color_key = tuple(round(c, 3) for c in voxel.color)
                if color_key not in color_groups:
                    color_groups[color_key] = []
                color_groups[color_key].append(grid_pos)

            # Generate material file if colors enabled
            mtl_filepath = filepath.rsplit('.', 1)[0] + '.mtl'
            if include_colors:
                self._write_mtl_file(mtl_filepath, color_groups.keys())

            # Write OBJ file
            with open(filepath, 'w') as f:
                f.write(f"# JARVIS Voxel Editor Export\n")
                f.write(f"# Voxels: {len(self.voxels)}\n\n")

                if include_colors:
                    f.write(f"mtllib {os.path.basename(mtl_filepath)}\n\n")

                vertex_index = 1
                half = self.grid_size / 2.0

                # Cube vertices template (8 vertices per cube)
                cube_offsets = [
                    (-half, -half, -half),
                    ( half, -half, -half),
                    ( half,  half, -half),
                    (-half,  half, -half),
                    (-half, -half,  half),
                    ( half, -half,  half),
                    ( half,  half,  half),
                    (-half,  half,  half),
                ]

                # Cube faces (6 faces, each with 4 vertices)
                cube_faces = [
                    (1, 2, 3, 4),  # Back
                    (5, 8, 7, 6),  # Front
                    (1, 5, 6, 2),  # Bottom
                    (4, 3, 7, 8),  # Top
                    (1, 4, 8, 5),  # Left
                    (2, 6, 7, 3),  # Right
                ]

                # Write vertices and faces for each color group
                for color_idx, (_, voxels_in_group) in enumerate(color_groups.items()):
                    if include_colors:
                        f.write(f"\nusemtl material_{color_idx}\n")

                    for grid_pos in voxels_in_group:
                        world_pos = self.grid_to_world(grid_pos)

                        # Write 8 vertices for this cube
                        for offset in cube_offsets:
                            vx = world_pos[0] + offset[0]
                            vy = world_pos[1] + offset[1]
                            vz = world_pos[2] + offset[2]
                            f.write(f"v {vx:.4f} {vy:.4f} {vz:.4f}\n")

                        # Write 6 faces for this cube
                        for face in cube_faces:
                            f.write(f"f {vertex_index + face[0] - 1} "
                                   f"{vertex_index + face[1] - 1} "
                                   f"{vertex_index + face[2] - 1} "
                                   f"{vertex_index + face[3] - 1}\n")

                        vertex_index += 8

            return True

        except Exception as e:
            print(f"Error exporting to OBJ: {e}")
            return False

    def _write_mtl_file(self, filepath: str, colors):
        """Write material file for OBJ export."""
        with open(filepath, 'w') as f:
            f.write("# JARVIS Voxel Editor Materials\n\n")

            for idx, color in enumerate(colors):
                f.write(f"newmtl material_{idx}\n")
                f.write(f"Kd {color[0]:.4f} {color[1]:.4f} {color[2]:.4f}\n")
                f.write(f"Ka {color[0]*0.3:.4f} {color[1]*0.3:.4f} {color[2]*0.3:.4f}\n")
                f.write(f"Ks 0.5 0.5 0.5\n")
                f.write(f"Ns 50.0\n")
                f.write(f"d 1.0\n")
                f.write(f"illum 2\n\n")

    def get_visible_voxels(self, camera_pos: Tuple[float, float, float],
                          camera_dir: Tuple[float, float, float],
                          fov: float = 60.0,
                          max_distance: float = 100.0) -> List[Tuple[int, int, int]]:
        """
        Get list of voxels visible from camera (frustum culling).

        Simple distance and direction based culling for performance.

        Args:
            camera_pos: Camera world position
            camera_dir: Camera look direction (normalized)
            fov: Field of view in degrees
            max_distance: Maximum render distance

        Returns:
            List of grid positions for visible voxels
        """
        visible = []
        cos_half_fov = math.cos(math.radians(fov / 2.0))

        for grid_pos, voxel in self.voxels.items():
            # Distance check
            dx = voxel.position[0] - camera_pos[0]
            dy = voxel.position[1] - camera_pos[1]
            dz = voxel.position[2] - camera_pos[2]

            dist_sq = dx*dx + dy*dy + dz*dz

            if dist_sq > max_distance * max_distance:
                continue

            # Direction check (cone culling)
            dist = math.sqrt(dist_sq)
            if dist > 0.1:
                dir_x = dx / dist
                dir_y = dy / dist
                dir_z = dz / dist

                dot = dir_x * camera_dir[0] + dir_y * camera_dir[1] + dir_z * camera_dir[2]

                # Add margin for voxel size
                margin = self.grid_size / dist
                if dot < cos_half_fov - margin:
                    continue

            visible.append(grid_pos)

        return visible

    def get_neighbor_mask(self, grid_pos: Tuple[int, int, int]) -> int:
        """
        Get bitmask of which neighbors exist (for hidden face culling).

        Bit positions:
        0: +X, 1: -X, 2: +Y, 3: -Y, 4: +Z, 5: -Z

        Returns:
            6-bit mask where 1 means neighbor exists
        """
        mask = 0
        neighbors = [
            (1, 0, 0),   # +X
            (-1, 0, 0),  # -X
            (0, 1, 0),   # +Y
            (0, -1, 0),  # -Y
            (0, 0, 1),   # +Z
            (0, 0, -1),  # -Z
        ]

        for i, offset in enumerate(neighbors):
            neighbor_pos = (
                grid_pos[0] + offset[0],
                grid_pos[1] + offset[1],
                grid_pos[2] + offset[2]
            )
            if neighbor_pos in self.voxels:
                mask |= (1 << i)

        return mask


if __name__ == "__main__":
    # Test the voxel engine
    print("Voxel Engine Test")
    print("=" * 50)

    engine = VoxelEngine()

    # Place some voxels
    engine.place_voxel((0, 0, 0))
    engine.place_voxel((1, 0, 0))
    engine.place_voxel((0, 1, 0))
    print(f"Placed 3 voxels. Total: {engine.get_voxel_count()}")

    # Test undo
    engine.undo()
    print(f"After undo: {engine.get_voxel_count()}")

    # Test redo
    engine.redo()
    print(f"After redo: {engine.get_voxel_count()}")

    # Test save/load
    engine.save_to_file("test_voxels.json")
    print("Saved to test_voxels.json")

    engine2 = VoxelEngine()
    engine2.load_from_file("test_voxels.json")
    print(f"Loaded: {engine2.get_voxel_count()} voxels")

    # Cleanup
    import os
    os.remove("test_voxels.json")
    print("Test complete!")
