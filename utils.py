"""
Utilities Module for JARVIS Voxel Editor
=========================================
Math helpers, filters, and common utilities.
"""

import numpy as np
import math
import time
from typing import Tuple, List, Optional
from dataclasses import dataclass


class OneEuroFilter:
    """
    One Euro Filter for smooth input filtering.

    This filter adapts its cutoff frequency based on signal velocity:
    - Slow movements get heavy smoothing (low cutoff)
    - Fast movements get minimal smoothing (high cutoff)

    Reference: http://cristal.univ-lille.fr/~casiez/1euro/
    """

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.5, d_cutoff: float = 1.0):
        """
        Initialize the One Euro Filter.

        Args:
            min_cutoff: Minimum cutoff frequency (higher = less smoothing)
            beta: Speed coefficient (higher = faster response to quick movements)
            d_cutoff: Derivative cutoff frequency
        """
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff

        self.x_prev: Optional[float] = None
        self.dx_prev: float = 0.0
        self.t_prev: Optional[float] = None

    def _smoothing_factor(self, t_e: float, cutoff: float) -> float:
        """Calculate smoothing factor alpha."""
        r = 2.0 * math.pi * cutoff * t_e
        return r / (r + 1.0)

    def _exponential_smoothing(self, a: float, x: float, x_prev: float) -> float:
        """Apply exponential smoothing."""
        return a * x + (1.0 - a) * x_prev

    def filter(self, x: float, t: Optional[float] = None) -> float:
        """
        Filter a single value.

        Args:
            x: Input value
            t: Timestamp (optional, uses current time if not provided)

        Returns:
            Filtered value
        """
        if t is None:
            t = time.time()

        if self.x_prev is None:
            self.x_prev = x
            self.t_prev = t
            return x

        # Time delta
        t_e = t - self.t_prev
        if t_e <= 0:
            t_e = 1.0 / 60.0  # Default to 60 FPS

        # Derivative
        dx = (x - self.x_prev) / t_e

        # Filter derivative
        a_d = self._smoothing_factor(t_e, self.d_cutoff)
        dx_hat = self._exponential_smoothing(a_d, dx, self.dx_prev)

        # Adaptive cutoff
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)

        # Filter signal
        a = self._smoothing_factor(t_e, cutoff)
        x_hat = self._exponential_smoothing(a, x, self.x_prev)

        # Update state
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t

        return x_hat

    def reset(self):
        """Reset the filter state."""
        self.x_prev = None
        self.dx_prev = 0.0
        self.t_prev = None


class OneEuroFilter3D:
    """One Euro Filter for 3D coordinates."""

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.5, d_cutoff: float = 1.0):
        self.filter_x = OneEuroFilter(min_cutoff, beta, d_cutoff)
        self.filter_y = OneEuroFilter(min_cutoff, beta, d_cutoff)
        self.filter_z = OneEuroFilter(min_cutoff, beta, d_cutoff)

    def filter(self, x: float, y: float, z: float, t: Optional[float] = None) -> Tuple[float, float, float]:
        """Filter 3D coordinates."""
        return (
            self.filter_x.filter(x, t),
            self.filter_y.filter(y, t),
            self.filter_z.filter(z, t)
        )

    def reset(self):
        """Reset all filters."""
        self.filter_x.reset()
        self.filter_y.reset()
        self.filter_z.reset()


class VelocityTracker:
    """
    Track velocity of a position over time.
    Used for gesture detection (scatter, swipe).
    """

    def __init__(self, history_size: int = 10):
        self.history_size = history_size
        self.positions: List[Tuple[float, float, float]] = []
        self.timestamps: List[float] = []

    def add_position(self, x: float, y: float, z: float = 0.0, t: Optional[float] = None):
        """Add a position sample."""
        if t is None:
            t = time.time()

        self.positions.append((x, y, z))
        self.timestamps.append(t)

        # Keep only recent history
        if len(self.positions) > self.history_size:
            self.positions.pop(0)
            self.timestamps.pop(0)

    def get_velocity(self) -> Tuple[float, float, float]:
        """Get current velocity (units per second)."""
        if len(self.positions) < 2:
            return (0.0, 0.0, 0.0)

        # Use positions at start and end of window
        p1 = self.positions[0]
        p2 = self.positions[-1]
        t1 = self.timestamps[0]
        t2 = self.timestamps[-1]

        dt = t2 - t1
        if dt <= 0:
            return (0.0, 0.0, 0.0)

        return (
            (p2[0] - p1[0]) / dt,
            (p2[1] - p1[1]) / dt,
            (p2[2] - p1[2]) / dt
        )

    def get_speed(self) -> float:
        """Get magnitude of velocity."""
        v = self.get_velocity()
        return math.sqrt(v[0]**2 + v[1]**2 + v[2]**2)

    def get_direction(self) -> Optional[Tuple[float, float, float]]:
        """Get normalized direction of movement."""
        speed = self.get_speed()
        if speed < 0.001:
            return None
        v = self.get_velocity()
        return (v[0] / speed, v[1] / speed, v[2] / speed)

    def reset(self):
        """Clear velocity history."""
        self.positions.clear()
        self.timestamps.clear()


def lerp(a: float, b: float, t: float) -> float:
    """Linear interpolation."""
    return a + (b - a) * t


def lerp3(a: Tuple[float, float, float], b: Tuple[float, float, float], t: float) -> Tuple[float, float, float]:
    """Linear interpolation for 3D points."""
    return (
        lerp(a[0], b[0], t),
        lerp(a[1], b[1], t),
        lerp(a[2], b[2], t)
    )


def ease_in_out(t: float, power: float = 2.0) -> float:
    """Ease-in-out interpolation curve."""
    if t < 0.5:
        return 0.5 * pow(2 * t, power)
    else:
        return 1.0 - 0.5 * pow(2 * (1 - t), power)


def clamp(value: float, min_val: float, max_val: float) -> float:
    """Clamp a value between min and max."""
    return max(min_val, min(max_val, value))


def map_range(value: float, in_min: float, in_max: float, out_min: float, out_max: float) -> float:
    """Map a value from one range to another."""
    return (value - in_min) * (out_max - out_min) / (in_max - in_min) + out_min


def snap_to_grid(value: float, grid_size: float) -> float:
    """Snap a value to the nearest grid point."""
    return round(value / grid_size) * grid_size


def snap_to_grid_3d(pos: Tuple[float, float, float], grid_size: float) -> Tuple[float, float, float]:
    """Snap a 3D position to the nearest grid point."""
    return (
        snap_to_grid(pos[0], grid_size),
        snap_to_grid(pos[1], grid_size),
        snap_to_grid(pos[2], grid_size)
    )


def distance_3d(p1: Tuple[float, float, float], p2: Tuple[float, float, float]) -> float:
    """Calculate 3D Euclidean distance."""
    return math.sqrt(
        (p2[0] - p1[0])**2 +
        (p2[1] - p1[1])**2 +
        (p2[2] - p1[2])**2
    )


def normalize_3d(v: Tuple[float, float, float]) -> Tuple[float, float, float]:
    """Normalize a 3D vector."""
    length = math.sqrt(v[0]**2 + v[1]**2 + v[2]**2)
    if length < 0.0001:
        return (0.0, 0.0, 0.0)
    return (v[0] / length, v[1] / length, v[2] / length)


def quantize_direction(direction: Tuple[float, float, float]) -> Tuple[int, int, int]:
    """
    Quantize a direction vector to the nearest axis.
    Returns one of: (1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)
    """
    abs_x = abs(direction[0])
    abs_y = abs(direction[1])
    abs_z = abs(direction[2])

    if abs_x >= abs_y and abs_x >= abs_z:
        return (1 if direction[0] > 0 else -1, 0, 0)
    elif abs_y >= abs_x and abs_y >= abs_z:
        return (0, 1 if direction[1] > 0 else -1, 0)
    else:
        return (0, 0, 1 if direction[2] > 0 else -1)


def rotation_matrix_y(angle: float) -> np.ndarray:
    """Create a rotation matrix around Y axis."""
    c = math.cos(angle)
    s = math.sin(angle)
    return np.array([
        [c, 0, s],
        [0, 1, 0],
        [-s, 0, c]
    ], dtype=np.float32)


def rotation_matrix_x(angle: float) -> np.ndarray:
    """Create a rotation matrix around X axis."""
    c = math.cos(angle)
    s = math.sin(angle)
    return np.array([
        [1, 0, 0],
        [0, c, -s],
        [0, s, c]
    ], dtype=np.float32)


def rotation_matrix_z(angle: float) -> np.ndarray:
    """Create a rotation matrix around Z axis."""
    c = math.cos(angle)
    s = math.sin(angle)
    return np.array([
        [c, -s, 0],
        [s, c, 0],
        [0, 0, 1]
    ], dtype=np.float32)


def rotate_point_around_center(
    point: Tuple[float, float, float],
    center: Tuple[float, float, float],
    angle: float,
    axis: str = 'y'
) -> Tuple[float, float, float]:
    """Rotate a point around a center point."""
    # Translate to origin
    p = np.array([
        point[0] - center[0],
        point[1] - center[1],
        point[2] - center[2]
    ], dtype=np.float32)

    # Create rotation matrix
    if axis == 'x':
        R = rotation_matrix_x(angle)
    elif axis == 'y':
        R = rotation_matrix_y(angle)
    else:
        R = rotation_matrix_z(angle)

    # Rotate
    p_rotated = R @ p

    # Translate back
    return (
        p_rotated[0] + center[0],
        p_rotated[1] + center[1],
        p_rotated[2] + center[2]
    )


@dataclass
class Ray:
    """A ray with origin and direction."""
    origin: Tuple[float, float, float]
    direction: Tuple[float, float, float]


def screen_to_ray(
    screen_x: float,
    screen_y: float,
    screen_width: int,
    screen_height: int,
    view_matrix: np.ndarray,
    projection_matrix: np.ndarray,
    camera_position: Tuple[float, float, float]
) -> Ray:
    """
    Convert screen coordinates to a world-space ray (Phase 3).

    Args:
        screen_x: X coordinate in pixels (0 = left)
        screen_y: Y coordinate in pixels (0 = top)
        screen_width: Screen width in pixels
        screen_height: Screen height in pixels
        view_matrix: 4x4 view matrix
        projection_matrix: 4x4 projection matrix
        camera_position: Camera world position

    Returns:
        Ray from camera through the screen point
    """
    # Convert screen to normalized device coordinates (-1 to 1)
    ndc_x = (2.0 * screen_x / screen_width) - 1.0
    ndc_y = 1.0 - (2.0 * screen_y / screen_height)  # Flip Y for OpenGL

    # Create inverse matrices
    try:
        inv_projection = np.linalg.inv(projection_matrix)
        inv_view = np.linalg.inv(view_matrix)
    except np.linalg.LinAlgError:
        # Return a default forward ray if matrices are singular
        return Ray(origin=camera_position, direction=(0.0, 0.0, -1.0))

    # Unproject near and far points
    near_ndc = np.array([ndc_x, ndc_y, -1.0, 1.0])
    far_ndc = np.array([ndc_x, ndc_y, 1.0, 1.0])

    # Transform to clip space, then to view space
    near_clip = inv_projection @ near_ndc
    far_clip = inv_projection @ far_ndc

    # Perspective divide
    if abs(near_clip[3]) > 1e-8:
        near_clip = near_clip / near_clip[3]
    if abs(far_clip[3]) > 1e-8:
        far_clip = far_clip / far_clip[3]

    # Transform to world space
    near_world = inv_view @ near_clip
    far_world = inv_view @ far_clip

    # Perspective divide again
    if abs(near_world[3]) > 1e-8:
        near_world = near_world / near_world[3]
    if abs(far_world[3]) > 1e-8:
        far_world = far_world / far_world[3]

    # Calculate ray direction
    direction = np.array([
        far_world[0] - near_world[0],
        far_world[1] - near_world[1],
        far_world[2] - near_world[2]
    ])

    # Normalize direction
    length = np.linalg.norm(direction)
    if length > 1e-8:
        direction = direction / length
    else:
        direction = np.array([0.0, 0.0, -1.0])

    return Ray(
        origin=camera_position,
        direction=(float(direction[0]), float(direction[1]), float(direction[2]))
    )


def ray_plane_intersection(
    ray: Ray,
    plane_point: Tuple[float, float, float],
    plane_normal: Tuple[float, float, float]
) -> Optional[Tuple[float, float, float]]:
    """
    Calculate ray-plane intersection point (Phase 3).

    Args:
        ray: The ray to test
        plane_point: A point on the plane
        plane_normal: Normal vector of the plane

    Returns:
        Intersection point or None if ray is parallel to plane
    """
    # Calculate denominator (dot product of ray direction and plane normal)
    denom = (ray.direction[0] * plane_normal[0] +
             ray.direction[1] * plane_normal[1] +
             ray.direction[2] * plane_normal[2])

    if abs(denom) < 1e-8:
        # Ray is parallel to plane
        return None

    # Calculate t parameter
    diff = (plane_point[0] - ray.origin[0],
            plane_point[1] - ray.origin[1],
            plane_point[2] - ray.origin[2])

    t = (diff[0] * plane_normal[0] +
         diff[1] * plane_normal[1] +
         diff[2] * plane_normal[2]) / denom

    if t < 0:
        # Intersection is behind the ray origin
        return None

    # Calculate intersection point
    return (
        ray.origin[0] + ray.direction[0] * t,
        ray.origin[1] + ray.direction[1] * t,
        ray.origin[2] + ray.direction[2] * t
    )


def ray_box_intersection(
    ray: Ray,
    box_min: Tuple[float, float, float],
    box_max: Tuple[float, float, float]
) -> Optional[float]:
    """
    Calculate ray-box intersection distance.
    Returns distance to intersection point, or None if no intersection.
    """
    t_min = 0.0
    t_max = float('inf')

    for i in range(3):
        if abs(ray.direction[i]) < 1e-8:
            # Ray is parallel to slab
            if ray.origin[i] < box_min[i] or ray.origin[i] > box_max[i]:
                return None
        else:
            inv_d = 1.0 / ray.direction[i]
            t1 = (box_min[i] - ray.origin[i]) * inv_d
            t2 = (box_max[i] - ray.origin[i]) * inv_d

            if t1 > t2:
                t1, t2 = t2, t1

            t_min = max(t_min, t1)
            t_max = min(t_max, t2)

            if t_min > t_max:
                return None

    return t_min if t_min >= 0 else None


class Timer:
    """Simple timer utility."""

    def __init__(self):
        self.start_time: Optional[float] = None
        self.running = False

    def start(self):
        """Start the timer."""
        self.start_time = time.time()
        self.running = True

    def stop(self):
        """Stop the timer."""
        self.running = False

    def reset(self):
        """Reset the timer."""
        self.start_time = None
        self.running = False

    def elapsed(self) -> float:
        """Get elapsed time in seconds."""
        if self.start_time is None:
            return 0.0
        return time.time() - self.start_time

    def progress(self, total_duration: float) -> float:
        """Get progress as 0-1 value."""
        if total_duration <= 0:
            return 1.0
        return clamp(self.elapsed() / total_duration, 0.0, 1.0)


if __name__ == "__main__":
    # Test One Euro Filter
    print("Testing One Euro Filter...")
    filter_1d = OneEuroFilter(min_cutoff=1.0, beta=0.5)

    # Simulate noisy input
    for i in range(20):
        noisy_value = math.sin(i * 0.3) + np.random.normal(0, 0.1)
        filtered = filter_1d.filter(noisy_value)
        print(f"Input: {noisy_value:.3f}, Filtered: {filtered:.3f}")

    print("\nTesting Velocity Tracker...")
    tracker = VelocityTracker(history_size=5)
    for i in range(10):
        tracker.add_position(i * 10, i * 5, 0)
        time.sleep(0.05)
    print(f"Velocity: {tracker.get_velocity()}")
    print(f"Speed: {tracker.get_speed():.2f}")
