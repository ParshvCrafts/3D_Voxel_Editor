"""
Configuration System for JARVIS-Style 3D Voxel Editor
======================================================
All adjustable parameters in one place for easy customization.
"""

from dataclasses import dataclass, field
from typing import Tuple, List, Dict
from enum import Enum, auto
import numpy as np


# ============ MODE SYSTEM (Phase 3) ============
class EditorMode(Enum):
    """Editor modes that change how gestures are interpreted."""
    NAVIGATE = auto()   # Camera controls, no building
    BUILD = auto()      # Place blocks (default)
    ERASE = auto()      # Delete blocks
    SELECT = auto()     # Selection mode
    PHYSICS = auto()    # Scatter/recombine mode


@dataclass
class Colors:
    """JARVIS holographic color palette."""
    # Primary colors (RGB normalized 0-1 for OpenGL)
    CYAN: Tuple[float, float, float] = (0.0, 0.831, 1.0)  # Electric cyan #00D4FF
    BLUE: Tuple[float, float, float] = (0.0, 0.478, 1.0)  # Neon blue #007AFF
    WHITE: Tuple[float, float, float] = (0.9, 0.95, 1.0)  # Holographic white
    ORANGE: Tuple[float, float, float] = (1.0, 0.4, 0.0)  # Alert orange
    RED: Tuple[float, float, float] = (1.0, 0.2, 0.2)  # Warning red
    GREEN: Tuple[float, float, float] = (0.0, 1.0, 0.4)  # Neon green
    PURPLE: Tuple[float, float, float] = (0.6, 0.2, 1.0)  # Purple
    PINK: Tuple[float, float, float] = (1.0, 0.2, 0.6)  # Pink
    GOLD: Tuple[float, float, float] = (1.0, 0.84, 0.0)  # Gold
    YELLOW: Tuple[float, float, float] = (1.0, 1.0, 0.2)  # Bright yellow (Phase 2)

    # Background
    BACKGROUND: Tuple[float, float, float] = (0.02, 0.04, 0.08)  # Near-black with blue tint

    # UI Colors (0-255 for OpenCV)
    UI_CYAN: Tuple[int, int, int] = (255, 212, 0)  # BGR format
    UI_GREEN: Tuple[int, int, int] = (0, 255, 100)
    UI_RED: Tuple[int, int, int] = (0, 80, 255)
    UI_ORANGE: Tuple[int, int, int] = (0, 165, 255)
    UI_WHITE: Tuple[int, int, int] = (255, 255, 255)

    @staticmethod
    def get_palette() -> List[Tuple[float, float, float]]:
        """Get the color palette for radial menu."""
        return [
            Colors.CYAN,
            Colors.BLUE,
            Colors.WHITE,
            Colors.GREEN,
            Colors.PURPLE,
            Colors.PINK,
            Colors.ORANGE,
            Colors.GOLD,
        ]


@dataclass
class WindowConfig:
    """Window and display settings."""
    WIDTH: int = 1280
    HEIGHT: int = 720
    TITLE: str = "JARVIS Voxel Editor"
    TARGET_FPS: int = 60
    VSYNC: bool = True
    FULLSCREEN: bool = False


@dataclass
class CameraConfig:
    """3D Camera settings."""
    FOV: float = 50.0  # Field of view in degrees (slightly wider)
    NEAR_PLANE: float = 0.1
    FAR_PLANE: float = 1000.0
    # Phase 7: Camera closer to match YouTube (camera.position.z = 20 in YouTube)
    # But our scale is different, so we use Z=15 for better visibility
    INITIAL_POSITION: Tuple[float, float, float] = (0.0, 5.0, 15.0)
    LOOK_AT: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    UP_VECTOR: Tuple[float, float, float] = (0.0, 1.0, 0.0)
    # Camera breathing effect
    BREATHING_ENABLED: bool = True
    BREATHING_AMPLITUDE: float = 0.05
    BREATHING_SPEED: float = 0.5


@dataclass
class HandTrackingConfig:
    """Hand tracking settings."""
    MAX_HANDS: int = 2
    DETECTION_CONFIDENCE: float = 0.7
    TRACKING_CONFIDENCE: float = 0.7
    WEBCAM_WIDTH: int = 1280
    WEBCAM_HEIGHT: int = 720
    WEBCAM_INDEX: int = 0
    FLIP_HORIZONTAL: bool = True

    # Phase 5: YouTube-style coordinate mapping formula
    # Formula: (0.5 - x) * X_SCALE, (0.5 - y) * Y_SCALE
    # This centers the world at screen center and provides intuitive mapping
    USE_YOUTUBE_MAPPING: bool = True  # Use YouTube-style mapping (Phase 5)
    # Phase 7: Reduced scale to bring blocks closer to center of view
    YOUTUBE_X_SCALE: float = 18.0     # X multiplier: (0.5 - x) * 18 (was 25)
    YOUTUBE_Y_SCALE: float = 12.0     # Y multiplier: (0.5 - y) * 12 (was 18)
    YOUTUBE_Z_SCALE: float = 18.0     # Z multiplier for depth

    # Legacy coordinate mapping (normalized -> world space)
    WORLD_BOUNDS_X: Tuple[float, float] = (-15.0, 15.0)
    WORLD_BOUNDS_Y: Tuple[float, float] = (-10.0, 15.0)
    WORLD_BOUNDS_Z: Tuple[float, float] = (-15.0, 15.0)

    # Z-depth estimation parameters
    HAND_SIZE_BASELINE: float = 150.0  # Baseline hand size in pixels at reference depth
    DEPTH_SCALE_FACTOR: float = 10.0  # How much z changes per hand size ratio


@dataclass
class OneEuroFilterConfig:
    """One Euro Filter parameters for smoothing."""
    MIN_CUTOFF: float = 1.0  # Minimum cutoff frequency (smoothness)
    BETA: float = 0.5  # Speed coefficient (how fast filter adapts)
    D_CUTOFF: float = 1.0  # Derivative cutoff frequency


@dataclass
class GestureConfig:
    """Gesture recognition thresholds and timing."""
    # Pinch detection with hysteresis (Phase 3: enhanced)
    PINCH_THRESHOLD: float = 40.0  # Distance in pixels for pinch detection
    PINCH_RELEASE_THRESHOLD: float = 60.0  # Hysteresis - higher threshold for release

    # Additional hysteresis thresholds (Phase 3)
    FINGERS_UP_HYSTERESIS: float = 0.15  # Delay before finger state changes

    # Hold times (seconds) - Phase 8: Definitive gesture timing
    PLACE_HOLD_TIME: float = 0.5      # Right pinch -> continuous building
    DELETE_HOLD_TIME: float = 0.5     # Left point + right pinch -> erasing
    GRAB_HOLD_TIME: float = 0.5       # Right fist -> drag structure
    ROTATE_HOLD_TIME: float = 1.0     # Both palms -> rotate structure
    SCATTER_HOLD_TIME: float = 0.8    # Left thumb down -> burst (one-shot)
    RESTORE_HOLD_TIME: float = 0.8    # Left thumb up -> restore (one-shot)
    FULL_RESET_HOLD_TIME: float = 5.0 # Right thumb up -> clear all voxels + reset transform
    RECOMBINE_HOLD_TIME: float = 2.0  # Legacy
    COLOR_MENU_HOLD_TIME: float = 0.8 # Legacy (disabled)

    # Scatter detection
    SCATTER_VELOCITY_THRESHOLD: float = 800.0  # Pixels per second
    SCATTER_VELOCITY_FRAMES: int = 5  # Frames to track for velocity

    # Swipe detection
    SWIPE_MIN_DISTANCE: float = 150.0  # Minimum swipe distance in pixels
    SWIPE_MAX_TIME: float = 0.5  # Maximum time for swipe gesture

    # Cooldowns (seconds) - Phase 3: Enhanced cooldown system
    ACTION_COOLDOWN: float = 0.3  # General action cooldown
    SCATTER_COOLDOWN: float = 1.0  # Scatter-specific cooldown
    PLACE_COOLDOWN: float = 0.5   # Cooldown after placing a block (Phase 3)
    DELETE_COOLDOWN: float = 0.5  # Cooldown after deleting a block (Phase 3)

    # Debounce settings (Phase 3)
    DEBOUNCE_TIME: float = 0.1  # Minimum time before same gesture can trigger again
    EDGE_TRIGGER_RESET_TIME: float = 0.2  # Time pinch must be released before re-triggering

    # Two-hand rotation
    ROTATION_SNAP_ANGLE: float = 90.0  # Degrees

    # Selection box
    MIN_SELECTION_SIZE: float = 20.0  # Minimum selection box size


@dataclass
class VoxelConfig:
    """Voxel engine settings."""
    GRID_SIZE: float = 1.0  # Size of each voxel
    INITIAL_GRID_EXTENT: int = 10  # Initial grid display extent
    MAX_VOXELS: int = 10000  # Maximum voxels for performance

    # Selection
    SELECTION_HIGHLIGHT_COLOR: Tuple[float, float, float] = (1.0, 1.0, 0.5)
    GHOST_ALPHA: float = 0.5  # Transparency of preview ghost


@dataclass
class PhysicsConfig:
    """Physics simulation for scatter effect."""
    GRAVITY: Tuple[float, float, float] = (0.0, -9.8, 0.0)
    DRAG: float = 0.98  # Velocity damping per frame
    EXPLOSION_FORCE: float = 15.0  # Base force for scatter
    EXPLOSION_VARIATION: float = 0.3  # Random variation (0-1)
    ANGULAR_VELOCITY_MAX: float = 5.0  # Max tumble speed

    # Recombine animation
    RECOMBINE_DURATION: float = 2.0  # Seconds
    RECOMBINE_EASE_POWER: float = 2.0  # Ease-in-out power


@dataclass
class RenderConfig:
    """Rendering settings."""
    # Bloom effect
    BLOOM_ENABLED: bool = True
    BLOOM_THRESHOLD: float = 0.8
    BLOOM_INTENSITY: float = 1.5
    BLOOM_BLUR_PASSES: int = 5

    # Fresnel effect
    FRESNEL_POWER: float = 2.0
    FRESNEL_BIAS: float = 0.1

    # Grid
    GRID_FADE_DISTANCE: float = 50.0
    GRID_LINE_WIDTH: float = 1.0

    # Preview wireframes (place/delete/grab previews)
    PREVIEW_WIREFRAME_WIDTH: float = 4.0

    # Cursor/reticle
    CURSOR_SIZE: float = 0.3
    CURSOR_ROTATION_SPEED: float = 1.0  # Radians per second
    CURSOR_PULSE_SPEED: float = 2.0  # Hz

    # Particles
    PARTICLE_COUNT: int = 100
    PARTICLE_SIZE: float = 0.02

    # Scanlines
    SCANLINE_ENABLED: bool = False
    SCANLINE_DENSITY: float = 400.0


@dataclass
class UIConfig:
    """UI element settings."""
    # Demo defaults
    PRESENTATION_MODE: bool = True
    SHOW_HAND_TRACKING_OVERLAY: bool = True
    SHOW_HAND_LABELS: bool = False

    # Loading circle
    LOADING_CIRCLE_RADIUS: int = 40
    LOADING_CIRCLE_THICKNESS: int = 3

    # Radial menu
    RADIAL_MENU_RADIUS: float = 2.0  # World units
    RADIAL_MENU_ITEM_SIZE: float = 0.3

    # HUD
    SHOW_FPS: bool = True
    SHOW_GESTURE_STATE: bool = True
    SHOW_HAND_CONFIDENCE: bool = True
    SHOW_HELP_OVERLAY: bool = False
    SHOW_WEBCAM_STATUS: bool = True
    SHOW_GESTURE_LABELS: bool = False
    SHOW_STATUS_FPS: bool = False
    SHOW_STATUS_VOXEL_COUNT: bool = True
    SHOW_STATUS_MODE: bool = True
    SHOW_STATUS_SCATTER: bool = False
    SHOW_STATUS_COLOR_SWATCH: bool = True
    VOXEL_COUNTER_TEXT_THICKNESS: int = 1
    STATUS_TEXT_THICKNESS: int = 2


@dataclass
class HistoryConfig:
    """Undo/redo history settings."""
    MAX_UNDO_STEPS: int = 50


@dataclass
class DebugConfig:
    """Debug overlay settings (Phase 3)."""
    SHOW_DEBUG_OVERLAY: bool = False  # Toggle with D key
    ENABLE_HAND_TRACKER_DIAGNOSTICS: bool = False
    SHOW_GESTURE_STATE: bool = True
    SHOW_HAND_DATA: bool = True
    SHOW_CURSOR_COORDS: bool = True
    SHOW_VOXEL_STATS: bool = True
    SHOW_MODE_INDICATOR: bool = True
    SHOW_FPS_GRAPH: bool = False
    DEBUG_FONT_SIZE: float = 0.5
    DEBUG_LINE_HEIGHT: int = 20
    DEBUG_PADDING: int = 10


@dataclass
class ARConfig:
    """AR overlay settings (Phase 4)."""
    # AR Mode
    ENABLED: bool = True  # Use webcam as fullscreen background

    # Holographic transparency (0.0 = invisible, 1.0 = opaque)
    VOXEL_OPACITY: float = 0.85  # Voxel transparency for holographic effect
    GRID_OPACITY: float = 0.4   # Grid transparency
    CURSOR_OPACITY: float = 0.9  # Cursor visibility

    # Background dimming
    BACKGROUND_DIM: float = 0.0  # How much to dim the webcam (0 = none, 1 = black)

    # Holographic effects
    EDGE_GLOW_INTENSITY: float = 1.2  # Glow on voxel edges
    SCANLINE_ALPHA: float = 0.1       # Subtle scanlines for hologram effect

    # Fallback if webcam unavailable
    FALLBACK_TO_DARK: bool = True  # Use dark background if no webcam


@dataclass
class Config:
    """Master configuration containing all sub-configs."""
    colors: Colors = field(default_factory=Colors)
    window: WindowConfig = field(default_factory=WindowConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    hand_tracking: HandTrackingConfig = field(default_factory=HandTrackingConfig)
    filter: OneEuroFilterConfig = field(default_factory=OneEuroFilterConfig)
    gesture: GestureConfig = field(default_factory=GestureConfig)
    voxel: VoxelConfig = field(default_factory=VoxelConfig)
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    render: RenderConfig = field(default_factory=RenderConfig)
    ui: UIConfig = field(default_factory=UIConfig)
    history: HistoryConfig = field(default_factory=HistoryConfig)
    debug: DebugConfig = field(default_factory=DebugConfig)
    ar: ARConfig = field(default_factory=ARConfig)


# Global configuration instance
CONFIG = Config()


if __name__ == "__main__":
    # Print configuration for verification
    print("JARVIS Voxel Editor Configuration")
    print("=" * 50)
    print(f"Window: {CONFIG.window.WIDTH}x{CONFIG.window.HEIGHT}")
    print(f"Target FPS: {CONFIG.window.TARGET_FPS}")
    print(f"Color Palette: {len(CONFIG.colors.get_palette())} colors")
    print(f"Max Voxels: {CONFIG.voxel.MAX_VOXELS}")
    print(f"Bloom Enabled: {CONFIG.render.BLOOM_ENABLED}")
