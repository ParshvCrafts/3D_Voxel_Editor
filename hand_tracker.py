"""
Enhanced Hand Tracking Module for 3D Voxel Editor
==================================================
Extends the base hand tracking with:
- 3D coordinate extraction (including z-depth)
- Velocity tracking for gesture detection
- Pinch detection
- Two-hand gesture support
- World coordinate mapping
"""

import cv2
import mediapipe as mp
import numpy as np
import math
import time
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from utils import OneEuroFilter3D, VelocityTracker, map_range, clamp
from config import CONFIG


@dataclass
class Landmark3D:
    """3D landmark with world-mapped coordinates."""
    id: int
    x: float  # Screen x (pixels)
    y: float  # Screen y (pixels)
    z: float  # Normalized depth from MediaPipe
    world_x: float  # World space x
    world_y: float  # World space y
    world_z: float  # World space z (estimated from hand size)


@dataclass
class Hand3D:
    """Enhanced hand data with 3D information."""
    hand_type: str  # "Left" or "Right"
    landmarks: Dict[int, Landmark3D] = field(default_factory=dict)
    fingers_up: List[int] = field(default_factory=lambda: [0, 0, 0, 0, 0])
    confidence: float = 0.0

    # Derived data
    center_world: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    index_tip_world: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    thumb_tip_world: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    palm_normal: Tuple[float, float, float] = (0.0, 0.0, 1.0)

    # Gesture helpers
    pinch_distance: float = 100.0  # Distance between index and thumb
    is_pinching: bool = False
    hand_size: float = 100.0  # Approximate hand size in pixels

    # Velocity tracking
    index_velocity: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    palm_velocity: Tuple[float, float, float] = (0.0, 0.0, 0.0)


class HandTracker3D:
    """
    Enhanced hand tracker with 3D capabilities for voxel editing.

    Features:
    - Full 3D landmark extraction
    - World coordinate mapping
    - Smooth filtering (One Euro Filter)
    - Velocity tracking for gesture detection
    - Pinch detection with hysteresis
    - Two-hand gesture support
    """

    # MediaPipe landmark indices
    WRIST = 0
    THUMB_CMC = 1
    THUMB_MCP = 2
    THUMB_IP = 3
    THUMB_TIP = 4
    INDEX_MCP = 5
    INDEX_PIP = 6
    INDEX_DIP = 7
    INDEX_TIP = 8
    MIDDLE_MCP = 9
    MIDDLE_PIP = 10
    MIDDLE_DIP = 11
    MIDDLE_TIP = 12
    RING_MCP = 13
    RING_PIP = 14
    RING_DIP = 15
    RING_TIP = 16
    PINKY_MCP = 17
    PINKY_PIP = 18
    PINKY_DIP = 19
    PINKY_TIP = 20

    FINGER_TIPS = [THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]
    FINGER_PIPS = [THUMB_IP, INDEX_PIP, MIDDLE_PIP, RING_PIP, PINKY_PIP]

    def __init__(self):
        """Initialize the enhanced hand tracker."""
        cfg = CONFIG.hand_tracking

        # MediaPipe setup
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=cfg.MAX_HANDS,
            min_detection_confidence=cfg.DETECTION_CONFIDENCE,
            min_tracking_confidence=cfg.TRACKING_CONFIDENCE
        )
        self.mp_draw = mp.solutions.drawing_utils

        # Drawing specs for visualization
        self.landmark_spec = mp.solutions.drawing_utils.DrawingSpec(
            thickness=2, circle_radius=2, color=(0, 255, 255)  # Cyan
        )
        self.connection_spec = mp.solutions.drawing_utils.DrawingSpec(
            thickness=2, color=(0, 200, 255)
        )

        # Image dimensions (updated each frame)
        self.img_width = cfg.WEBCAM_WIDTH
        self.img_height = cfg.WEBCAM_HEIGHT

        # One Euro Filters for smoothing (one per hand per key point)
        filter_cfg = CONFIG.filter
        self.filters: Dict[str, Dict[str, OneEuroFilter3D]] = {
            "Left": {
                "index_tip": OneEuroFilter3D(filter_cfg.MIN_CUTOFF, filter_cfg.BETA, filter_cfg.D_CUTOFF),
                "thumb_tip": OneEuroFilter3D(filter_cfg.MIN_CUTOFF, filter_cfg.BETA, filter_cfg.D_CUTOFF),
                "center": OneEuroFilter3D(filter_cfg.MIN_CUTOFF, filter_cfg.BETA, filter_cfg.D_CUTOFF),
            },
            "Right": {
                "index_tip": OneEuroFilter3D(filter_cfg.MIN_CUTOFF, filter_cfg.BETA, filter_cfg.D_CUTOFF),
                "thumb_tip": OneEuroFilter3D(filter_cfg.MIN_CUTOFF, filter_cfg.BETA, filter_cfg.D_CUTOFF),
                "center": OneEuroFilter3D(filter_cfg.MIN_CUTOFF, filter_cfg.BETA, filter_cfg.D_CUTOFF),
            }
        }

        # Velocity trackers
        velocity_frames = CONFIG.gesture.SCATTER_VELOCITY_FRAMES
        self.velocity_trackers: Dict[str, Dict[str, VelocityTracker]] = {
            "Left": {
                "index": VelocityTracker(velocity_frames),
                "palm": VelocityTracker(velocity_frames),
            },
            "Right": {
                "index": VelocityTracker(velocity_frames),
                "palm": VelocityTracker(velocity_frames),
            }
        }

        # Pinch state with hysteresis
        self.pinch_state: Dict[str, bool] = {"Left": False, "Right": False}

        # YouTube-style simple exponential smoothing storage
        # Replaces One Euro Filter for simpler, more predictable smoothing
        self.smoothed_positions: Dict[str, Dict[str, Tuple[float, float, float]]] = {
            "Left": {},
            "Right": {}
        }

        # Results cache
        self.hands_data: List[Hand3D] = []
        self.results = None
        self.draw_hand_overlay = CONFIG.ui.SHOW_HAND_TRACKING_OVERLAY
        self.show_hand_labels = CONFIG.ui.SHOW_HAND_LABELS
        self.enable_diagnostics = CONFIG.debug.ENABLE_HAND_TRACKER_DIAGNOSTICS

    def process_frame(self, img: np.ndarray, draw: bool = True) -> Tuple[np.ndarray, List[Hand3D]]:
        """
        Process a frame and extract 3D hand data.

        Args:
            img: BGR image from webcam
            draw: Whether to draw landmarks on image

        Returns:
            Tuple of (processed image, list of Hand3D objects)
        """
        self.img_height, self.img_width = img.shape[:2]

        # DIAGNOSTIC: Print frame info sent to MediaPipe
        if not hasattr(self, '_debug_frame_count'):
            self._debug_frame_count = 0
        self._debug_frame_count += 1
        
        if self.enable_diagnostics and self._debug_frame_count % 60 == 1:  # Print every 60 frames (~1 second)
            print(f"\n[DIAG] Frame sent to MediaPipe: {self.img_width}x{self.img_height}, flipped: NO (raw frame)")

        # CRITICAL FIX: Do NOT flip before MediaPipe processing!
        # YouTube approach: MediaPipe processes RAW frame, display is mirrored via shader only.
        # This ensures coordinates align correctly with mirrored display.
        # The shader's u_mirror_x handles the visual mirroring.

        # Convert to RGB for MediaPipe
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(img_rgb)

        # Clear previous data
        self.hands_data = []

        if self.results.multi_hand_landmarks:
            for hand_idx, hand_landmarks in enumerate(self.results.multi_hand_landmarks):
                # Get hand type from MediaPipe
                hand_type = "Right"
                confidence = 0.0
                if self.results.multi_handedness:
                    classification = self.results.multi_handedness[hand_idx].classification[0]
                    # Strip whitespace and control characters (fixes Windows 'Right\r' bug)
                    hand_type = classification.label.strip()
                    confidence = classification.score

                # Extract 3D hand data
                hand_3d = self._extract_hand_3d(hand_landmarks, hand_type, confidence)
                self.hands_data.append(hand_3d)

        # CRITICAL FIX: Mirror frame AFTER MediaPipe processing but BEFORE drawing HUD
        # This matches YouTube's approach:
        # - MediaPipe sees raw frame (correct hand detection)
        # - Display shows mirrored frame (natural mirror view)
        # - HUD is drawn on mirrored frame (text readable, skeleton aligns)
        # - No shader mirroring needed (prevents double-mirror issues)
        img = cv2.flip(img, 1)  # Horizontal flip for mirror effect
        
        # Now draw HUD on the mirrored frame
        # HUD coordinates need to be mirrored to match the flipped frame
        if draw and self.draw_hand_overlay and self.results.multi_hand_landmarks:
            for hand_idx, hand_landmarks in enumerate(self.results.multi_hand_landmarks):
                hand_3d = self.hands_data[hand_idx]
                self._draw_hand_mirrored(img, hand_landmarks, hand_3d)

        return img, self.hands_data

    def _extract_hand_3d(self, hand_landmarks, hand_type: str, confidence: float) -> Hand3D:
        """Extract 3D hand data from MediaPipe landmarks."""
        hand = Hand3D(hand_type=hand_type, confidence=confidence)
        cfg = CONFIG.hand_tracking
        current_time = time.time()

        # DIAGNOSTIC: Print coordinate info for index finger tip (landmark 8)
        index_lm = hand_landmarks.landmark[8]  # Index fingertip
        if self.enable_diagnostics and self._debug_frame_count % 60 == 1:
            raw_x = index_lm.x
            screen_x = int(raw_x * self.img_width)
            world_x = (0.5 - raw_x) * cfg.YOUTUBE_X_SCALE
            mirror_screen_x = self.img_width - screen_x
            print(f"[DIAG] Hand: {hand_type}")
            print(f"[DIAG]   Raw MediaPipe X: {raw_x:.3f} (0=left, 1=right of RAW image)")
            print(f"[DIAG]   Screen X (raw): {screen_x} px")
            print(f"[DIAG]   Screen X (mirrored): {mirror_screen_x} px")
            print(f"[DIAG]   World X: {world_x:.2f} (formula: (0.5 - {raw_x:.3f}) * {cfg.YOUTUBE_X_SCALE})")

        # Extract all landmarks with 3D coordinates
        screen_coords = []
        for idx, lm in enumerate(hand_landmarks.landmark):
            # Screen coordinates (pixels)
            x_px = int(lm.x * self.img_width)
            y_px = int(lm.y * self.img_height)
            z_norm = lm.z  # MediaPipe's normalized z (depth relative to wrist)

            screen_coords.append((x_px, y_px))

            # Map to world coordinates
            # Phase 5: Use YouTube-style mapping formula for better hand-cursor alignment
            if cfg.USE_YOUTUBE_MAPPING:
                # YouTube formula: (0.5 - x) * scale, (0.5 - y) * scale
                # This centers world origin at screen center
                world_x = (0.5 - lm.x) * cfg.YOUTUBE_X_SCALE
                world_y = (0.5 - lm.y) * cfg.YOUTUBE_Y_SCALE
                # CRITICAL FIX: YouTube uses gz = 0 ALWAYS for 2D placement
                # All blocks are placed on the same Z plane
                # The 3D structure can be rotated for viewing, but placement is 2D
                world_z = 0.0  # Always 0 for 2D placement
            else:
                # Legacy mapping using bounds
                world_x = map_range(lm.x, 0.0, 1.0, cfg.WORLD_BOUNDS_X[0], cfg.WORLD_BOUNDS_X[1])
                world_y = map_range(1.0 - lm.y, 0.0, 1.0, cfg.WORLD_BOUNDS_Y[0], cfg.WORLD_BOUNDS_Y[1])
                world_z = 0.0  # Always 0 for 2D placement

            hand.landmarks[idx] = Landmark3D(
                id=idx,
                x=x_px, y=y_px, z=z_norm,
                world_x=world_x, world_y=world_y, world_z=world_z
            )

        # Calculate hand size for depth estimation
        wrist = hand.landmarks[self.WRIST]
        middle_mcp = hand.landmarks[self.MIDDLE_MCP]
        hand.hand_size = math.sqrt(
            (wrist.x - middle_mcp.x)**2 + (wrist.y - middle_mcp.y)**2
        )

        # Estimate world Z based on hand size
        depth_ratio = cfg.HAND_SIZE_BASELINE / max(hand.hand_size, 50.0)
        estimated_z = (depth_ratio - 1.0) * cfg.DEPTH_SCALE_FACTOR

        # CRITICAL FIX: Keep world_z = 0 for all landmarks (2D placement)
        # YouTube approach: gz = 0 always - no Z depth variation
        # The voxel structure can be rotated in 3D for viewing, but placement is strictly 2D
        for lm in hand.landmarks.values():
            lm.world_z = 0.0  # Always 0 for 2D placement

        # Get key points with filtering
        index_tip = hand.landmarks[self.INDEX_TIP]
        thumb_tip = hand.landmarks[self.THUMB_TIP]

        # Calculate hand center
        center_x = sum(lm.world_x for lm in hand.landmarks.values()) / 21
        center_y = sum(lm.world_y for lm in hand.landmarks.values()) / 21
        center_z = 0.0  # Always 0 for 2D placement

        # Apply YouTube-style simple exponential smoothing
        # YouTube reference (index_advance.html lines 106-110):
        # smoothedLandmarks[label][i].x += (p.x - smoothedLandmarks[label][i].x) * 0.45;
        # smoothedLandmarks[label][i].y += (p.y - smoothedLandmarks[label][i].y) * 0.45;
        # smoothedLandmarks[label][i].z += (p.z - smoothedLandmarks[label][i].z) * 0.1;
        smoothing = self.smoothed_positions[hand_type]
        
        # Smoothing factors from YouTube
        SMOOTH_XY = 0.45  # Responsive for X/Y
        SMOOTH_Z = 0.1    # More stable for Z
        
        # Initialize if first frame
        if "index_tip" not in smoothing:
            smoothing["index_tip"] = (index_tip.world_x, index_tip.world_y, index_tip.world_z)
            smoothing["thumb_tip"] = (thumb_tip.world_x, thumb_tip.world_y, thumb_tip.world_z)
            smoothing["center"] = (center_x, center_y, center_z)
        else:
            # Apply exponential smoothing
            prev = smoothing["index_tip"]
            smoothing["index_tip"] = (
                prev[0] + (index_tip.world_x - prev[0]) * SMOOTH_XY,
                prev[1] + (index_tip.world_y - prev[1]) * SMOOTH_XY,
                prev[2] + (index_tip.world_z - prev[2]) * SMOOTH_Z
            )
            
            prev = smoothing["thumb_tip"]
            smoothing["thumb_tip"] = (
                prev[0] + (thumb_tip.world_x - prev[0]) * SMOOTH_XY,
                prev[1] + (thumb_tip.world_y - prev[1]) * SMOOTH_XY,
                prev[2] + (thumb_tip.world_z - prev[2]) * SMOOTH_Z
            )
            
            prev = smoothing["center"]
            smoothing["center"] = (
                prev[0] + (center_x - prev[0]) * SMOOTH_XY,
                prev[1] + (center_y - prev[1]) * SMOOTH_XY,
                prev[2] + (center_z - prev[2]) * SMOOTH_Z
            )
        
        hand.index_tip_world = smoothing["index_tip"]
        hand.thumb_tip_world = smoothing["thumb_tip"]
        hand.center_world = smoothing["center"]

        # Update velocity trackers
        trackers = self.velocity_trackers[hand_type]
        trackers["index"].add_position(*hand.index_tip_world, current_time)
        trackers["palm"].add_position(*hand.center_world, current_time)

        hand.index_velocity = trackers["index"].get_velocity()
        hand.palm_velocity = trackers["palm"].get_velocity()

        # Calculate pinch distance
        hand.pinch_distance = math.sqrt(
            (index_tip.x - thumb_tip.x)**2 + (index_tip.y - thumb_tip.y)**2
        )

        # Pinch detection with hysteresis
        pinch_cfg = CONFIG.gesture
        if self.pinch_state[hand_type]:
            # Currently pinching - use release threshold
            hand.is_pinching = hand.pinch_distance < pinch_cfg.PINCH_RELEASE_THRESHOLD
        else:
            # Not pinching - use pinch threshold
            hand.is_pinching = hand.pinch_distance < pinch_cfg.PINCH_THRESHOLD
        self.pinch_state[hand_type] = hand.is_pinching

        # Detect fingers up
        hand.fingers_up = self._detect_fingers_up(hand)

        # Calculate palm normal (approximate)
        hand.palm_normal = self._calculate_palm_normal(hand)

        return hand

    def _detect_fingers_up(self, hand: Hand3D) -> List[int]:
        """Detect which fingers are extended.
        
        CRITICAL: Thumb uses X-axis (horizontal), other fingers use Y-axis (vertical).
        
        Thumb Detection (X-axis, handedness-aware):
        - RIGHT hand: thumb extended = thumb_tip.x < thumb_ip.x (tip LEFT of IP)
        - LEFT hand: thumb extended = thumb_tip.x > thumb_ip.x (tip RIGHT of IP)
        
        Other Fingers (Y-axis):
        - EXTENDED: tip.y < pip.y (tip ABOVE pip in screen coords)
        - CURLED: tip.y > pip.y (tip BELOW pip)
        
        MediaPipe coordinate system:
        - X = 0 at LEFT, X = 1 at RIGHT of image
        - Y = 0 at TOP, Y = 1 at BOTTOM of image
        """
        fingers = []

        # THUMB DETECTION - X-axis based, handedness-aware
        # Thumb moves horizontally, not vertically like other fingers
        thumb_tip = hand.landmarks[self.THUMB_TIP]      # Landmark 4
        thumb_ip = hand.landmarks[self.THUMB_IP]        # Landmark 3
        
        # For RIGHT hand: thumb points LEFT when extended (tip.x < ip.x)
        # For LEFT hand: thumb points RIGHT when extended (tip.x > ip.x)
        if hand.hand_type == "Right":
            # Right hand: extended thumb tip is to the LEFT of IP joint
            thumb_extended = thumb_tip.x < thumb_ip.x
        else:
            # Left hand: extended thumb tip is to the RIGHT of IP joint
            thumb_extended = thumb_tip.x > thumb_ip.x
        
        fingers.append(1 if thumb_extended else 0)

        # OTHER FINGERS - Y-axis based (tip above pip = extended)
        for tip_id, pip_id in zip(self.FINGER_TIPS[1:], self.FINGER_PIPS[1:]):
            tip = hand.landmarks[tip_id]
            pip = hand.landmarks[pip_id]
            # tip.y < pip.y means tip is ABOVE pip = finger EXTENDED
            fingers.append(1 if tip.y < pip.y else 0)

        return fingers

    def _calculate_palm_normal(self, hand: Hand3D) -> Tuple[float, float, float]:
        """Calculate approximate palm normal vector."""
        # Use wrist, index MCP, and pinky MCP to define palm plane
        wrist = hand.landmarks[self.WRIST]
        index_mcp = hand.landmarks[self.INDEX_MCP]
        pinky_mcp = hand.landmarks[self.PINKY_MCP]

        # Vectors on palm plane
        v1 = np.array([
            index_mcp.world_x - wrist.world_x,
            index_mcp.world_y - wrist.world_y,
            index_mcp.world_z - wrist.world_z
        ])
        v2 = np.array([
            pinky_mcp.world_x - wrist.world_x,
            pinky_mcp.world_y - wrist.world_y,
            pinky_mcp.world_z - wrist.world_z
        ])

        # Cross product gives normal
        normal = np.cross(v1, v2)
        length = np.linalg.norm(normal)
        if length > 0.001:
            normal = normal / length
        else:
            normal = np.array([0.0, 0.0, 1.0])

        return tuple(normal)

    def _draw_hand_mirrored(self, img: np.ndarray, hand_landmarks, hand_3d: Hand3D):
        """Draw JARVIS-style cyber hand skeleton on MIRRORED frame.

        CRITICAL: Frame has already been flipped with cv2.flip(img, 1).
        We must mirror X coordinates to match: mirrored_x = width - raw_x
        
        This approach:
        - MediaPipe processes RAW frame (accurate detection)
        - Frame is flipped for display (mirror effect)
        - HUD drawn with mirrored coords (skeleton aligns with hand)
        - Text is readable (no shader mirroring)
        """
        # JARVIS-style hand skeleton connections
        CYBER_CONNECTIONS = [
            [0, 1], [1, 2], [2, 3], [3, 4],      # Thumb
            [0, 5], [5, 6], [6, 7], [7, 8],      # Index
            [9, 10], [10, 11], [11, 12],         # Middle
            [13, 14], [14, 15], [15, 16],        # Ring
            [0, 17], [17, 18], [18, 19], [19, 20],  # Pinky
            [5, 9], [9, 13], [13, 17], [0, 5]    # Palm cross-connections
        ]

        # BGR format for OpenCV
        glow_color = (255, 240, 0)  # Cyan in BGR (#00f0ff)
        
        # Helper to mirror X coordinate for the flipped frame
        def mirror_x(x):
            return self.img_width - x

        # Draw connections with mirrored coordinates
        for connection in CYBER_CONNECTIONS:
            pt1 = hand_3d.landmarks[connection[0]]
            pt2 = hand_3d.landmarks[connection[1]]
            p1 = (mirror_x(int(pt1.x)), int(pt1.y))
            p2 = (mirror_x(int(pt2.x)), int(pt2.y))
            cv2.line(img, p1, p2, glow_color, 2)

        # Draw joint markers
        FINGERTIP_IDS = [4, 8, 12, 16, 20]

        for idx, lm in hand_3d.landmarks.items():
            x, y = mirror_x(int(lm.x)), int(lm.y)

            if idx in FINGERTIP_IDS:
                cv2.rectangle(img, (x - 6, y - 6), (x + 6, y + 6), glow_color, 1)
            else:
                cv2.rectangle(img, (x - 2, y - 2), (x + 2, y + 2), (255, 255, 255), -1)

        # Draw a subtle pinch reticle instead of a solid debug blob.
        if hand_3d.is_pinching:
            index_tip = hand_3d.landmarks[self.INDEX_TIP]
            thumb_tip = hand_3d.landmarks[self.THUMB_TIP]
            mid_x = mirror_x(int((index_tip.x + thumb_tip.x) // 2))
            mid_y = int((index_tip.y + thumb_tip.y) // 2)
            cv2.circle(img, (mid_x, mid_y), 13, glow_color, 2)
            cv2.circle(img, (mid_x, mid_y), 3, (255, 255, 255), cv2.FILLED)

        if self.show_hand_labels:
            wrist = hand_3d.landmarks[self.WRIST]
            finger_count = sum(hand_3d.fingers_up)
            fingers = hand_3d.fingers_up
            finger_str = (
                f"[{'T' if fingers[0] else '_'}{'I' if fingers[1] else '_'}"
                f"{'M' if fingers[2] else '_'}{'R' if fingers[3] else '_'}"
                f"{'P' if fingers[4] else '_'}]"
            )
            text = f"{hand_3d.hand_type}: {finger_count} {finger_str}"
            label_x = mirror_x(int(wrist.x)) - 60
            cv2.putText(img, text, (label_x, int(wrist.y) - 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, glow_color, 2)

    def _draw_hand(self, img: np.ndarray, hand_landmarks, hand_3d: Hand3D):
        """Draw JARVIS-style cyber hand skeleton (DEPRECATED - use _draw_hand_mirrored).
        
        This method draws with RAW coordinates, expecting shader to mirror.
        Kept for reference but _draw_hand_mirrored is now used.
        """
        CYBER_CONNECTIONS = [
            [0, 1], [1, 2], [2, 3], [3, 4],
            [0, 5], [5, 6], [6, 7], [7, 8],
            [9, 10], [10, 11], [11, 12],
            [13, 14], [14, 15], [15, 16],
            [0, 17], [17, 18], [18, 19], [19, 20],
            [5, 9], [9, 13], [13, 17], [0, 5]
        ]

        glow_color = (255, 240, 0)

        for connection in CYBER_CONNECTIONS:
            pt1 = hand_3d.landmarks[connection[0]]
            pt2 = hand_3d.landmarks[connection[1]]
            p1 = (int(pt1.x), int(pt1.y))
            p2 = (int(pt2.x), int(pt2.y))
            cv2.line(img, p1, p2, glow_color, 2)

        FINGERTIP_IDS = [4, 8, 12, 16, 20]

        for idx, lm in hand_3d.landmarks.items():
            x, y = int(lm.x), int(lm.y)
            if idx in FINGERTIP_IDS:
                cv2.rectangle(img, (x - 6, y - 6), (x + 6, y + 6), glow_color, 1)
            else:
                cv2.rectangle(img, (x - 2, y - 2), (x + 2, y + 2), (255, 255, 255), -1)

        if hand_3d.is_pinching:
            index_tip = hand_3d.landmarks[self.INDEX_TIP]
            thumb_tip = hand_3d.landmarks[self.THUMB_TIP]
            mid_x = int((index_tip.x + thumb_tip.x) // 2)
            mid_y = int((index_tip.y + thumb_tip.y) // 2)
            cv2.circle(img, (mid_x, mid_y), 15, (0, 255, 0), cv2.FILLED)

        wrist = hand_3d.landmarks[self.WRIST]
        finger_count = sum(hand_3d.fingers_up)
        text = f"{hand_3d.hand_type}: {finger_count}"
        cv2.putText(img, text, (int(wrist.x) - 30, int(wrist.y) - 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, glow_color, 2)

    def get_left_hand(self) -> Optional[Hand3D]:
        """Get the left hand if detected."""
        for hand in self.hands_data:
            if hand.hand_type == "Left":
                return hand
        return None

    def get_right_hand(self) -> Optional[Hand3D]:
        """Get the right hand if detected."""
        for hand in self.hands_data:
            if hand.hand_type == "Right":
                return hand
        return None

    def get_primary_hand(self) -> Optional[Hand3D]:
        """Get the primary (right) hand, or left if right not available."""
        right = self.get_right_hand()
        if right:
            return right
        return self.get_left_hand()

    def is_two_hands_detected(self) -> bool:
        """Check if both hands are detected."""
        return len(self.hands_data) >= 2

    def get_two_hand_rotation(self) -> Optional[float]:
        """
        Calculate rotation angle from two-hand gesture.
        Returns angle in radians, or None if not applicable.
        """
        if not self.is_two_hands_detected():
            return None

        left = self.get_left_hand()
        right = self.get_right_hand()

        if not left or not right:
            return None

        # Use wrist positions to calculate angle
        left_wrist = left.landmarks[self.WRIST]
        right_wrist = right.landmarks[self.WRIST]

        dx = right_wrist.x - left_wrist.x
        dy = right_wrist.y - left_wrist.y

        return math.atan2(dy, dx)

    def reset_filters(self, hand_type: str = None):
        """Reset smoothing filters."""
        if hand_type:
            for filter_obj in self.filters[hand_type].values():
                filter_obj.reset()
        else:
            for hand_filters in self.filters.values():
                for filter_obj in hand_filters.values():
                    filter_obj.reset()

    def release(self):
        """Release MediaPipe resources."""
        self.hands.close()


def main():
    """Test the enhanced hand tracker."""
    import time

    cap = cv2.VideoCapture(CONFIG.hand_tracking.WEBCAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CONFIG.hand_tracking.WEBCAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CONFIG.hand_tracking.WEBCAM_HEIGHT)

    tracker = HandTracker3D()

    print("=" * 60)
    print("ENHANCED 3D HAND TRACKING TEST")
    print("=" * 60)
    print("Testing features:")
    print("- 3D coordinate extraction")
    print("- Pinch detection (green circle when pinching)")
    print("- Finger counting")
    print("Press 'q' to quit")
    print("=" * 60)

    pTime = 0

    while True:
        success, img = cap.read()
        if not success:
            break

        # Process frame
        img, hands = tracker.process_frame(img, draw=True)

        # Display hand info
        for hand in hands:
            # Show world coordinates
            info_y = 30 if hand.hand_type == "Right" else 200
            cv2.putText(img, f"{hand.hand_type} Hand:", (10, info_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, CONFIG.colors.UI_WHITE, 1)
            cv2.putText(img, f"  World: ({hand.index_tip_world[0]:.1f}, {hand.index_tip_world[1]:.1f}, {hand.index_tip_world[2]:.1f})",
                       (10, info_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, CONFIG.colors.UI_CYAN, 1)
            cv2.putText(img, f"  Pinch: {hand.pinch_distance:.0f}px {'(PINCHING)' if hand.is_pinching else ''}",
                       (10, info_y + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                       CONFIG.colors.UI_GREEN if hand.is_pinching else CONFIG.colors.UI_WHITE, 1)
            cv2.putText(img, f"  Fingers: {hand.fingers_up}",
                       (10, info_y + 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, CONFIG.colors.UI_WHITE, 1)

        # FPS
        cTime = time.time()
        fps = 1 / (cTime - pTime) if (cTime - pTime) > 0 else 0
        pTime = cTime
        h, w = img.shape[:2]
        cv2.putText(img, f"FPS: {int(fps)}", (w - 100, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, CONFIG.colors.UI_WHITE, 2)

        cv2.imshow("3D Hand Tracking Test", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    tracker.release()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
