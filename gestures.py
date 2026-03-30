"""
Gesture Recognition State Machine for JARVIS Voxel Editor
==========================================================
Robust gesture detection with confirmation timers and conflict resolution.

Phase 3 Enhancements:
- Edge-triggered gesture detection (prevents multi-placement)
- Mode system integration
- Enhanced hysteresis for thresholds
- Cooldown and debounce system
"""

import time
import math
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Callable, Dict, Any
from hand_tracker import Hand3D, HandTracker3D
from utils import Timer, VelocityTracker
from config import CONFIG, EditorMode


class GestureState(Enum):
    """Possible gesture states."""
    IDLE = auto()
    PLACING = auto()
    DELETING = auto()
    EXTENDING = auto()
    ROTATING = auto()
    COLOR_MENU = auto()
    SCATTERING = auto()
    RECOMBINING = auto()
    SELECTING = auto()
    SWIPING_LEFT = auto()
    SWIPING_RIGHT = auto()
    # New two-hand gesture states for Phase 2
    PANNING = auto()      # Camera pan with two hands
    ZOOMING = auto()      # Camera zoom with two-hand pinch
    TWO_HAND_PLACING = auto()   # Two-hand coordinated placement
    TWO_HAND_DELETING = auto()  # Two-hand coordinated deletion
    # YouTube-style continuous building (Phase 6 fix)
    CONTINUOUS_BUILDING = auto()  # Continuous voxel placement while pinching
    # Phase 7: Color and disco gestures
    COLOR_TOGGLE = auto()  # Left victory sign - toggle color
    DISCO_MODE = auto()    # Right-hand disco start/freeze/restore controls
    # Phase 8: Definitive gesture states
    GRABBING = auto()      # Right fist - drag structure
    SCATTER_CHARGING = auto()   # Left thumb down charging
    RESTORE_CHARGING = auto()   # Left thumb up charging
    # Phase 9.3: Reset gesture
    RESETTING = auto()     # Both fists - reset position/rotation
    # Phase 15: Full reset gesture
    FULL_RESETTING = auto()  # Right thumb up only - clear all voxels (5s timer)


@dataclass
class GestureEvent:
    """Event generated when a gesture is confirmed."""
    gesture_type: GestureState
    position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    direction: Optional[Tuple[float, float, float]] = None
    rotation_angle: float = 0.0
    selected_color_index: int = -1
    selection_box: Optional[Tuple[float, float, float, float, float, float]] = None  # min_x, min_y, min_z, max_x, max_y, max_z
    # Camera control data
    pan_delta: Tuple[float, float] = (0.0, 0.0)  # Camera pan X, Y
    zoom_delta: float = 0.0  # Camera zoom amount
    # Two-hand info
    left_hand_pos: Optional[Tuple[float, float, float]] = None
    right_hand_pos: Optional[Tuple[float, float, float]] = None
    extra_data: Dict[str, Any] = field(default_factory=dict)


class GestureRecognizer:
    """
    State machine for gesture recognition.

    Handles:
    - Gesture detection from hand data
    - Confirmation timers for hold-based gestures
    - Velocity-based gesture detection (scatter, swipe)
    - Two-hand gesture support
    - Conflict resolution and debouncing
    - Edge-triggered detection (Phase 3)
    - Mode-based gesture interpretation (Phase 3)
    """

    def __init__(self):
        """Initialize the gesture recognizer."""
        self.state = GestureState.IDLE
        self.previous_state = GestureState.IDLE

        # Reference to voxel engine for direct block checking during delete
        self._voxel_engine = None

        # Timers for hold-based gestures
        self.place_timer = Timer()
        self.delete_timer = Timer()
        self.recombine_timer = Timer()
        self.color_menu_timer = Timer()
        
        # Phase 8: New gesture timers for definitive gesture system
        self.grab_timer = Timer()      # Right fist grab (500ms)
        self.rotate_timer = Timer()    # Both palms rotate (1000ms)
        self.scatter_timer = Timer()   # Left thumb down (800ms)
        self.restore_timer = Timer()   # Left thumb up (800ms)
        self.reset_timer = Timer()     # PHASE 9.3: Both fists reset (1000ms)
        self.full_reset_timer = Timer()  # Right thumb-up full reset (5000ms)
        
        # One-shot gesture flags (prevent re-trigger while held)
        self.scatter_triggered = False
        self.restore_triggered = False
        self.reset_triggered = False   # PHASE 9.3: Prevent re-trigger of reset
        self.full_reset_triggered = False  # Prevent re-trigger of full reset
        
        # Grab gesture state
        self.grab_offset = (0.0, 0.0, 0.0)  # Offset from hand to voxel group
        self.is_grabbing = False
        
        # PHASE 9.3: Delete cursor position (follows LEFT index during delete)
        self.delete_cursor_position: Optional[Tuple[float, float, float]] = None

        # Cooldown tracking
        self.last_action_time = 0.0
        self.last_scatter_time = 0.0

        # Phase 3: Per-action cooldown tracking
        self.last_place_time = 0.0
        self.last_delete_time = 0.0

        # Gesture tracking data
        self.pinch_start_position: Optional[Tuple[float, float, float]] = None
        self.extend_start_position: Optional[Tuple[float, float, float]] = None
        self.swipe_start_position: Optional[Tuple[float, float]] = None
        self.swipe_start_time: float = 0.0

        # Selection box
        self.selection_start: Optional[Tuple[float, float, float]] = None
        self.selection_end: Optional[Tuple[float, float, float]] = None

        # Two-hand rotation
        self.initial_rotation_angle: Optional[float] = None
        self.current_rotation_angle: float = 0.0

        # Two-hand gesture tracking (Phase 2)
        self.two_hand_start_distance: float = 0.0  # For zoom gesture
        self.pan_start_positions: Optional[Tuple[Tuple[float, float, float], Tuple[float, float, float]]] = None
        self.gesture_pending_timer = Timer()  # 0.3s window for second hand
        self.pending_gesture_type: Optional[GestureState] = None
        self.left_hand_gesture: str = ""  # Current gesture of left hand
        self.right_hand_gesture: str = ""  # Current gesture of right hand

        # Color menu
        self.color_menu_active = False
        self.selected_color_index = 0

        # YouTube-style continuous building state (Phase 6 fix)
        # Reference: index_advance.html lines 131-135, 302-331
        self.is_building = False           # Currently in continuous build mode
        self.build_start_pos = None        # Grid position where build started
        self.active_axis = None            # Locked axis ('x', 'y', or None)
        self.sketch_keys: set = set()      # Set of grid positions in current sketch
        self.last_grid_pos = None          # Last grid position to avoid duplicates

        # Phase 7: Color toggle and disco mode (YouTube reference)
        # Victory sign: index + middle extended, ring + pinky curled
        self.left_victory_was_active = False   # Edge detection for left victory
        self.right_victory_was_active = False  # Edge detection for right victory
        self.right_palm_was_active = False     # Edge detection for right palm freeze
        self.disco_mode_active = False         # Disco mode state
        self.current_color_index = 0           # Current selected color index
        
        # Phase 9: Strict edge detection for color toggle
        self.left_hand_was_visible = False     # Track if left hand was visible last frame
        self.right_hand_was_visible = False    # Track if right hand was visible last frame
        self.last_color_toggle_time = 0.0      # Cooldown for color toggle
        self.last_disco_toggle_time = 0.0      # Cooldown for disco start/freeze/restore

        # Phase 3: Edge-triggering state
        # These track the PREVIOUS frame's state to detect rising/falling edges
        self.prev_pinch_state: Dict[str, bool] = {"Left": False, "Right": False}
        self.prev_fingers_up: Dict[str, List[int]] = {"Left": [0, 0, 0, 0, 0], "Right": [0, 0, 0, 0, 0]}
        self.pinch_released_time: Dict[str, float] = {"Left": 0.0, "Right": 0.0}
        self.gesture_completed_this_cycle = False  # Prevent re-triggering in same gesture

        # Phase 3: Mode system
        self.current_mode = EditorMode.BUILD  # Default mode

        # Phase 3: Debug info
        self.debug_info: Dict[str, Any] = {
            "state": "IDLE",
            "progress": 0.0,
            "mode": "BUILD",
            "pinch_distance": 0.0,
            "fingers_up": [0, 0, 0, 0, 0],
            "edge_triggered": False
        }

        # Event callbacks
        self.on_gesture_start: Optional[Callable[[GestureState], None]] = None
        self.on_gesture_progress: Optional[Callable[[GestureState, float], None]] = None
        self.on_gesture_complete: Optional[Callable[[GestureEvent], None]] = None
        self.on_gesture_cancel: Optional[Callable[[GestureState], None]] = None

        # Config shortcuts
        self.cfg = CONFIG.gesture

    def set_mode(self, mode: EditorMode):
        """Set the current editor mode (Phase 3)."""
        self.current_mode = mode
        self.debug_info["mode"] = mode.name

    def set_voxel_engine(self, voxel_engine):
        """Set reference to voxel engine for direct block checking during delete."""
        self._voxel_engine = voxel_engine

    def get_mode(self) -> EditorMode:
        """Get the current editor mode."""
        return self.current_mode

    def get_debug_info(self) -> Dict[str, Any]:
        """Get debug information for overlay (Phase 3)."""
        return self.debug_info.copy()

    def _is_disco_active(self) -> bool:
        """Use the voxel engine as the source of truth for disco state when available."""
        if self._voxel_engine is not None:
            return self._voxel_engine.disco_mode
        return self.disco_mode_active

    def _is_victory_sign(self, fingers: List[int]) -> bool:
        """
        Check if hand is showing victory sign (Phase 7).
        
        Victory sign: index + middle extended, ring + pinky curled
        YouTube reference: index_advance.html lines 205, 272
        """
        # fingers = [thumb, index, middle, ring, pinky]
        # Victory: index(1) and middle(1) extended, ring(0) and pinky(0) curled
        return (fingers[1] == 1 and fingers[2] == 1 and 
                fingers[3] == 0 and fingers[4] == 0)
    
    def _is_palm_open(self, fingers: List[int]) -> bool:
        """
        Check if hand is showing open palm (all fingers extended).
        
        YouTube reference: index_advance.html lines 166-167, 200, 270
        """
        # All 4 main fingers extended (thumb can vary)
        return (fingers[1] == 1 and fingers[2] == 1 and 
                fingers[3] == 1 and fingers[4] == 1)
    
    def _is_fist(self, fingers: List[int]) -> bool:
        """
        Check if hand is making a fist (all fingers curled including thumb).
        
        YouTube reference: Used for grab/move gesture.
        """
        # All 5 fingers curled (including thumb)
        return sum(fingers) == 0
    
    def _is_thumb_up(self, hand: Hand3D) -> bool:
        """
        Check if hand is showing thumbs up gesture.
        
        Thumb up: thumb extended (pointing up), all other fingers curled.
        Also check that thumb tip is ABOVE wrist (thumb pointing upward).
        """
        fingers = hand.fingers_up
        # Thumb extended, all others curled
        if fingers != [1, 0, 0, 0, 0]:
            return False
        # Check thumb tip is above wrist (lower Y = higher on screen)
        thumb_tip = hand.landmarks[4]
        wrist = hand.landmarks[0]
        return thumb_tip.y < wrist.y
    
    def _is_thumb_down(self, hand: Hand3D) -> bool:
        """
        Check if hand is showing thumbs down gesture.
        
        Thumb down: thumb extended (pointing down), all other fingers curled.
        Also check that thumb tip is BELOW wrist (thumb pointing downward).
        """
        fingers = hand.fingers_up
        # Thumb extended, all others curled
        if fingers != [1, 0, 0, 0, 0]:
            return False
        # Check thumb tip is below wrist (higher Y = lower on screen)
        thumb_tip = hand.landmarks[4]
        wrist = hand.landmarks[0]
        return thumb_tip.y > wrist.y

    def _left_hand_blocks_full_reset(self, left_hand: Optional[Hand3D]) -> bool:
        """Return True when the left hand is busy with another deliberate gesture."""
        if not left_hand:
            return False

        fingers = left_hand.fingers_up
        left_pointing = fingers[1] == 1 and fingers[2] == 0
        return (
            left_hand.is_pinching or
            self._is_palm_open(fingers) or
            self._is_victory_sign(fingers) or
            left_pointing or
            self._is_thumb_up(left_hand) or
            self._is_thumb_down(left_hand)
        )

    def _is_full_reset_pose(self,
                            left_hand: Optional[Hand3D],
                            right_hand: Optional[Hand3D]) -> bool:
        """Full reset is isolated to a right-hand thumbs-up pose."""
        if not right_hand:
            return False

        if right_hand.is_pinching or not self._is_thumb_up(right_hand):
            return False

        return not self._left_hand_blocks_full_reset(left_hand)

    def _is_pinch_rising_edge(self, hand: Hand3D, current_time: float) -> bool:
        """
        Check if pinch just started (rising edge detection) - Phase 3.

        Returns True only on the frame when pinch transitions from False to True,
        AND sufficient time has passed since last pinch release.
        """
        hand_type = hand.hand_type
        current_pinch = hand.is_pinching
        prev_pinch = self.prev_pinch_state.get(hand_type, False)

        # Detect rising edge (was not pinching, now is pinching)
        is_rising = current_pinch and not prev_pinch

        # Check if enough time has passed since pinch was released
        time_since_release = current_time - self.pinch_released_time.get(hand_type, 0.0)
        edge_reset_ok = time_since_release >= self.cfg.EDGE_TRIGGER_RESET_TIME

        return is_rising and edge_reset_ok

    def _update_edge_tracking(self, hands: List[Hand3D], current_time: float):
        """Update edge-triggering state tracking (Phase 3)."""
        for hand in hands:
            hand_type = hand.hand_type
            current_pinch = hand.is_pinching

            # Track pinch release time
            if self.prev_pinch_state.get(hand_type, False) and not current_pinch:
                # Just released pinch
                self.pinch_released_time[hand_type] = current_time

            # Update previous state for next frame
            self.prev_pinch_state[hand_type] = current_pinch
            self.prev_fingers_up[hand_type] = hand.fingers_up.copy()

    def update(self, tracker: HandTracker3D, hands: List[Hand3D]) -> Optional[GestureEvent]:
        """
        Update gesture state based on current hand data.

        Args:
            tracker: The hand tracker instance
            hands: List of detected hands

        Returns:
            GestureEvent if a gesture was completed, None otherwise
        """
        current_time = time.time()

        # Update debug info
        self.debug_info["state"] = self.state.name
        self.debug_info["mode"] = self.current_mode.name

        # Check general cooldown
        if current_time - self.last_action_time < self.cfg.ACTION_COOLDOWN:
            self._update_edge_tracking(hands, current_time)
            return None

        primary_hand = tracker.get_primary_hand()
        event = None

        # Update debug info from primary hand
        if primary_hand:
            self.debug_info["pinch_distance"] = primary_hand.pinch_distance
            self.debug_info["fingers_up"] = primary_hand.fingers_up.copy()

        # State machine logic
        if self.state == GestureState.IDLE:
            event = self._handle_idle_state(tracker, hands, primary_hand, current_time)

        elif self.state == GestureState.PLACING:
            event = self._handle_placing_state(primary_hand, current_time)

        elif self.state == GestureState.DELETING:
            event = self._handle_deleting_state(tracker, primary_hand, current_time)

        elif self.state == GestureState.EXTENDING:
            event = self._handle_extending_state(primary_hand, current_time)

        elif self.state == GestureState.ROTATING:
            event = self._handle_rotating_state(tracker, current_time)

        elif self.state == GestureState.COLOR_MENU:
            event = self._handle_color_menu_state(primary_hand, current_time)

        elif self.state == GestureState.RECOMBINING:
            event = self._handle_recombining_state(primary_hand, current_time)

        elif self.state == GestureState.SELECTING:
            event = self._handle_selecting_state(primary_hand, current_time)

        elif self.state in (GestureState.SWIPING_LEFT, GestureState.SWIPING_RIGHT):
            event = self._handle_swipe_state(primary_hand, current_time)

        # New two-hand gesture state handlers (Phase 2)
        elif self.state == GestureState.PANNING:
            event = self._handle_panning_state(tracker, current_time)

        elif self.state == GestureState.ZOOMING:
            event = self._handle_zooming_state(tracker, current_time)

        elif self.state == GestureState.TWO_HAND_PLACING:
            event = self._handle_two_hand_placing_state(tracker, current_time)

        elif self.state == GestureState.TWO_HAND_DELETING:
            event = self._handle_two_hand_deleting_state(tracker, current_time)

        # Phase 8: New gesture state handlers
        elif self.state == GestureState.GRABBING:
            event = self._handle_grabbing_state(tracker, current_time)
        
        elif self.state == GestureState.SCATTER_CHARGING:
            event = self._handle_scatter_charging_state(tracker, current_time)
        
        elif self.state == GestureState.RESTORE_CHARGING:
            event = self._handle_restore_charging_state(tracker, current_time)
        
        elif self.state == GestureState.FULL_RESETTING:
            event = self._handle_full_resetting_state(tracker, current_time)

        # DISABLED: Velocity-based scatter removed - now uses hold timer
        # if self.current_mode in (EditorMode.BUILD, EditorMode.PHYSICS):
        #     scatter_event = self._check_scatter_gesture(primary_hand, current_time)
        #     if scatter_event:
        #         event = scatter_event

        # Phase 7: Check for victory sign gestures (color toggle and disco mode)
        # These are edge-triggered and work independently of the state machine
        left_hand = tracker.get_left_hand()
        right_hand = tracker.get_right_hand()
        
        # Left victory sign -> Toggle color (STRICT edge-triggered: close→open only)
        # PHASE 9 FIX: Only trigger on deliberate close→open, NOT on hand enter/exit
        if left_hand:
            left_victory = self._is_victory_sign(left_hand.fingers_up)
            
            # Only trigger if:
            # 1. Current frame: victory sign detected
            # 2. Previous frame: hand was visible AND NOT showing victory (was closed)
            # 3. Cooldown has passed
            if (left_victory and 
                self.left_hand_was_visible and 
                not self.left_victory_was_active and
                current_time - self.last_color_toggle_time >= 0.5):  # 500ms cooldown
                # Rising edge from closed to open - toggle color
                self.current_color_index = (self.current_color_index + 1) % 8
                event = GestureEvent(
                    gesture_type=GestureState.COLOR_TOGGLE,
                    selected_color_index=self.current_color_index
                )
                self.last_color_toggle_time = current_time
                print(f"[GESTURE] Color toggle -> index {self.current_color_index}")
            
            self.left_victory_was_active = left_victory
            self.left_hand_was_visible = True
        else:
            # Hand left screen - reset victory state but mark hand as not visible
            # This prevents false trigger when hand re-enters
            self.left_victory_was_active = False
            self.left_hand_was_visible = False
        
        # Right-hand disco controls:
        # - Right victory close->open starts disco when inactive
        # - Right victory close->open restores authored colors when disco is active
        # - Right palm close->open freezes the current disco colors
        if right_hand:
            right_victory = self._is_victory_sign(right_hand.fingers_up)
            right_palm = self._is_palm_open(right_hand.fingers_up)

            disco_toggle_ready = current_time - self.last_disco_toggle_time >= 0.5
            disco_active = self._is_disco_active()

            if (
                right_victory and
                self.right_hand_was_visible and
                not self.right_victory_was_active and
                disco_toggle_ready
            ):
                action = "restore" if disco_active else "start"
                event = GestureEvent(
                    gesture_type=GestureState.DISCO_MODE,
                    extra_data={"action": action}
                )
                self.disco_mode_active = action == "start"
                self.last_disco_toggle_time = current_time
                print(f"[GESTURE] Disco {action.upper()}")
            elif (
                right_palm and
                self.right_hand_was_visible and
                not self.right_palm_was_active and
                disco_active and
                disco_toggle_ready
            ):
                event = GestureEvent(
                    gesture_type=GestureState.DISCO_MODE,
                    extra_data={"action": "freeze"}
                )
                self.disco_mode_active = False
                self.last_disco_toggle_time = current_time
                print("[GESTURE] Disco FREEZE")

            self.right_victory_was_active = right_victory
            self.right_palm_was_active = right_palm
            self.right_hand_was_visible = True
        else:
            self.right_victory_was_active = False
            self.right_palm_was_active = False
            self.right_hand_was_visible = False

        # Update edge tracking for next frame (Phase 3)
        self._update_edge_tracking(hands, current_time)

        if event:
            self.last_action_time = current_time
            self.gesture_completed_this_cycle = True
            self.debug_info["edge_triggered"] = True

            # Update per-action cooldowns (Phase 3)
            if event.gesture_type == GestureState.PLACING:
                self.last_place_time = current_time
            elif event.gesture_type == GestureState.DELETING:
                self.last_delete_time = current_time
        else:
            self.debug_info["edge_triggered"] = False

        # Update debug progress
        self.debug_info["progress"] = self.get_progress()

        return event

    def _handle_idle_state(self, tracker: HandTracker3D, hands: List[Hand3D],
                          primary_hand: Optional[Hand3D], current_time: float) -> Optional[GestureEvent]:
        """Handle the IDLE state - detect gesture starts.
        
        PHASE 15: Updated gesture priority order.
        Priority order (check in this exact order):
        1. RESET (both fists) - highest priority, 1s timer
        2. ROTATE (both palms) - with timer
        2.5. FULL_RESET (right thumb up only, left hand idle) - 5s timer
        3. DELETE (right pinch + left index pointing)
        4. GRAB (right fist alone)
        5. PLACE (left pinch alone, NO right pinch)
        6. SCATTER/RESTORE (left thumb gestures)
        7. COLOR (left victory cycle) - handled separately
        """
        if not primary_hand:
            return None

        # Reset gesture completed flag when back to IDLE
        self.gesture_completed_this_cycle = False

        # Get both hands
        left_hand = tracker.get_left_hand()
        right_hand = tracker.get_right_hand()
        
        # Get finger states
        left_fingers = left_hand.fingers_up if left_hand else [0, 0, 0, 0, 0]
        right_fingers = right_hand.fingers_up if right_hand else [0, 0, 0, 0, 0]
        
        # Compute gesture conditions
        left_fist = self._is_fist(left_fingers)
        right_fist = self._is_fist(right_fingers)
        left_palm = sum(left_fingers) >= 4
        right_palm = sum(right_fingers) >= 4
        left_pinching = left_hand.is_pinching if left_hand else False
        right_pinching = right_hand.is_pinching if right_hand else False
        
        # Left pointing: index up, middle DOWN (not victory)
        left_pointing = left_hand and (left_fingers[1] == 1 and left_fingers[2] == 0)
        
        # Debug output (uncomment to diagnose)
        # print(f"[GESTURE] L_pinch={left_pinching}, R_pinch={right_pinching}, L_point={left_pointing}, R_fist={right_fist}")
        
        # ============ PRIORITY 1: RESET (both fists) ============
        if left_hand and right_hand and left_fist and right_fist:
            if not self.reset_triggered:
                if self.reset_timer.elapsed() == 0:
                    self.reset_timer.start()
                    print("[GESTURE] RESET charging (both fists)")
                
                progress = self.reset_timer.progress(1.0)
                if self.on_gesture_progress:
                    self.on_gesture_progress(GestureState.RESETTING, progress)
                
                if self.reset_timer.elapsed() >= 1.0:
                    self.reset_triggered = True
                    self.reset_timer.reset()
                    print("[GESTURE] RESET COMPLETE")
                    return GestureEvent(gesture_type=GestureState.RESETTING, position=(0, 0, 0))
            return None
        else:
            if self.reset_timer.elapsed() > 0:
                self.reset_timer.reset()
            self.reset_triggered = False
        
        # ============ PRIORITY 2: ROTATE (both palms) ============
        if left_hand and right_hand and left_palm and right_palm:
            if not hasattr(self, 'rotate_timer_started') or not self.rotate_timer_started:
                self.rotate_timer.start()
                self.rotate_timer_started = True
                print("[GESTURE] ROTATE charging (both palms)")
            
            progress = self.rotate_timer.progress(1.0)
            if self.on_gesture_progress:
                self.on_gesture_progress(GestureState.ROTATING, progress)
            
            if self.rotate_timer.elapsed() >= 1.0:
                self._transition_to(GestureState.ROTATING)
                self.initial_rotation_angle = tracker.get_two_hand_rotation()
                self.current_rotation_angle = 0.0
                self.rotate_timer_started = False
                print("[GESTURE] ROTATE ACTIVATED")
            return None
        else:
            if hasattr(self, 'rotate_timer_started') and self.rotate_timer_started:
                self.rotate_timer.reset()
                self.rotate_timer_started = False
        
        # ============ PRIORITY 2.5: FULL_RESET (right thumb up only) - 5s timer ============
        # Checked after rotation and gated away from left-hand actions like placing/deleting.
        if self._is_full_reset_pose(left_hand, right_hand):
            self._transition_to(GestureState.FULL_RESETTING)
            self.full_reset_timer.start()
            self.full_reset_triggered = False
            print("[RESET-1] FULL RESET charging (right thumb up only) - hold 5 seconds")
            return None
        
        # ============ PRIORITY 3: DELETE (right pinch + left index) ============
        if right_pinching and left_pointing:
            if self._is_pinch_rising_edge(right_hand, current_time):
                if current_time - self.last_delete_time >= self.cfg.DELETE_COOLDOWN:
                    self._transition_to(GestureState.DELETING)
                    self.delete_timer.start()
                    self.delete_cursor_position = left_hand.index_tip_world
                    print("[GESTURE] DELETE: Right pinch + Left pointing")
                    return None
        
        # ============ PRIORITY 4: GRAB (right fist alone) ============
        # Only if NOT left pinching (to avoid conflict)
        if right_hand and right_fist and not left_pinching:
            self._transition_to(GestureState.GRABBING)
            self.grab_timer.start()
            self.is_grabbing = False
            print("[GESTURE] GRAB charging (right fist)")
            return None
        
        # ============ PRIORITY 5: PLACE (left pinch ONLY, no right pinch) ============
        # CRITICAL: Only trigger if right hand is NOT pinching
        if left_pinching and not right_pinching:
            if self._is_pinch_rising_edge(left_hand, current_time):
                if current_time - self.last_place_time >= self.cfg.PLACE_COOLDOWN:
                    self._transition_to(GestureState.PLACING)
                    self.place_timer.start()
                    self.pinch_start_position = left_hand.index_tip_world
                    print("[GESTURE] PLACE: Left pinch only")
                    return None
        
        # ============ PRIORITY 6: SCATTER/RESTORE (left thumb gestures) ============
        if left_hand and not self.scatter_triggered:
            if self._is_thumb_down(left_hand):
                self._transition_to(GestureState.SCATTER_CHARGING)
                self.scatter_timer.start()
                print("[GESTURE] Scatter charging")
                return None
        
        if left_hand and not self.restore_triggered:
            if self._is_thumb_up(left_hand):
                self._transition_to(GestureState.RESTORE_CHARGING)
                self.restore_timer.start()
                print("[GESTURE] Restore charging")
                return None

        return None

    def _handle_placing_state(self, hand: Optional[Hand3D], current_time: float) -> Optional[GestureEvent]:
        """Handle the PLACING state - YouTube-style with continuous building.
        
        YouTube reference (index_advance.html lines 302-331):
        1. Initial hold (buildTimer < INTENT_HOLD): Show progress circle
        2. Once timer complete: Enter continuous build mode
        3. While pinching: Continuously add voxels as hand moves
        4. Axis locking: Lock to X or Y based on initial movement direction
        5. On palm open: Commit sketch voxels to permanent
        """
        if not hand:
            self._transition_to(GestureState.IDLE)
            self._reset_build_state()
            return None

        # Check if pinch released (palm open = commit)
        if not hand.is_pinching:
            # If we were in continuous building mode, commit the sketch
            if self.is_building and self.sketch_keys:
                # Return event with all sketch positions
                event = GestureEvent(
                    gesture_type=GestureState.CONTINUOUS_BUILDING,
                    position=hand.index_tip_world,
                    extra_data={
                        'sketch_positions': list(self.sketch_keys),
                        'commit': True
                    }
                )
                self._transition_to(GestureState.IDLE)
                self._reset_build_state()
                return event
            
            # Pinch released before timer - cancel
            self._cancel_gesture()
            self._reset_build_state()
            return None

        # Update progress during initial hold
        progress = self.place_timer.progress(self.cfg.PLACE_HOLD_TIME)
        if self.on_gesture_progress:
            self.on_gesture_progress(GestureState.PLACING, progress)

        # Check if hold time reached - enter continuous build mode
        if self.place_timer.elapsed() >= self.cfg.PLACE_HOLD_TIME:
            if not self.is_building:
                # First time entering build mode - initialize
                self.is_building = True
                self.build_start_pos = hand.index_tip_world
                self.active_axis = None
                self.sketch_keys.clear()
                self.last_grid_pos = None
            
            # Continuous building - add voxel at current position
            current_pos = hand.index_tip_world
            
            # YouTube-style axis locking (lines 311-317)
            if self.build_start_pos and not self.active_axis:
                dx = abs(current_pos[0] - self.build_start_pos[0])
                dy = abs(current_pos[1] - self.build_start_pos[1])
                
                # Lock to axis with larger delta (threshold 0.4 grid units)
                if dx > 0.4 or dy > 0.4:
                    if dx >= dy:
                        self.active_axis = 'x'
                    else:
                        self.active_axis = 'y'
            
            # Apply axis lock to get target position
            if self.build_start_pos:
                if self.active_axis == 'x':
                    # Only X changes, Y stays at start
                    target_pos = (current_pos[0], self.build_start_pos[1], 0.0)
                elif self.active_axis == 'y':
                    # Only Y changes, X stays at start
                    target_pos = (self.build_start_pos[0], current_pos[1], 0.0)
                else:
                    # No axis lock yet
                    target_pos = (current_pos[0], current_pos[1], 0.0)
            else:
                target_pos = current_pos
            
            # Return continuous building event with current position
            # The main app will handle adding to sketch and committing
            event = GestureEvent(
                gesture_type=GestureState.CONTINUOUS_BUILDING,
                position=target_pos,
                extra_data={
                    'sketch_positions': list(self.sketch_keys),
                    'commit': False,
                    'active_axis': self.active_axis
                }
            )
            return event

        return None
    
    def _reset_build_state(self):
        """Reset continuous building state."""
        self.is_building = False
        self.build_start_pos = None
        self.active_axis = None
        self.sketch_keys.clear()
        self.last_grid_pos = None

    def _handle_deleting_state(self, tracker, hand: Optional[Hand3D], current_time: float) -> Optional[GestureEvent]:
        """Handle the DELETING state.
        
        PHASE 13 FIX: Direct deletion system - no events during collecting.
        - RIGHT pinch + LEFT index pointing activates delete mode
        - Cursor follows LEFT index finger (updated every frame)
        - Blocks are collected DIRECTLY inside this handler (no events)
        - Only returns ONE event: batch_delete when right pinch is released
        
        Previous bug: Returning 'collecting' events triggered ACTION_COOLDOWN,
        which prevented pinch release detection for 0.3s windows.
        """
        # Get right hand (the one pinching)
        right_hand = tracker.get_right_hand() if tracker else hand
        left_hand = tracker.get_left_hand() if tracker else None
        
        if not right_hand:
            print("[DELETE-1] No right hand, cancelling")
            self._cancel_gesture()
            self._reset_delete_state()
            return None

        # Initialize delete tracking set if not exists
        if not hasattr(self, 'blocks_to_delete'):
            self.blocks_to_delete = set()
        if not hasattr(self, 'delete_mode_active'):
            self.delete_mode_active = False

        # Continuously update delete cursor to LEFT index position
        if left_hand:
            self.delete_cursor_position = left_hand.index_tip_world

        # CRITICAL: Check pinch release FIRST before anything else
        if not right_hand.is_pinching:
            print(f"[DELETE-3] Pinch RELEASED! delete_mode_active={self.delete_mode_active}, blocks={len(self.blocks_to_delete)}")
            if self.delete_mode_active and self.blocks_to_delete:
                # Return batch delete event with all collected positions
                positions_to_delete = list(self.blocks_to_delete)
                print(f"[DELETE-3] Generating batch_delete event with {len(positions_to_delete)} blocks: {positions_to_delete}")
                event = GestureEvent(
                    gesture_type=GestureState.DELETING,
                    position=self.delete_cursor_position or (0, 0, 0),
                    extra_data={
                        'batch_delete': True,
                        'positions': positions_to_delete
                    }
                )
                self._transition_to(GestureState.IDLE)
                self._reset_delete_state()
                return event
            else:
                # Cancelled before activation or no blocks collected
                print("[DELETE-3] Cancelled (no blocks or not active)")
                self._cancel_gesture()
                self._reset_delete_state()
                return None

        # Update progress during charging phase
        if not self.delete_mode_active:
            progress = self.delete_timer.progress(self.cfg.DELETE_HOLD_TIME)
            if self.on_gesture_progress:
                self.on_gesture_progress(GestureState.DELETING, progress)

            # Check if hold time reached - activate continuous delete mode
            if self.delete_timer.elapsed() >= self.cfg.DELETE_HOLD_TIME:
                self.delete_mode_active = True
                self.blocks_to_delete = set()  # Fresh set
                print("[DELETE-1] DELETE mode ACTIVE - paint over blocks to mark for deletion")

        # Continuous delete mode: Collect blocks DIRECTLY (no events)
        if self.delete_mode_active and self._voxel_engine and self.delete_cursor_position:
            # Convert cursor world position to grid position
            grid_pos = self._voxel_engine.world_to_grid(self.delete_cursor_position)
            # Check if a voxel exists at this grid position
            if grid_pos in self._voxel_engine.voxels:
                if grid_pos not in self.blocks_to_delete:
                    self.blocks_to_delete.add(grid_pos)
                    print(f"[DELETE-2] Block at {grid_pos} marked for deletion (total: {len(self.blocks_to_delete)})")

        # DO NOT return any event during collecting - this prevents cooldown from blocking pinch release detection
        return None
    
    def _reset_delete_state(self):
        """Reset continuous delete tracking state."""
        if hasattr(self, 'blocks_to_delete'):
            self.blocks_to_delete.clear()
        self.delete_mode_active = False
        self.delete_cursor_position = None
    
    def add_block_to_delete(self, grid_pos: tuple):
        """Add a grid position to the delete tracking set."""
        if not hasattr(self, 'blocks_to_delete'):
            self.blocks_to_delete = set()
        self.blocks_to_delete.add(grid_pos)

    def _handle_extending_state(self, hand: Optional[Hand3D], current_time: float) -> Optional[GestureEvent]:
        """Handle the EXTENDING state (pinch and drag)."""
        # This is now handled in _handle_placing_state
        return None

    def _handle_rotating_state(self, tracker: HandTracker3D, current_time: float) -> Optional[GestureEvent]:
        """Handle the ROTATING state (two-hand rotation).
        
        PHASE 11: Hologram-style rotation control.
        - Y rotation (horizontal spin): Based on X difference between hands
        - X rotation (tilt up/down): Based on Y difference between hands
        - Hands level = no rotation
        - Smooth, controllable feel like manipulating a hologram
        """
        if not tracker.is_two_hands_detected():
            self._transition_to(GestureState.IDLE)
            print("[GESTURE] ROTATE ended (lost hand)")
            return None

        left_hand = tracker.get_left_hand()
        right_hand = tracker.get_right_hand()
        
        # PHASE 12: Null check to prevent NoneType error
        if left_hand is None or right_hand is None:
            self._transition_to(GestureState.IDLE)
            print("[GESTURE] ROTATE ended (hand is None)")
            return None
        
        # Check if both hands still have open palms
        if sum(left_hand.fingers_up) < 4 or sum(right_hand.fingers_up) < 4:
            self._transition_to(GestureState.IDLE)
            print("[GESTURE] ROTATE ended (palms closed)")
            return None
        
        # Get hand positions (normalized 0-1 from MediaPipe)
        # Use wrist position (landmark 0) for more stable tracking
        left_x = left_hand.landmarks[0].x if left_hand.landmarks else 0.5
        right_x = right_hand.landmarks[0].x if right_hand.landmarks else 0.5
        left_y = left_hand.landmarks[0].y if left_hand.landmarks else 0.5
        right_y = right_hand.landmarks[0].y if right_hand.landmarks else 0.5
        
        # Calculate rotation based on hand positions
        # dx: horizontal spread (typically 0.3 to 0.7)
        # dy: vertical offset (typically -0.2 to 0.2)
        dx = right_x - left_x
        dy = right_y - left_y
        
        # PHASE 12: Rotation with dead zones, smoothing - JARVIS-like feel
        NEUTRAL_SPREAD = 0.4  # Neutral horizontal spread
        DEAD_ZONE = 0.03      # 3% dead zone for responsive feel
        SENSITIVITY = 0.025   # Medium speed for JARVIS-like control
        SMOOTH_FACTOR = 0.2   # Slightly more responsive smoothing
        MAX_SPEED = 0.04      # Higher cap for faster rotation when needed
        
        # Initialize smoothed values if not exists
        if not hasattr(self, 'smooth_rot_y'):
            self.smooth_rot_y = 0.0
        if not hasattr(self, 'smooth_rot_x'):
            self.smooth_rot_x = 0.0
        
        # Y-axis rotation (spin left/right) with dead zone
        if abs(dx - NEUTRAL_SPREAD) > DEAD_ZONE:
            target_rot_y = (dx - NEUTRAL_SPREAD) * SENSITIVITY
        else:
            target_rot_y = 0.0
        
        # X-axis rotation (tilt up/down) with dead zone
        if abs(dy) > DEAD_ZONE:
            target_rot_x = dy * SENSITIVITY
        else:
            target_rot_x = 0.0
        
        # Apply smoothing (exponential moving average)
        self.smooth_rot_y = self.smooth_rot_y * (1 - SMOOTH_FACTOR) + target_rot_y * SMOOTH_FACTOR
        self.smooth_rot_x = self.smooth_rot_x * (1 - SMOOTH_FACTOR) + target_rot_x * SMOOTH_FACTOR
        
        # Clamp to max speed
        rotation_y_delta = max(-MAX_SPEED, min(MAX_SPEED, self.smooth_rot_y))
        rotation_x_delta = max(-MAX_SPEED, min(MAX_SPEED, self.smooth_rot_x))
        
        # Report progress (visual feedback)
        if self.on_gesture_progress:
            self.on_gesture_progress(GestureState.ROTATING, 0.5)
        
        # Return continuous rotation event with both axes
        return GestureEvent(
            gesture_type=GestureState.ROTATING,
            rotation_angle=rotation_y_delta,
            extra_data={
                'continuous': True,
                'rotation_x': rotation_x_delta,
                'rotation_y': rotation_y_delta
            }
        )

    def _handle_color_menu_state(self, hand: Optional[Hand3D], current_time: float) -> Optional[GestureEvent]:
        """Handle the COLOR_MENU state."""
        if not hand:
            self._cancel_gesture()
            return None

        fingers = hand.fingers_up

        if not self.color_menu_active:
            # Waiting for menu to open
            if sum(fingers) >= 4:  # Still open palm
                progress = self.color_menu_timer.progress(self.cfg.COLOR_MENU_HOLD_TIME)
                if self.on_gesture_progress:
                    self.on_gesture_progress(GestureState.COLOR_MENU, progress)

                if self.color_menu_timer.elapsed() >= self.cfg.COLOR_MENU_HOLD_TIME:
                    self.color_menu_active = True
            else:
                self._cancel_gesture()
        else:
            # Menu is open - track index finger for selection
            if fingers[1] == 1:  # Index finger up
                # Calculate which color is selected based on finger position
                # This would use the radial menu position in the actual implementation
                index_pos = hand.landmarks[8]
                center = hand.center_world

                # Calculate angle from center to index tip
                dx = index_pos.world_x - center[0]
                dy = index_pos.world_y - center[1]
                angle = math.atan2(dy, dx)

                # Map angle to color index (8 colors)
                num_colors = len(CONFIG.colors.get_palette())
                self.selected_color_index = int((angle + math.pi) / (2 * math.pi) * num_colors) % num_colors

            # Check for fist (confirm selection)
            if sum(fingers) == 0:
                event = GestureEvent(
                    gesture_type=GestureState.COLOR_MENU,
                    selected_color_index=self.selected_color_index
                )
                self._transition_to(GestureState.IDLE)
                self.color_menu_active = False
                return event

        return None

    def _handle_recombining_state(self, hand: Optional[Hand3D], current_time: float) -> Optional[GestureEvent]:
        """Handle the RECOMBINING state."""
        if not hand:
            self._cancel_gesture()
            return None

        fingers = hand.fingers_up

        # Check if still in fist
        if sum(fingers) == 0:
            progress = self.recombine_timer.progress(self.cfg.RECOMBINE_HOLD_TIME)
            if self.on_gesture_progress:
                self.on_gesture_progress(GestureState.RECOMBINING, progress)

            if self.recombine_timer.elapsed() >= self.cfg.RECOMBINE_HOLD_TIME:
                event = GestureEvent(
                    gesture_type=GestureState.RECOMBINING,
                    position=hand.center_world
                )
                self._transition_to(GestureState.IDLE)
                return event
        else:
            self._cancel_gesture()

        return None

    def _handle_selecting_state(self, hand: Optional[Hand3D], current_time: float) -> Optional[GestureEvent]:
        """Handle the SELECTING state (box selection)."""
        if not hand:
            self._cancel_gesture()
            return None

        fingers = hand.fingers_up

        # Check if still in selection gesture
        if fingers[1] == 1 and fingers[2] == 1:  # Index and middle up
            self.selection_end = hand.index_tip_world

            if self.on_gesture_progress:
                self.on_gesture_progress(GestureState.SELECTING, 0.5)
        else:
            # Gesture ended - complete selection
            if self.selection_start and self.selection_end:
                # Calculate selection box bounds
                min_x = min(self.selection_start[0], self.selection_end[0])
                max_x = max(self.selection_start[0], self.selection_end[0])
                min_y = min(self.selection_start[1], self.selection_end[1])
                max_y = max(self.selection_start[1], self.selection_end[1])
                min_z = min(self.selection_start[2], self.selection_end[2])
                max_z = max(self.selection_start[2], self.selection_end[2])

                # Check if box is big enough
                size = max(max_x - min_x, max_y - min_y, max_z - min_z)
                if size >= self.cfg.MIN_SELECTION_SIZE / 10:  # Convert from pixels to world units approximately
                    event = GestureEvent(
                        gesture_type=GestureState.SELECTING,
                        selection_box=(min_x, min_y, min_z, max_x, max_y, max_z)
                    )
                    self._transition_to(GestureState.IDLE)
                    return event

            self._cancel_gesture()

        return None

    def _handle_swipe_state(self, hand: Optional[Hand3D], current_time: float) -> Optional[GestureEvent]:
        """Handle swipe states."""
        # Swipe is detected in idle state, this handles completion
        self._transition_to(GestureState.IDLE)
        return None

    # ============ NEW TWO-HAND GESTURE HANDLERS (Phase 2) ============

    def _handle_panning_state(self, tracker: HandTracker3D, current_time: float) -> Optional[GestureEvent]:
        """Handle the PANNING state (left palm + right fist drag)."""
        left_hand = tracker.get_left_hand()
        right_hand = tracker.get_right_hand()

        if not left_hand or not right_hand:
            self._cancel_gesture()
            return None

        left_fingers = left_hand.fingers_up
        right_fingers = right_hand.fingers_up

        # Check if gesture is still valid (left open palm + right fist)
        if sum(left_fingers) < 4 or sum(right_fingers) > 1:
            self._cancel_gesture()
            return None

        # Calculate pan delta from fist movement
        if self.pan_start_positions:
            _, start_right = self.pan_start_positions
            current_right = right_hand.center_world

            pan_dx = current_right[0] - start_right[0]
            pan_dy = current_right[1] - start_right[1]

            # Return continuous pan events
            event = GestureEvent(
                gesture_type=GestureState.PANNING,
                pan_delta=(pan_dx * 0.5, pan_dy * 0.5),  # Scale factor
                left_hand_pos=left_hand.center_world,
                right_hand_pos=right_hand.center_world
            )

            # Update start position for next frame (continuous pan)
            self.pan_start_positions = (left_hand.center_world, right_hand.center_world)

            if self.on_gesture_progress:
                self.on_gesture_progress(GestureState.PANNING, 0.5)

            return event

        return None

    def _handle_zooming_state(self, tracker: HandTracker3D, current_time: float) -> Optional[GestureEvent]:
        """Handle the ZOOMING state (both hands pinching, move apart/together)."""
        left_hand = tracker.get_left_hand()
        right_hand = tracker.get_right_hand()

        if not left_hand or not right_hand:
            self._cancel_gesture()
            return None

        # Check if both hands still pinching
        if not left_hand.is_pinching or not right_hand.is_pinching:
            self._cancel_gesture()
            return None

        # Calculate current distance between pinch points
        current_dist = math.sqrt(
            (left_hand.index_tip_world[0] - right_hand.index_tip_world[0])**2 +
            (left_hand.index_tip_world[1] - right_hand.index_tip_world[1])**2 +
            (left_hand.index_tip_world[2] - right_hand.index_tip_world[2])**2
        )

        # Calculate zoom delta
        if self.two_hand_start_distance > 0.1:
            zoom_delta = (current_dist - self.two_hand_start_distance) / self.two_hand_start_distance

            event = GestureEvent(
                gesture_type=GestureState.ZOOMING,
                zoom_delta=zoom_delta,
                left_hand_pos=left_hand.index_tip_world,
                right_hand_pos=right_hand.index_tip_world
            )

            # Update start distance for next frame (continuous zoom)
            self.two_hand_start_distance = current_dist

            if self.on_gesture_progress:
                self.on_gesture_progress(GestureState.ZOOMING, 0.5)

            return event

        return None

    def _handle_two_hand_placing_state(self, tracker: HandTracker3D, current_time: float) -> Optional[GestureEvent]:
        """Handle the TWO_HAND_PLACING state (more precise placement with two hands)."""
        right_hand = tracker.get_right_hand()

        if not right_hand:
            self._cancel_gesture()
            return None

        # Check if still pinching
        if not right_hand.is_pinching:
            # Check for drag/extend
            if self.pinch_start_position:
                current_pos = right_hand.index_tip_world
                distance = math.sqrt(
                    (current_pos[0] - self.pinch_start_position[0])**2 +
                    (current_pos[1] - self.pinch_start_position[1])**2 +
                    (current_pos[2] - self.pinch_start_position[2])**2
                )

                if distance > 2.0:  # Significant drag
                    direction = (
                        current_pos[0] - self.pinch_start_position[0],
                        current_pos[1] - self.pinch_start_position[1],
                        current_pos[2] - self.pinch_start_position[2]
                    )
                    event = GestureEvent(
                        gesture_type=GestureState.EXTENDING,
                        position=self.pinch_start_position,
                        direction=direction
                    )
                    self._transition_to(GestureState.IDLE)
                    return event

            self._cancel_gesture()
            return None

        # Update progress
        progress = self.place_timer.progress(self.cfg.PLACE_HOLD_TIME)
        if self.on_gesture_progress:
            self.on_gesture_progress(GestureState.TWO_HAND_PLACING, progress)

        # Check if hold time reached
        if self.place_timer.elapsed() >= self.cfg.PLACE_HOLD_TIME:
            left_hand = tracker.get_left_hand()
            event = GestureEvent(
                gesture_type=GestureState.PLACING,
                position=right_hand.index_tip_world,
                left_hand_pos=left_hand.center_world if left_hand else None,
                right_hand_pos=right_hand.index_tip_world
            )
            self._transition_to(GestureState.IDLE)
            return event

        return None

    def _handle_two_hand_deleting_state(self, tracker: HandTracker3D, current_time: float) -> Optional[GestureEvent]:
        """Handle the TWO_HAND_DELETING state (left pointing + right pinch)."""
        left_hand = tracker.get_left_hand()
        right_hand = tracker.get_right_hand()

        if not left_hand or not right_hand:
            self._cancel_gesture()
            return None

        left_fingers = left_hand.fingers_up
        # Allow some flexibility - left hand pointing (index up)
        if left_fingers[1] != 1:
            self._cancel_gesture()
            return None

        # Check if right hand still pinching
        if not right_hand.is_pinching:
            self._cancel_gesture()
            return None

        # Update progress
        progress = self.delete_timer.progress(self.cfg.DELETE_HOLD_TIME)
        if self.on_gesture_progress:
            self.on_gesture_progress(GestureState.TWO_HAND_DELETING, progress)

        # Check if hold time reached
        if self.delete_timer.elapsed() >= self.cfg.DELETE_HOLD_TIME:
            event = GestureEvent(
                gesture_type=GestureState.DELETING,
                position=left_hand.index_tip_world,  # Delete at left hand's pointing position
                left_hand_pos=left_hand.index_tip_world,
                right_hand_pos=right_hand.index_tip_world
            )
            self._transition_to(GestureState.IDLE)
            return event

        return None

    # ============ END TWO-HAND GESTURE HANDLERS ============

    # ============ PHASE 8: DEFINITIVE GESTURE HANDLERS ============
    
    def _handle_grabbing_state(self, tracker: HandTracker3D, current_time: float) -> Optional[GestureEvent]:
        """Handle the GRABBING state - right fist to drag structure.
        
        Hold 500ms to activate, then drag while fist is held.
        """
        right_hand = tracker.get_right_hand()
        
        if not right_hand:
            self._cancel_gesture()
            self.is_grabbing = False
            self.grab_timer.reset()
            return None
        
        # Check if still making fist
        if not self._is_fist(right_hand.fingers_up):
            self._cancel_gesture()
            self.is_grabbing = False
            self.grab_timer.reset()
            return None
        
        # Update progress
        progress = self.grab_timer.progress(self.cfg.GRAB_HOLD_TIME)
        if self.on_gesture_progress:
            self.on_gesture_progress(GestureState.GRABBING, progress)
        
        # Check if hold time reached
        if not self.is_grabbing and self.grab_timer.elapsed() >= self.cfg.GRAB_HOLD_TIME:
            self.is_grabbing = True
            print("[GESTURE] Grab ACTIVATED - drag to move structure")
        
        # While grabbing, continuously send grab events with hand position
        if self.is_grabbing:
            return GestureEvent(
                gesture_type=GestureState.GRABBING,
                position=right_hand.center_world
            )
        
        return None
    
    def _handle_scatter_charging_state(self, tracker: HandTracker3D, current_time: float) -> Optional[GestureEvent]:
        """Handle the SCATTER_CHARGING state - left thumb down to scatter.
        
        Hold 800ms to trigger scatter ONCE.
        """
        left_hand = tracker.get_left_hand()
        
        if not left_hand:
            self._cancel_gesture()
            self.scatter_timer.reset()
            return None
        
        # Check if still thumb down
        if not self._is_thumb_down(left_hand):
            self._cancel_gesture()
            self.scatter_timer.reset()
            self.scatter_triggered = False  # Allow re-trigger
            return None
        
        # Update progress
        progress = self.scatter_timer.progress(self.cfg.SCATTER_HOLD_TIME)
        if self.on_gesture_progress:
            self.on_gesture_progress(GestureState.SCATTER_CHARGING, progress)
        
        # Check if hold time reached
        if self.scatter_timer.elapsed() >= self.cfg.SCATTER_HOLD_TIME and not self.scatter_triggered:
            self.scatter_triggered = True
            print("[GESTURE] Scatter TRIGGERED!")
            event = GestureEvent(
                gesture_type=GestureState.SCATTERING,
                position=left_hand.center_world
            )
            # Stay in charging state but prevent re-trigger
            return event
        
        return None
    
    def _handle_restore_charging_state(self, tracker: HandTracker3D, current_time: float) -> Optional[GestureEvent]:
        """Handle the RESTORE_CHARGING state - left thumb up to restore.
        
        Hold 800ms to trigger restore ONCE.
        """
        left_hand = tracker.get_left_hand()
        
        if not left_hand:
            self._cancel_gesture()
            self.restore_timer.reset()
            return None
        
        # Check if still thumb up
        if not self._is_thumb_up(left_hand):
            self._cancel_gesture()
            self.restore_timer.reset()
            self.restore_triggered = False  # Allow re-trigger
            return None
        
        # Update progress
        progress = self.restore_timer.progress(self.cfg.RESTORE_HOLD_TIME)
        if self.on_gesture_progress:
            self.on_gesture_progress(GestureState.RESTORE_CHARGING, progress)
        
        # Check if hold time reached
        if self.restore_timer.elapsed() >= self.cfg.RESTORE_HOLD_TIME and not self.restore_triggered:
            self.restore_triggered = True
            print("[GESTURE] Restore TRIGGERED!")
            event = GestureEvent(
                gesture_type=GestureState.RECOMBINING,  # Use existing recombine event type
                position=left_hand.center_world
            )
            # Stay in charging state but prevent re-trigger
            return event
        
        return None
    
    def _handle_full_resetting_state(self, tracker: HandTracker3D, current_time: float) -> Optional[GestureEvent]:
        """Handle the FULL_RESETTING state - right thumb up to clear all voxels.

        Hold right thumb up for the configured duration to trigger full reset.
        The left hand must stay clear of active gestures so placement and delete gestures win cleanly.
        """
        left_hand = tracker.get_left_hand()
        right_hand = tracker.get_right_hand()

        if not right_hand:
            self._cancel_gesture()
            self.full_reset_timer.reset()
            self.full_reset_triggered = False
            return None

        if not self._is_full_reset_pose(left_hand, right_hand):
            self._cancel_gesture()
            self.full_reset_timer.reset()
            self.full_reset_triggered = False
            return None
        
        # Already triggered, stay in state until palm closes
        if self.full_reset_triggered:
            return None
        
        # Update progress
        elapsed = self.full_reset_timer.elapsed()
        progress = self.full_reset_timer.progress(self.cfg.FULL_RESET_HOLD_TIME)
        
        if self.on_gesture_progress:
            self.on_gesture_progress(GestureState.FULL_RESETTING, progress)
        
        # Print progress every ~1 second
        if int(elapsed) != int(elapsed - 0.017) or elapsed < 0.05:
            print(f"[RESET-2] Full reset timer: {elapsed:.1f}s / {self.cfg.FULL_RESET_HOLD_TIME:.1f}s ({progress*100:.0f}%)")
        
        # Check if hold time reached
        if elapsed >= self.cfg.FULL_RESET_HOLD_TIME:
            self.full_reset_triggered = True
            print("[RESET-3] FULL RESET COMPLETE - clearing all voxels")
            return GestureEvent(gesture_type=GestureState.FULL_RESETTING, position=(0, 0, 0))
        
        return None

    # ============ END PHASE 8 GESTURE HANDLERS ============

    def _check_scatter_gesture(self, hand: Optional[Hand3D], current_time: float) -> Optional[GestureEvent]:
        """Check for scatter gesture (fast hand spread)."""
        if not hand:
            return None

        # Check cooldown
        if current_time - self.last_scatter_time < self.cfg.SCATTER_COOLDOWN:
            return None

        # Check for rapid palm velocity with open hand
        palm_speed = math.sqrt(
            hand.palm_velocity[0]**2 +
            hand.palm_velocity[1]**2 +
            hand.palm_velocity[2]**2
        )

        # Convert world units to approximate pixel velocity for threshold comparison
        pixel_velocity = palm_speed * 50  # Rough conversion factor

        if pixel_velocity > self.cfg.SCATTER_VELOCITY_THRESHOLD:
            if sum(hand.fingers_up) >= 4:  # Open hand
                self.last_scatter_time = current_time
                event = GestureEvent(
                    gesture_type=GestureState.SCATTERING,
                    position=hand.center_world
                )
                self._transition_to(GestureState.IDLE)
                return event

        return None

    def _check_swipe_gesture(self, hand: Hand3D, current_time: float) -> Optional[GestureEvent]:
        """Check for swipe gestures."""
        if not self.swipe_start_position:
            return None

        if current_time - self.swipe_start_time > self.cfg.SWIPE_MAX_TIME:
            self.swipe_start_position = None
            return None

        current_pos = (hand.landmarks[8].x, hand.landmarks[8].y)
        dx = current_pos[0] - self.swipe_start_position[0]
        dy = current_pos[1] - self.swipe_start_position[1]
        distance = math.sqrt(dx**2 + dy**2)

        if distance > self.cfg.SWIPE_MIN_DISTANCE:
            self.swipe_start_position = None

            # Determine swipe direction
            if abs(dx) > abs(dy):  # Horizontal swipe
                if dx < 0:  # Swipe left (undo)
                    return GestureEvent(gesture_type=GestureState.SWIPING_LEFT)
                else:  # Swipe right (redo)
                    return GestureEvent(gesture_type=GestureState.SWIPING_RIGHT)

        return None

    def _transition_to(self, new_state: GestureState):
        """Transition to a new state."""
        if new_state != self.state:
            self.previous_state = self.state
            self.state = new_state

            if self.on_gesture_start and new_state != GestureState.IDLE:
                self.on_gesture_start(new_state)

    def _cancel_gesture(self):
        """Cancel the current gesture and return to IDLE."""
        if self.on_gesture_cancel and self.state != GestureState.IDLE:
            self.on_gesture_cancel(self.state)

        self._reset_timers()
        self.state = GestureState.IDLE
        self.color_menu_active = False

    def _reset_timers(self):
        """Reset all gesture timers."""
        self.place_timer.reset()
        self.delete_timer.reset()
        self.recombine_timer.reset()
        self.color_menu_timer.reset()
        # Phase 8: New timers
        self.grab_timer.reset()
        self.rotate_timer.reset()
        self.scatter_timer.reset()
        self.restore_timer.reset()
        self.full_reset_timer.reset()
        # Reset one-shot flags
        self.scatter_triggered = False
        self.restore_triggered = False
        self.full_reset_triggered = False
        self.is_grabbing = False

    def get_state(self) -> GestureState:
        """Get current gesture state."""
        return self.state

    def get_progress(self) -> float:
        """Get progress of current timed gesture (0-1)."""
        if self.state == GestureState.PLACING:
            return self.place_timer.progress(self.cfg.PLACE_HOLD_TIME)
        elif self.state == GestureState.DELETING:
            return self.delete_timer.progress(self.cfg.DELETE_HOLD_TIME)
        elif self.state == GestureState.RECOMBINING:
            return self.recombine_timer.progress(self.cfg.RECOMBINE_HOLD_TIME)
        elif self.state == GestureState.COLOR_MENU:
            return self.color_menu_timer.progress(self.cfg.COLOR_MENU_HOLD_TIME)
        # Phase 8: New gesture progress
        elif self.state == GestureState.GRABBING:
            return self.grab_timer.progress(self.cfg.GRAB_HOLD_TIME)
        elif self.state == GestureState.SCATTER_CHARGING:
            return self.scatter_timer.progress(self.cfg.SCATTER_HOLD_TIME)
        elif self.state == GestureState.RESTORE_CHARGING:
            return self.restore_timer.progress(self.cfg.RESTORE_HOLD_TIME)
        elif self.state == GestureState.FULL_RESETTING:
            return self.full_reset_timer.progress(self.cfg.FULL_RESET_HOLD_TIME)
        return 0.0

    def is_color_menu_open(self) -> bool:
        """Check if color menu is currently open."""
        return self.color_menu_active

    def get_selection_box(self) -> Optional[Tuple[Tuple[float, float, float], Tuple[float, float, float]]]:
        """Get current selection box corners if selecting."""
        if self.state == GestureState.SELECTING and self.selection_start and self.selection_end:
            return (self.selection_start, self.selection_end)
        return None


if __name__ == "__main__":
    # Simple test
    print("Gesture Recognition Module")
    print("=" * 50)
    print("Available gestures:")
    print("  - Pinch (index + thumb): Place block")
    print("  - Peace sign -> Fist: Delete block")
    print("  - Pinch + drag: Extend/extrude")
    print("  - Two hands rotate: Rotate selection")
    print("  - Open palm hold: Color menu")
    print("  - Fist hold: Recombine scattered blocks")
    print("  - Fast hand spread: Scatter blocks")
    print("  - Swipe left/right: Undo/Redo")
