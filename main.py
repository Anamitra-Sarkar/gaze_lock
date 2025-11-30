"""
GAZE-LOCK: Real-Time Eye Tracking & Control System
===================================================
Control the mouse cursor using only Eye Gaze (Iris Tracking) and Blinks.

This application uses MediaPipe Face Mesh for iris detection and maps
eye gaze to screen coordinates for cursor control. Blinks are detected
using Eye Aspect Ratio (EAR) for click functionality.

Usage:
    python main.py [--debug] [--no-cursor]

Controls:
    'q' - Quit the application (safety kill switch)
    'c' - Toggle cursor control
    'd' - Toggle debug mode
    'r' - Reset smoothing filters
    'SPACE' - Manual calibration mode

Author: GAZE-LOCK Development Team
License: MIT
Python: 3.10+
"""

import cv2
import numpy as np
import mediapipe as mp
import pyautogui
import argparse
import time
import sys
from typing import Optional, Tuple

from gaze_math import (
    GazeRatioCalculator,
    EyeAspectRatioCalculator,
    KalmanFilter2D,
    MovingAverageFilter,
    ScreenMapper,
    GazeVectorCalculator,
    get_landmark_point,
    get_eye_landmarks_for_ear,
    LEFT_EYE_OUTER_CORNER,
    LEFT_EYE_INNER_CORNER,
    LEFT_EYE_TOP,
    LEFT_EYE_BOTTOM,
    RIGHT_EYE_OUTER_CORNER,
    RIGHT_EYE_INNER_CORNER,
    RIGHT_EYE_TOP,
    RIGHT_EYE_BOTTOM,
    LEFT_IRIS_CENTER,
    RIGHT_IRIS_CENTER,
    LEFT_EYE_EAR_LANDMARKS,
    RIGHT_EYE_EAR_LANDMARKS,
)


# ============================================================================
# Configuration Constants
# ============================================================================

# Application settings
APP_NAME = "GAZE-LOCK"
WINDOW_NAME = f"{APP_NAME} - Eye Tracking HUD"

# Camera settings
CAMERA_INDEX = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30

# Smoothing settings
SMOOTHING_WINDOW = 5  # For moving average
KALMAN_PROCESS_NOISE = 0.03
KALMAN_MEASUREMENT_NOISE = 0.1

# Blink detection settings
BLINK_EAR_THRESHOLD = 0.21
BLINK_CONSECUTIVE_FRAMES = 3

# Cursor settings
CURSOR_SENSITIVITY = 1.5
CURSOR_DEAD_ZONE = 0.1

# HUD colors (BGR format)
COLOR_IRIS = (0, 255, 0)  # Green
COLOR_GAZE_VECTOR = (255, 0, 255)  # Magenta
COLOR_EYE_LANDMARKS = (0, 255, 255)  # Yellow
COLOR_TEXT = (255, 255, 255)  # White
COLOR_TEXT_BG = (0, 0, 0)  # Black
COLOR_WARNING = (0, 0, 255)  # Red
COLOR_SUCCESS = (0, 255, 0)  # Green

# HUD settings
IRIS_CIRCLE_RADIUS = 3
GAZE_VECTOR_LENGTH = 50
HUD_FONT = cv2.FONT_HERSHEY_SIMPLEX
HUD_FONT_SCALE = 0.5
HUD_FONT_THICKNESS = 1


# ============================================================================
# GazeLock Application Class
# ============================================================================

class GazeLockApp:
    """
    Main application class for GAZE-LOCK eye tracking system.
    """
    
    def __init__(
        self,
        debug_mode: bool = False,
        cursor_enabled: bool = True,
        use_kalman: bool = True
    ):
        """
        Initialize the GAZE-LOCK application.
        
        Args:
            debug_mode: Enable verbose debug output
            cursor_enabled: Enable cursor control
            use_kalman: Use Kalman filter (True) or Moving Average (False)
        """
        self.debug_mode = debug_mode
        self.cursor_enabled = cursor_enabled
        self.use_kalman = use_kalman
        self.running = False
        
        # Initialize MediaPipe Face Mesh with iris refinement
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,  # Required for iris landmarks
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Initialize gaze processing components
        self.gaze_calculator = GazeRatioCalculator()
        self.ear_calculator = EyeAspectRatioCalculator(
            blink_threshold=BLINK_EAR_THRESHOLD,
            blink_frames=BLINK_CONSECUTIVE_FRAMES
        )
        
        # Initialize smoothing filter
        if use_kalman:
            self.smoother = KalmanFilter2D(
                process_noise=KALMAN_PROCESS_NOISE,
                measurement_noise=KALMAN_MEASUREMENT_NOISE
            )
        else:
            self.smoother = MovingAverageFilter(window_size=SMOOTHING_WINDOW)
        
        # Get screen dimensions
        self.screen_width, self.screen_height = pyautogui.size()
        
        # Initialize screen mapper
        self.screen_mapper = ScreenMapper(
            screen_width=self.screen_width,
            screen_height=self.screen_height,
            sensitivity=CURSOR_SENSITIVITY,
            dead_zone=CURSOR_DEAD_ZONE
        )
        
        # Initialize gaze vector calculator
        self.gaze_vector = GazeVectorCalculator(vector_length=GAZE_VECTOR_LENGTH)
        
        # State variables
        self.current_ear = 0.0
        self.last_blink_time = 0.0
        self.blink_cooldown = 0.3  # Seconds between clicks
        self.face_detected = False
        self.frame_count = 0
        self.fps = 0.0
        self.fps_start_time = time.time()
        
        # PyAutoGUI safety settings
        pyautogui.FAILSAFE = True  # Move mouse to corner to abort
        pyautogui.PAUSE = 0.01  # Small delay between actions
        
        print(f"[{APP_NAME}] Initialized")
        print(f"[{APP_NAME}] Screen: {self.screen_width}x{self.screen_height}")
        print(f"[{APP_NAME}] Smoothing: {'Kalman Filter' if use_kalman else 'Moving Average'}")
        print(f"[{APP_NAME}] Cursor Control: {'Enabled' if cursor_enabled else 'Disabled'}")
    
    def run(self):
        """
        Main application loop.
        """
        # Initialize camera
        cap = cv2.VideoCapture(CAMERA_INDEX)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
        
        if not cap.isOpened():
            print(f"[{APP_NAME}] ERROR: Could not open camera")
            return
        
        print(f"[{APP_NAME}] Camera opened successfully")
        print(f"[{APP_NAME}] Press 'q' to quit (safety kill switch)")
        
        self.running = True
        
        try:
            while self.running:
                # Read frame
                ret, frame = cap.read()
                if not ret:
                    print(f"[{APP_NAME}] ERROR: Failed to read frame")
                    break
                
                # Mirror frame for natural interaction
                frame = cv2.flip(frame, 1)
                
                # Process frame
                frame = self._process_frame(frame)
                
                # Update FPS
                self._update_fps()
                
                # Draw HUD
                frame = self._draw_hud(frame)
                
                # Display frame
                cv2.imshow(WINDOW_NAME, frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if not self._handle_key(key):
                    break
                
                self.frame_count += 1
        
        except KeyboardInterrupt:
            print(f"\n[{APP_NAME}] Interrupted by user")
        
        finally:
            # Cleanup
            cap.release()
            cv2.destroyAllWindows()
            self.face_mesh.close()
            print(f"[{APP_NAME}] Shutdown complete")
    
    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Process a single frame for gaze detection.
        
        Args:
            frame: Input BGR frame
            
        Returns:
            Processed frame with visualizations
        """
        frame_height, frame_width = frame.shape[:2]
        
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process with Face Mesh
        results = self.face_mesh.process(rgb_frame)
        
        if not results.multi_face_landmarks:
            self.face_detected = False
            return frame
        
        self.face_detected = True
        landmarks = results.multi_face_landmarks[0]
        
        # Extract iris positions
        left_iris = get_landmark_point(
            landmarks, LEFT_IRIS_CENTER, frame_width, frame_height
        )
        right_iris = get_landmark_point(
            landmarks, RIGHT_IRIS_CENTER, frame_width, frame_height
        )
        
        # Extract eye corner landmarks for left eye
        left_inner = get_landmark_point(
            landmarks, LEFT_EYE_INNER_CORNER, frame_width, frame_height
        )
        left_outer = get_landmark_point(
            landmarks, LEFT_EYE_OUTER_CORNER, frame_width, frame_height
        )
        left_top = get_landmark_point(
            landmarks, LEFT_EYE_TOP, frame_width, frame_height
        )
        left_bottom = get_landmark_point(
            landmarks, LEFT_EYE_BOTTOM, frame_width, frame_height
        )
        
        # Extract eye corner landmarks for right eye
        right_inner = get_landmark_point(
            landmarks, RIGHT_EYE_INNER_CORNER, frame_width, frame_height
        )
        right_outer = get_landmark_point(
            landmarks, RIGHT_EYE_OUTER_CORNER, frame_width, frame_height
        )
        right_top = get_landmark_point(
            landmarks, RIGHT_EYE_TOP, frame_width, frame_height
        )
        right_bottom = get_landmark_point(
            landmarks, RIGHT_EYE_BOTTOM, frame_width, frame_height
        )
        
        # Calculate gaze ratios for both eyes
        left_h, left_v = self.gaze_calculator.get_gaze_ratios(
            left_iris, left_inner, left_outer, left_top, left_bottom
        )
        right_h, right_v = self.gaze_calculator.get_gaze_ratios(
            right_iris, right_inner, right_outer, right_top, right_bottom
        )
        
        # Average the ratios from both eyes
        avg_h = (left_h + right_h) / 2
        avg_v = (left_v + right_v) / 2
        
        # Apply smoothing
        smooth_h, smooth_v = self.smoother.smooth(avg_h, avg_v)
        
        # Calculate EAR for blink detection
        left_eye_ear_points = get_eye_landmarks_for_ear(
            landmarks, LEFT_EYE_EAR_LANDMARKS, frame_width, frame_height
        )
        right_eye_ear_points = get_eye_landmarks_for_ear(
            landmarks, RIGHT_EYE_EAR_LANDMARKS, frame_width, frame_height
        )
        self.current_ear = self.ear_calculator.calculate_average_ear(
            left_eye_ear_points, right_eye_ear_points
        )
        
        # Detect blink for click
        if self.ear_calculator.detect_blink(self.current_ear):
            current_time = time.time()
            if current_time - self.last_blink_time > self.blink_cooldown:
                if self.cursor_enabled:
                    pyautogui.click()
                    if self.debug_mode:
                        print(f"[{APP_NAME}] CLICK!")
                self.last_blink_time = current_time
        
        # Move cursor
        if self.cursor_enabled:
            screen_x, screen_y = self.screen_mapper.map_to_screen(smooth_h, smooth_v)
            pyautogui.moveTo(screen_x, screen_y)
        
        # Draw visualizations
        frame = self._draw_iris_circles(frame, left_iris, right_iris)
        frame = self._draw_gaze_vectors(
            frame, left_iris, right_iris, smooth_h, smooth_v,
            frame_width, frame_height
        )
        
        if self.debug_mode:
            frame = self._draw_eye_landmarks(
                frame, left_inner, left_outer, left_top, left_bottom
            )
            frame = self._draw_eye_landmarks(
                frame, right_inner, right_outer, right_top, right_bottom
            )
        
        return frame
    
    def _draw_iris_circles(
        self,
        frame: np.ndarray,
        left_iris: Tuple[float, float],
        right_iris: Tuple[float, float]
    ) -> np.ndarray:
        """Draw green circles around detected irises."""
        cv2.circle(
            frame,
            (int(left_iris[0]), int(left_iris[1])),
            IRIS_CIRCLE_RADIUS,
            COLOR_IRIS,
            2
        )
        cv2.circle(
            frame,
            (int(right_iris[0]), int(right_iris[1])),
            IRIS_CIRCLE_RADIUS,
            COLOR_IRIS,
            2
        )
        return frame
    
    def _draw_gaze_vectors(
        self,
        frame: np.ndarray,
        left_iris: Tuple[float, float],
        right_iris: Tuple[float, float],
        h_ratio: float,
        v_ratio: float,
        frame_width: int,
        frame_height: int
    ) -> np.ndarray:
        """Draw gaze direction vectors from each iris."""
        # Left eye gaze vector
        left_start, left_end = self.gaze_vector.calculate_gaze_vector(
            left_iris, h_ratio, v_ratio, frame_width, frame_height
        )
        cv2.arrowedLine(
            frame, left_start, left_end, COLOR_GAZE_VECTOR, 2, tipLength=0.3
        )
        
        # Right eye gaze vector
        right_start, right_end = self.gaze_vector.calculate_gaze_vector(
            right_iris, h_ratio, v_ratio, frame_width, frame_height
        )
        cv2.arrowedLine(
            frame, right_start, right_end, COLOR_GAZE_VECTOR, 2, tipLength=0.3
        )
        
        return frame
    
    def _draw_eye_landmarks(
        self,
        frame: np.ndarray,
        inner: Tuple[float, float],
        outer: Tuple[float, float],
        top: Tuple[float, float],
        bottom: Tuple[float, float]
    ) -> np.ndarray:
        """Draw eye corner landmarks (debug mode)."""
        for point in [inner, outer, top, bottom]:
            cv2.circle(
                frame,
                (int(point[0]), int(point[1])),
                2,
                COLOR_EYE_LANDMARKS,
                -1
            )
        return frame
    
    def _draw_hud(self, frame: np.ndarray) -> np.ndarray:
        """Draw heads-up display with status information."""
        frame_height, frame_width = frame.shape[:2]
        
        # Background for HUD
        overlay = frame.copy()
        cv2.rectangle(overlay, (5, 5), (200, 120), COLOR_TEXT_BG, -1)
        frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)
        
        # Title
        cv2.putText(
            frame, APP_NAME, (10, 25),
            HUD_FONT, 0.7, COLOR_SUCCESS, 2
        )
        
        # EAR status
        ear_color = COLOR_WARNING if self.current_ear < BLINK_EAR_THRESHOLD else COLOR_TEXT
        cv2.putText(
            frame, f"EAR: {self.current_ear:.2f}", (10, 50),
            HUD_FONT, HUD_FONT_SCALE, ear_color, HUD_FONT_THICKNESS
        )
        
        # Face detection status
        face_status = "Face: OK" if self.face_detected else "Face: NOT FOUND"
        face_color = COLOR_SUCCESS if self.face_detected else COLOR_WARNING
        cv2.putText(
            frame, face_status, (10, 70),
            HUD_FONT, HUD_FONT_SCALE, face_color, HUD_FONT_THICKNESS
        )
        
        # Cursor status
        cursor_status = "Cursor: ON" if self.cursor_enabled else "Cursor: OFF"
        cursor_color = COLOR_SUCCESS if self.cursor_enabled else COLOR_TEXT
        cv2.putText(
            frame, cursor_status, (10, 90),
            HUD_FONT, HUD_FONT_SCALE, cursor_color, HUD_FONT_THICKNESS
        )
        
        # FPS
        cv2.putText(
            frame, f"FPS: {self.fps:.1f}", (10, 110),
            HUD_FONT, HUD_FONT_SCALE, COLOR_TEXT, HUD_FONT_THICKNESS
        )
        
        # Help text at bottom
        help_text = "Press 'q' to quit | 'c' toggle cursor | 'd' debug"
        cv2.putText(
            frame, help_text, (10, frame_height - 10),
            HUD_FONT, 0.4, COLOR_TEXT, 1
        )
        
        return frame
    
    def _update_fps(self):
        """Update FPS counter."""
        if self.frame_count % 30 == 0:
            current_time = time.time()
            elapsed = current_time - self.fps_start_time
            if elapsed > 0:
                self.fps = 30 / elapsed
            self.fps_start_time = current_time
    
    def _handle_key(self, key: int) -> bool:
        """
        Handle keyboard input.
        
        Args:
            key: Key code from cv2.waitKey()
            
        Returns:
            False if application should quit, True otherwise
        """
        if key == ord('q') or key == 27:  # 'q' or ESC
            print(f"[{APP_NAME}] Quit signal received")
            self.running = False
            return False
        
        elif key == ord('c'):
            self.cursor_enabled = not self.cursor_enabled
            print(f"[{APP_NAME}] Cursor control: {'ON' if self.cursor_enabled else 'OFF'}")
        
        elif key == ord('d'):
            self.debug_mode = not self.debug_mode
            print(f"[{APP_NAME}] Debug mode: {'ON' if self.debug_mode else 'OFF'}")
        
        elif key == ord('r'):
            self.smoother.reset()
            print(f"[{APP_NAME}] Smoothing filter reset")
        
        return True


# ============================================================================
# Entry Point
# ============================================================================

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description=f"{APP_NAME}: Real-Time Eye Tracking & Control System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Controls:
    'q' or ESC  - Quit the application (safety kill switch)
    'c'         - Toggle cursor control on/off
    'd'         - Toggle debug mode on/off
    'r'         - Reset smoothing filters

Safety:
    Move mouse to screen corner to trigger PyAutoGUI failsafe.
"""
    )
    
    parser.add_argument(
        '--debug', '-d',
        action='store_true',
        help='Enable debug mode with extra visualizations'
    )
    
    parser.add_argument(
        '--no-cursor', '-n',
        action='store_true',
        help='Disable cursor control (visualization only)'
    )
    
    parser.add_argument(
        '--moving-average', '-m',
        action='store_true',
        help='Use Moving Average instead of Kalman Filter for smoothing'
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_arguments()
    
    print("=" * 60)
    print(f"  {APP_NAME}: Real-Time Eye Tracking & Control System")
    print("=" * 60)
    print()
    
    try:
        app = GazeLockApp(
            debug_mode=args.debug,
            cursor_enabled=not args.no_cursor,
            use_kalman=not args.moving_average
        )
        app.run()
    
    except Exception as e:
        print(f"[{APP_NAME}] FATAL ERROR: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
