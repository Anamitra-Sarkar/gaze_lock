"""
GAZE-LOCK: Gaze Math Module
============================
This module contains all mathematical utilities for:
- Gaze ratio calculation (iris position relative to eye corners)
- Eye Aspect Ratio (EAR) for blink detection
- Coordinate smoothing (Moving Average and Kalman Filter)
- Screen coordinate mapping

Author: GAZE-LOCK Development Team
License: MIT
"""

import numpy as np
from collections import deque
from typing import Tuple, Optional, List
import cv2


# MediaPipe Face Mesh Landmark Indices
# Left Eye landmarks
LEFT_EYE_OUTER_CORNER = 33
LEFT_EYE_INNER_CORNER = 133
LEFT_EYE_TOP = 159
LEFT_EYE_BOTTOM = 145

# Right Eye landmarks
RIGHT_EYE_OUTER_CORNER = 362
RIGHT_EYE_INNER_CORNER = 263
RIGHT_EYE_TOP = 386
RIGHT_EYE_BOTTOM = 374

# Left Iris landmarks (center)
# Index 468-472 are left iris landmarks, 468 is the center
# Source: MediaPipe Face Mesh with refine_landmarks=True
# Reference: https://google.github.io/mediapipe/solutions/face_mesh.html#face-landmark-model
LEFT_IRIS_CENTER = 468

# Right Iris landmarks (center)
# Index 473-477 are right iris landmarks, 473 is the center
# Source: MediaPipe Face Mesh with refine_landmarks=True
RIGHT_IRIS_CENTER = 473

# Eye landmarks for EAR calculation (6 points per eye)
LEFT_EYE_EAR_LANDMARKS = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_EAR_LANDMARKS = [362, 385, 387, 263, 373, 380]


class GazeRatioCalculator:
    """
    Calculates the gaze ratio (horizontal and vertical) based on
    iris position relative to eye corners.
    """
    
    def __init__(self):
        """Initialize the gaze ratio calculator."""
        pass
    
    def calculate_horizontal_ratio(
        self,
        iris_x: float,
        inner_corner_x: float,
        outer_corner_x: float
    ) -> float:
        """
        Calculate horizontal gaze ratio.
        
        The ratio represents where the iris is positioned between
        the inner and outer eye corners (0.0 = outer, 1.0 = inner).
        
        Args:
            iris_x: X coordinate of iris center
            inner_corner_x: X coordinate of inner eye corner
            outer_corner_x: X coordinate of outer eye corner
            
        Returns:
            Horizontal ratio between 0.0 and 1.0
        """
        eye_width = abs(inner_corner_x - outer_corner_x)
        if eye_width < 1e-6:
            return 0.5
        
        # Calculate relative position
        ratio = (iris_x - outer_corner_x) / (inner_corner_x - outer_corner_x)
        return np.clip(ratio, 0.0, 1.0)
    
    def calculate_vertical_ratio(
        self,
        iris_y: float,
        top_y: float,
        bottom_y: float
    ) -> float:
        """
        Calculate vertical gaze ratio.
        
        The ratio represents where the iris is positioned between
        the top and bottom of the eye (0.0 = top, 1.0 = bottom).
        
        Args:
            iris_y: Y coordinate of iris center
            top_y: Y coordinate of top eyelid
            bottom_y: Y coordinate of bottom eyelid
            
        Returns:
            Vertical ratio between 0.0 and 1.0
        """
        eye_height = abs(bottom_y - top_y)
        if eye_height < 1e-6:
            return 0.5
        
        ratio = (iris_y - top_y) / (bottom_y - top_y)
        return np.clip(ratio, 0.0, 1.0)
    
    def get_gaze_ratios(
        self,
        iris_center: Tuple[float, float],
        inner_corner: Tuple[float, float],
        outer_corner: Tuple[float, float],
        top: Tuple[float, float],
        bottom: Tuple[float, float]
    ) -> Tuple[float, float]:
        """
        Calculate both horizontal and vertical gaze ratios.
        
        Args:
            iris_center: (x, y) coordinates of iris center
            inner_corner: (x, y) of inner eye corner
            outer_corner: (x, y) of outer eye corner
            top: (x, y) of top eyelid point
            bottom: (x, y) of bottom eyelid point
            
        Returns:
            Tuple of (horizontal_ratio, vertical_ratio)
        """
        h_ratio = self.calculate_horizontal_ratio(
            iris_center[0], inner_corner[0], outer_corner[0]
        )
        v_ratio = self.calculate_vertical_ratio(
            iris_center[1], top[1], bottom[1]
        )
        return h_ratio, v_ratio


class EyeAspectRatioCalculator:
    """
    Calculates Eye Aspect Ratio (EAR) for blink detection.
    
    EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)
    
    Where p1-p6 are the 6 eye landmarks:
    p1 = outer corner, p2 = upper outer, p3 = upper inner,
    p4 = inner corner, p5 = lower inner, p6 = lower outer
    """
    
    def __init__(self, blink_threshold: float = 0.21, blink_frames: int = 3):
        """
        Initialize EAR calculator.
        
        Args:
            blink_threshold: EAR value below which a blink is detected
            blink_frames: Consecutive frames below threshold to register blink
        """
        self.blink_threshold = blink_threshold
        self.blink_frames = blink_frames
        self.consecutive_frames = 0
        self.blink_detected = False
    
    def calculate_ear(self, eye_landmarks: List[Tuple[float, float]]) -> float:
        """
        Calculate Eye Aspect Ratio for a single eye.
        
        Args:
            eye_landmarks: List of 6 (x, y) coordinates for eye landmarks
            
        Returns:
            Eye Aspect Ratio value
        """
        if len(eye_landmarks) != 6:
            return 0.3  # Default open eye value
        
        p1, p2, p3, p4, p5, p6 = eye_landmarks
        
        # Calculate vertical distances
        vertical_1 = self._euclidean_distance(p2, p6)
        vertical_2 = self._euclidean_distance(p3, p5)
        
        # Calculate horizontal distance
        horizontal = self._euclidean_distance(p1, p4)
        
        if horizontal < 1e-6:
            return 0.3
        
        ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
        return ear
    
    def calculate_average_ear(
        self,
        left_eye_landmarks: List[Tuple[float, float]],
        right_eye_landmarks: List[Tuple[float, float]]
    ) -> float:
        """
        Calculate average EAR for both eyes.
        
        Args:
            left_eye_landmarks: 6 landmarks for left eye
            right_eye_landmarks: 6 landmarks for right eye
            
        Returns:
            Average EAR value
        """
        left_ear = self.calculate_ear(left_eye_landmarks)
        right_ear = self.calculate_ear(right_eye_landmarks)
        return (left_ear + right_ear) / 2.0
    
    def detect_blink(self, ear: float) -> bool:
        """
        Detect if a blink occurred based on EAR value.
        
        Args:
            ear: Current Eye Aspect Ratio value
            
        Returns:
            True if blink was just completed, False otherwise
        """
        if ear < self.blink_threshold:
            self.consecutive_frames += 1
            self.blink_detected = False
        else:
            if self.consecutive_frames >= self.blink_frames:
                self.blink_detected = True
            else:
                self.blink_detected = False
            self.consecutive_frames = 0
        
        return self.blink_detected
    
    @staticmethod
    def _euclidean_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        """Calculate Euclidean distance between two points."""
        return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


class MovingAverageFilter:
    """
    Smooths coordinates using a Moving Average filter.
    Reduces cursor jitter/shaking.
    """
    
    def __init__(self, window_size: int = 5):
        """
        Initialize moving average filter.
        
        Args:
            window_size: Number of samples to average
        """
        self.window_size = window_size
        self.x_buffer = deque(maxlen=window_size)
        self.y_buffer = deque(maxlen=window_size)
    
    def smooth(self, x: float, y: float) -> Tuple[float, float]:
        """
        Apply moving average smoothing to coordinates.
        
        Args:
            x: Raw X coordinate
            y: Raw Y coordinate
            
        Returns:
            Smoothed (x, y) coordinates
        """
        self.x_buffer.append(x)
        self.y_buffer.append(y)
        
        smooth_x = np.mean(self.x_buffer)
        smooth_y = np.mean(self.y_buffer)
        
        return smooth_x, smooth_y
    
    def reset(self):
        """Clear the filter buffers."""
        self.x_buffer.clear()
        self.y_buffer.clear()


class KalmanFilter2D:
    """
    2D Kalman Filter for smooth cursor movement.
    Provides better smoothing than moving average with less lag.
    """
    
    def __init__(
        self,
        process_noise: float = 0.03,
        measurement_noise: float = 0.1
    ):
        """
        Initialize 2D Kalman Filter.
        
        Args:
            process_noise: Process noise covariance (Q)
            measurement_noise: Measurement noise covariance (R)
        """
        # State: [x, y, dx, dy] (position and velocity)
        self.kalman = cv2.KalmanFilter(4, 2)
        
        # Transition matrix (constant velocity model)
        self.kalman.transitionMatrix = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)
        
        # Measurement matrix (we only measure position)
        self.kalman.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=np.float32)
        
        # Process noise covariance
        self.kalman.processNoiseCov = np.eye(4, dtype=np.float32) * process_noise
        
        # Measurement noise covariance
        self.kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * measurement_noise
        
        # Error covariance
        self.kalman.errorCovPost = np.eye(4, dtype=np.float32)
        
        self.initialized = False
    
    def smooth(self, x: float, y: float) -> Tuple[float, float]:
        """
        Apply Kalman filter smoothing to coordinates.
        
        Args:
            x: Raw X coordinate
            y: Raw Y coordinate
            
        Returns:
            Smoothed (x, y) coordinates
        """
        measurement = np.array([[x], [y]], dtype=np.float32)
        
        if not self.initialized:
            # Initialize state with first measurement
            self.kalman.statePost = np.array([
                [x], [y], [0], [0]
            ], dtype=np.float32)
            self.initialized = True
            return x, y
        
        # Predict
        self.kalman.predict()
        
        # Correct with measurement
        corrected = self.kalman.correct(measurement)
        
        return float(corrected[0]), float(corrected[1])
    
    def reset(self):
        """Reset the Kalman filter state."""
        self.initialized = False


class ScreenMapper:
    """
    Maps gaze ratios to screen coordinates.
    Includes calibration support for accurate mapping.
    """
    
    def __init__(
        self,
        screen_width: int,
        screen_height: int,
        sensitivity: float = 1.5,
        dead_zone: float = 0.1
    ):
        """
        Initialize screen mapper.
        
        Args:
            screen_width: Screen width in pixels
            screen_height: Screen height in pixels
            sensitivity: Cursor movement sensitivity multiplier
            dead_zone: Center dead zone to reduce jitter (0.0-0.5)
        """
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.sensitivity = sensitivity
        self.dead_zone = dead_zone
        
        # Calibration points (gaze ratios at screen corners/center)
        self.calibrated = False
        self.h_min = 0.3  # Default: looking right
        self.h_max = 0.7  # Default: looking left
        self.v_min = 0.3  # Default: looking up
        self.v_max = 0.7  # Default: looking down
    
    def calibrate(
        self,
        h_min: float,
        h_max: float,
        v_min: float,
        v_max: float
    ):
        """
        Set calibration values for gaze mapping.
        
        Args:
            h_min: Horizontal ratio when looking at left edge
            h_max: Horizontal ratio when looking at right edge
            v_min: Vertical ratio when looking at top edge
            v_max: Vertical ratio when looking at bottom edge
        """
        self.h_min = h_min
        self.h_max = h_max
        self.v_min = v_min
        self.v_max = v_max
        self.calibrated = True
    
    def map_to_screen(
        self,
        h_ratio: float,
        v_ratio: float
    ) -> Tuple[int, int]:
        """
        Map gaze ratios to screen coordinates.
        
        Args:
            h_ratio: Horizontal gaze ratio (0.0-1.0)
            v_ratio: Vertical gaze ratio (0.0-1.0)
            
        Returns:
            Screen coordinates (x, y) in pixels
        """
        # Normalize ratios to calibration range
        h_normalized = (h_ratio - self.h_min) / (self.h_max - self.h_min)
        v_normalized = (v_ratio - self.v_min) / (self.v_max - self.v_min)
        
        # Apply dead zone (center region with reduced movement)
        h_normalized = self._apply_dead_zone(h_normalized)
        v_normalized = self._apply_dead_zone(v_normalized)
        
        # Apply sensitivity
        center = 0.5
        h_final = center + (h_normalized - center) * self.sensitivity
        v_final = center + (v_normalized - center) * self.sensitivity
        
        # Clamp to [0, 1]
        h_final = np.clip(h_final, 0.0, 1.0)
        v_final = np.clip(v_final, 0.0, 1.0)
        
        # Map to screen coordinates
        # Invert horizontal to match natural gaze direction
        screen_x = int((1.0 - h_final) * self.screen_width)
        screen_y = int(v_final * self.screen_height)
        
        # Clamp to screen bounds
        screen_x = np.clip(screen_x, 0, self.screen_width - 1)
        screen_y = np.clip(screen_y, 0, self.screen_height - 1)
        
        return screen_x, screen_y
    
    def _apply_dead_zone(self, value: float) -> float:
        """
        Apply dead zone to reduce jitter in center region.
        
        Args:
            value: Normalized value (0.0-1.0)
            
        Returns:
            Value with dead zone applied
        """
        center = 0.5
        half_zone = self.dead_zone / 2
        
        if abs(value - center) < half_zone:
            return center
        
        if value > center:
            return center + (value - center - half_zone) / (0.5 - half_zone) * 0.5
        else:
            return center - (center - value - half_zone) / (0.5 - half_zone) * 0.5


class GazeVectorCalculator:
    """
    Calculates and projects 3D gaze vector for visualization.
    """
    
    def __init__(self, vector_length: float = 100.0):
        """
        Initialize gaze vector calculator.
        
        Args:
            vector_length: Length of the projected gaze vector in pixels
        """
        self.vector_length = vector_length
    
    def calculate_gaze_vector(
        self,
        iris_center: Tuple[float, float],
        h_ratio: float,
        v_ratio: float,
        frame_width: int,
        frame_height: int
    ) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """
        Calculate gaze vector start and end points for visualization.
        
        Args:
            iris_center: (x, y) of iris center in frame coordinates
            h_ratio: Horizontal gaze ratio
            v_ratio: Vertical gaze ratio
            frame_width: Width of video frame
            frame_height: Height of video frame
            
        Returns:
            Tuple of (start_point, end_point) for drawing the vector
        """
        # Convert ratios to direction (-1 to 1)
        h_direction = (h_ratio - 0.5) * 2
        v_direction = (v_ratio - 0.5) * 2
        
        # Calculate end point
        end_x = int(iris_center[0] + h_direction * self.vector_length)
        end_y = int(iris_center[1] + v_direction * self.vector_length)
        
        # Clamp to frame bounds
        end_x = np.clip(end_x, 0, frame_width - 1)
        end_y = np.clip(end_y, 0, frame_height - 1)
        
        start = (int(iris_center[0]), int(iris_center[1]))
        end = (end_x, end_y)
        
        return start, end


def get_landmark_point(
    landmarks,
    index: int,
    frame_width: int,
    frame_height: int
) -> Tuple[float, float]:
    """
    Extract a single landmark point as frame coordinates.
    
    Args:
        landmarks: MediaPipe face landmarks
        index: Landmark index
        frame_width: Width of video frame
        frame_height: Height of video frame
        
    Returns:
        (x, y) coordinates in pixels
    """
    landmark = landmarks.landmark[index]
    x = landmark.x * frame_width
    y = landmark.y * frame_height
    return x, y


def get_eye_landmarks_for_ear(
    landmarks,
    eye_indices: List[int],
    frame_width: int,
    frame_height: int
) -> List[Tuple[float, float]]:
    """
    Extract 6 eye landmarks for EAR calculation.
    
    Args:
        landmarks: MediaPipe face landmarks
        eye_indices: List of 6 landmark indices for the eye
        frame_width: Width of video frame
        frame_height: Height of video frame
        
    Returns:
        List of 6 (x, y) coordinates
    """
    return [
        get_landmark_point(landmarks, idx, frame_width, frame_height)
        for idx in eye_indices
    ]
