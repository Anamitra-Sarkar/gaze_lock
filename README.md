# GAZE-LOCK

**Real-Time Eye Tracking & Control System**

Control your mouse cursor using only Eye Gaze (Iris Tracking) and Blinks.

## Features

- **Iris Tracking**: Uses MediaPipe Face Mesh with refined landmarks for precise iris detection
- **Cursor Control**: Maps eye gaze to screen coordinates for hands-free mouse control
- **Blink Detection**: Click by blinking using Eye Aspect Ratio (EAR) algorithm
- **Smooth Movement**: Kalman Filter or Moving Average for jitter-free cursor movement
- **Real-time HUD**: Visual feedback showing iris tracking, gaze vectors, and EAR status
- **Safety Controls**: Multiple ways to quit or disable the application

## Requirements

- Python 3.10+
- Webcam
- Display (for cursor control)

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
# Basic usage
python main.py

# Debug mode with extra visualizations
python main.py --debug

# Visualization only (no cursor control)
python main.py --no-cursor

# Use Moving Average instead of Kalman Filter
python main.py --moving-average
```

## Controls

| Key | Action |
|-----|--------|
| `q` or `ESC` | Quit the application (safety kill switch) |
| `c` | Toggle cursor control on/off |
| `d` | Toggle debug mode on/off |
| `r` | Reset smoothing filters |

## Safety Features

- Press `q` or `ESC` at any time to instantly quit
- Move mouse to any screen corner to trigger PyAutoGUI failsafe
- Toggle cursor control with `c` key without stopping the application

## Architecture

```
gaze_lock/
├── main.py           # Main application with webcam, HUD, and cursor control
├── gaze_math.py      # Mathematical utilities (gaze ratios, EAR, smoothing)
├── requirements.txt  # Python dependencies
└── README.md         # This file
```

### Core Components

1. **Iris Extraction**: MediaPipe Face Mesh with `refine_landmarks=True` for iris landmarks
2. **Gaze Ratio Calculation**: Relative position of iris between eye corners
3. **Eye Aspect Ratio (EAR)**: Blink detection algorithm
4. **Smoothing**: Kalman Filter (default) or Moving Average for stable cursor
5. **Screen Mapping**: Converts gaze ratios to screen coordinates with calibration support

## How It Works

1. **Capture**: Webcam captures your face in real-time
2. **Detect**: MediaPipe detects face landmarks including iris centers
3. **Calculate**: Compute horizontal/vertical gaze ratios from iris position
4. **Smooth**: Apply Kalman Filter to reduce jitter
5. **Map**: Convert gaze ratios to screen coordinates
6. **Control**: Move cursor and detect blinks for clicks
7. **Display**: Show HUD with visual feedback

## Tips for Best Results

- Ensure good lighting on your face
- Position camera at eye level
- Keep your head relatively still
- Calibrate for your specific setup if needed
- Use `--debug` mode to see detailed tracking information

## License

MIT License - See LICENSE file for details.
