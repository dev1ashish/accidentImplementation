# Video Processing System

A computer vision system for real-time accident detection in traffic camera feeds.

## System Architecture

### Core Components
1. **Backend Services**
   - Master Service: Orchestrates communication between modules
   - Detection Service: Vehicle detection using YOLO
   - Tracker Service: Vehicle tracking using MOSSE/Dlib
   - Crash Service: Accident detection and analysis

2. **Client Services**
   - GUI Interface: Video playback and results display
   - Camera Node: Video input processing

## Processing Pipeline

1. **Video Input**
   ```python:Argus/RunCamera.py
   startLine: 130
   endLine: 141
   ```
   - Accepts .mp4, .mkv, .avi formats
   - Videos are processed through CameraNode

2. **Vehicle Detection**
   - Uses YOLO neural network
   - Option to use pre-saved detections
   - Outputs bounding boxes for detected vehicles

3. **Vehicle Tracking**
   ```python:Argus/Debug.py
   startLine: 207
   endLine: 219
   ```
   - MOSSE/Dlib tracker implementation
   - Track-compensated frame interpolation (TCFI)
   - Maintains vehicle IDs across frames

4. **Crash Detection**
   - Two-phase detection:
     a. Crash estimation algorithm
     b. ViF (Violent Flow) descriptor + SVM classification
   - 93% detection accuracy

## Data Flow

1. **Input Processing**
   ```python:Argus/RunCamera.py
   startLine: 89
   endLine: 102
   ```
   - Video loaded through GUI
   - Assigned unique camera_id
   - Sent to backend services

2. **Backend Processing**
   ```python:Argus/run_argus.py
   startLine: 7
   endLine: 19
   ```
   - Parallel processing through services
   - ZMQ-based communication
   - Real-time frame analysis

3. **Output Generation**
   - Crash sequences saved to `saved_crash_vid/`
   - Regular sequences saved to `saved_frames_vid/`
   - Real-time notifications through GUI

## Configuration Options

Located in `System/Data/Constants.py`:

1. **Detection**
   - `Work_Detect_Files`: Toggle between YOLO/pre-saved detections

2. **Tracking**
   - `Work_Tracker_Type_Mosse`: Choose tracker implementation
   - `Work_Tracker_Interpolation`: Enable/disable TCFI

3. **Crash Detection**
   - `Work_Crash_Estimation_Only`: Toggle full ViF processing