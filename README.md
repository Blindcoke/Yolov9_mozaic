# YOLOv9 Mozaic Face Blur

An advanced face detection and blurring tool that uses YOLOv9 and MediaPipe for accurate face detection and applies a dynamic mosaic effect to protect privacy in videos.

## Features

- Real-time face detection using YOLOv9 and MediaPipe
- Dynamic mosaic effect with smooth transitions
- Face tracking across video frames 
- Adjustable blur intensity and face detection parameters
- Handles multiple faces simultaneously
- Supports various video formats
- GPU acceleration support through CUDA (when available)

## Prerequisites

Before running this project, make sure you have the following dependencies installed:

`pip install ultralytics`
`pip install opencv-python`
`pip install mediapipe`
`pip install numpy`
`pip install matplotlib`

You'll also need to download the YOLOv9 model weights. Place them in the `yolo` directory:
- `yolov9c_best.pt` (or your custom trained model)

## Installation

1. Clone the repository:
`git clone https://github.com/Blindcoke/Yolov9_mozaic.git`
`cd Yolov9_mozaic`

2. Install the required packages into your venv:
`pip install -r requirements.txt`

## Usage

Run the script with a video file as an argument:

`python main.py path/to/your/video.mp4`

The processed video will be saved with "_mozaic" appended to the original filename.

## Configuration

The script includes several adjustable parameters:

- `pixel_size_rel`: Controls the mosaic block size (default: 0.015)
- `scale`: Face detection box scaling factor (default: 1.15)  
- `damping_factor`: Controls the smoothness of the tracking (default: 5)
- `buffer_size`: Frame buffer size for smooth transitions (default: 0.6 * FPS)

You can modify these parameters in the code to achieve different effects.

## How It Works

1. **Face Detection**: The system uses a two-stage detection approach:
  - YOLOv9 for initial face detection and tracking
  - MediaPipe for precise face boundary detection

2. **Face Tracking**: The `FaceBook` class maintains a registry of detected faces and tracks them across frames using:
  - Unique ID assignment
  - Position-based tracking 
  - Smooth transitions

3. **Mosaic Effect**: The blur effect is applied using:
  - Dynamic pixel size based on video resolution
  - Gaussian blur for smooth edges
  - Alpha blending for natural transitions


## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- YOLOv9 by WongKinYiu
- MediaPipe by Google
- OpenCV community

