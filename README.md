# Iskomate Individual Mode

A comprehensive engagement detection and real-time analysis system using deep learning and computer vision.

## Quick Start

### Prerequisites

- Python 3.9 or higher
- Git

### Setup Instructions

1. **Clone the repository**

   ```bash
   git clone https://github.com/romulojr-dev/iskomate_individual_mode.git
   cd iskomate_individual_mode
   ```

2. **Create a virtual environment**

   ```bash
   python -m venv engagement_env
   ```

3. **Activate the virtual environment**

   **Windows (PowerShell):**

   ```powershell
   .\engagement_env\Scripts\Activate.ps1
   ```

   **Windows (Command Prompt):**

   ```cmd
   .\engagement_env\Scripts\activate.bat
   ```

   **macOS/Linux:**

   ```bash
   source engagement_env/bin/activate
   ```

4. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Project Structure

- **`realtime_engagement.py`** - Real-time engagement detection using MediaPipe
- **`realtime_engagement_openface.py`** - Engagement detection using OpenFace features
- **`realtime_engagement_hybrid.py`** - Hybrid approach combining multiple feature streams
- **`TCCT_Net/`** - Two-Stream Network Architecture for engagement estimation
  - Contains model weights, training/inference scripts, and utilities
- **`gemini/`** - Alternative engagement detection implementations
- **`calibrate_*.py`** - Calibration scripts for normalization and sleep detection
- **`mediapipe_extractor.py`** - MediaPipe feature extraction utilities
- **`ai_server.py`** & **`ai_websocket_server.py`** - WebSocket servers for real-time analysis

## Usage

### Basic Real-time Engagement Detection

```bash
python realtime_engagement.py
```

### With OpenFace Features

```bash
python realtime_engagement_openface.py
```

### Hybrid Detection

```bash
python realtime_engagement_hybrid.py
```

### TCCT-Net Model

For detailed instructions on using the TCCT-Net model, see [TCCT_Net/README.md](TCCT_Net/README.md)

```bash
cd TCCT_Net
python inference.py
```

## OpenFace Setup

OpenFace is required for advanced feature extraction. Follow the setup for your platform:

### Option 1: Windows (Recommended - Pre-built Binaries)

1. **Download OpenFace**
   - Visit: https://github.com/TadasBaltrusaitis/OpenFace/releases
   - Download: `OpenFace_2.2.0_win_x64.zip`

2. **Extract & Install**
   - Extract to a location like `C:\OpenFace\`
   - Verify `FeatureExtraction.exe` exists in the folder

3. **Configure Path**
   - Open `realtime_engagement_openface.py`
   - Update line 35:
     ```python
     OPENFACE_BIN = r"C:\OpenFace\FeatureExtraction.exe"
     ```

4. **Verify Installation**
   ```bash
   C:\OpenFace\FeatureExtraction.exe -help
   ```

5. **Run the Pipeline**
   ```bash
   python realtime_engagement_openface.py
   ```

### Option 2: Linux/macOS (Build from Source)

Follow the official guide: https://github.com/TadasBaltrusaitis/OpenFace/wiki/Unix-Installation

After building, update `realtime_engagement_openface.py`:

```python
OPENFACE_BIN = "/usr/local/bin/FeatureExtraction"  # macOS
# or
OPENFACE_BIN = "/usr/bin/FeatureExtraction"        # Linux
```

### Option 3: Docker (All Platforms)

If you have Docker installed:

```bash
docker run -it -v $(pwd):/workspace algebr/openface:latest \
    FeatureExtraction -f /workspace/video.mp4 -out_dir /workspace/output
```

### Tuning Parameters

Edit `realtime_engagement_openface.py` to optimize performance:

```python
CHUNK_SIZE = 30                    # frames between OpenFace runs
                                   # Smaller = more CPU, lower latency
                                   # Larger = less CPU, higher latency

FRAMES_BETWEEN_PREDICTIONS = 10    # how often to predict engagement

AMPLIFICATION_FACTOR = 2.0         # pose signal amplification

WINDOW_SIZE = 280                  # MUST match TCCT_Net/config.json
```

### Quick Verification

Run this one-time check:

```bash
python QUICKSTART_OPENFACE_PIPELINE.py
```

This verifies:
- OpenFace is installed
- TCCT-Net configuration is valid
- All dependencies are ready

## Key Features

- Real-time facial engagement detection
- Head pose analysis
- Action unit recognition
- Temporal-frequency analysis
- WebSocket-based real-time streaming
- Calibration tools for normalization

## Dependencies

All dependencies are listed in `requirements.txt` and include:

- **Computer Vision**: OpenCV, MediaPipe
- **Deep Learning**: PyTorch
- **Data Processing**: NumPy, Pandas, SciPy
- **Audio**: sounddevice
- **Visualization**: Matplotlib
- **Networking**: websockets

## Model Information

- TCCT-Net: Two-Stream Network Architecture for Fast and Efficient Engagement Estimation
- See `TCCT_Net/README.md` for full paper details and model information

## Hardware Recommendations

- GPU: NVIDIA GPU with CUDA support recommended for real-time inference
- RAM: 8GB minimum, 16GB recommended
- CPU: Intel i5/i7 or equivalent AMD processor

## Configuration

See `QUICK_REFERENCE.txt`, `TUNING_GUIDE.py`, and `OPTIMIZATION_SUMMARY.txt` for:

- Camera calibration settings
- Model tuning parameters
- Performance optimization tips

## Troubleshooting

### Camera Issues

- Use `test_cam.py` to verify your webcam setup
- Check window size configuration in `FIX_WINDOW_SIZE_ERROR.txt`

### Model Issues

- Refer to `calibrate_normalization.py` for normalization calibration
- Check `calibrate_sleep.py` for sleep detection calibration

## License

This project is based on TCCT-Net research. For academic use, please cite the original paper.

## References

- TCCT-Net paper: [arXiv:2404.09474](https://arxiv.org/abs/2404.09474)
- MediaPipe: [https://mediapipe.dev](https://mediapipe.dev)
- OpenFace: [https://github.com/TadasBaltrusaitis/OpenFace](https://github.com/TadasBaltrusaitis/OpenFace)
