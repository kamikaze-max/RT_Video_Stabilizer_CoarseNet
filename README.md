# Real-Time Video Stabilizer with CoarseNet

This project implements a real-time video stabilization system using a lightweight deep learning model called CoarseNet. It was developed as part of a Master's-level artificial intelligence assignment for Mo-Sys, targeting media production use cases in real-time camera motion correction.

Final script: `RT_video_stabilization.py`

## Features

* Real-time stabilization of video files and live webcam input
* Deep learning-based motion estimation using a U-Net (CoarseNet)
* Kalman filter smoothing for stable output
* Auto-cropping to reduce border artifacts
* Live FPS monitoring and video output recording
* Qt-based graphical interface (PyQt5)

## Technical Overview

* **Model:** CoarseNet (U-Net + fully connected layers)
* **Training Dataset:** DAVIS (480p)
* **Supervision:** Synthetic jitter with inverse motion vectors
* **Inference Pipeline:** Optical flow → \[dx, dy, θ] transform → affine warp
* **Stabilization:** Global warp + Kalman filtering

## Project Structure

```
RT_Video_Stabilizer_CoarseNet/
├── RT_video_stabilization.py        # GUI and runtime logic
├── models/
│   └── coarse_net.py                # CoarseNet model architecture
├── utils/
│   └── flow.py                      # Optical flow utilities
├── realtime/
│   └── inference.py                 # Warping and transformation logic
├── training/
│   ├── dataset.py                   # DAVIS loader
│   └── train_coarsenet.py          # Model training script
├── pretrained/
│   └── coarsenet_trained.pth       # Trained weights
├── videos/
│   ├── test.mp4
│   └── output_stabilized.mp4
├── assets/
│   └── screenshots/
├── requirements.txt
└── README.md
```

## Installation

1. Clone the repository:

```bash
git clone https://github.com/kamikaze-max/RT_Video_Stabilizer_CoarseNet.git
cd RT_Video_Stabilizer_CoarseNet
```

2. Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Linux/macOS
venv\Scripts\activate     # On Windows
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Running the Stabilizer

Launch the graphical interface:

```bash
python RT_video_stabilization.py
```

The GUI allows you to:

* Load a video file or activate the webcam
* Start and stop stabilization
* Save output to `output_stabilized.mp4`
* View input and stabilized video side-by-side

## Training the CoarseNet Model

To train CoarseNet from scratch using the DAVIS dataset:

```bash
python training/train_coarsenet.py
```

Ensure that DAVIS frames are placed under:

```
data/DAVIS/JPEGImages/480p/<video_name>/
```

The script applies synthetic jitter and learns to predict the inverse transformation using optical flow as input.

## Academic and Industry Context

This project was developed as part of a postgraduate assignment in the MSc Artificial Intelligence for Media programme. It applies real-time computer vision and machine learning for stabilization in live camera systems.

It is loosely inspired by:
Choi, J., Park, K., & Kweon, I.S. (2021). *Self-Supervised Real-Time Video Stabilization*. British Machine Vision Conference (BMVC). [https://arxiv.org/abs/2111.05980](https://arxiv.org/abs/2111.05980)

Note: This implementation only uses a CoarseNet-style model and does not include FineNet, MarginNet, or unsupervised loss.

## Author

Karan Kapadia
MSc Artificial Intelligence for Media
GitHub: [kamikaze-max](https://github.com/kamikaze-max)

## License

This repository is provided for academic and demonstration purposes. Please contact the author for reuse, collaboration, or industry use.

