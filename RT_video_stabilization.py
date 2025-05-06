import sys
import time
import cv2
import torch
import numpy as np
from PyQt5.QtWidgets import (QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
                             QProgressBar, QFileDialog, QSizePolicy)
from PyQt5.QtCore import QTimer, Qt, QSize
from PyQt5.QtGui import QImage, QPixmap
from models.coarse_net import CoarseNet
from utils.flow import compute_optical_flow
from realtime.inference import warp_image_global

class KalmanFilter1D:
    def __init__(self, q=1e-4, r=1e-1):
        self.q = q
        self.r = r
        self.p = 1.0
        self.x = 0.0

    def update(self, measurement):
        self.p += self.q
        k = self.p / (self.p + self.r)
        self.x += k * (measurement - self.x)
        self.p *= (1 - k)
        return self.x

class StabilizerApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Real-Time Video Stabilizer")
        self.resize(1280, 720)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.target_height = 256

        self.coarse_net = CoarseNet().to(self.device).eval()
        try:
            self.coarse_net.load_state_dict(torch.load("coarsenet_trained.pth", map_location=self.device))
        except Exception as e:
            print(f"Model loading failed: {e}")

        self.kalman_filters_params = [KalmanFilter1D(q=1e-3, r=5e-2) for _ in range(3)]
        self.video_writer = None
        self.init_ui()
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.prev_frame_full = None
        self.last_time = time.time()

    def init_ui(self):
        self.setStyleSheet("""
            QWidget {
                background-color: #2b2b2b;
                color: #e0e0e0;
                font-size: 14px;
            }
            QPushButton {
                background-color: #444;
                border: 1px solid #666;
                padding: 5px;
            }
            QPushButton:hover {
                background-color: #555;
            }
            QLabel {
                color: #f0f0f0;
            }
            QProgressBar {
                border: 1px solid #555;
                background: #3c3c3c;
                text-align: center;
                color: white;
            }
            QProgressBar::chunk {
                background-color: #007acc;
            }
        """)

        self.display_label = QLabel()
        self.display_label.setMinimumSize(960, 512)
        self.display_label.setAlignment(Qt.AlignCenter)
        self.display_label.setStyleSheet("border: 1px solid #555;")
        self.display_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.load_btn = QPushButton("Load Video")
        self.load_btn.clicked.connect(self.load_video)
        self.webcam_btn = QPushButton("Webcam")
        self.webcam_btn.clicked.connect(self.start_webcam)

        self.start_btn = QPushButton("Start")
        self.start_btn.clicked.connect(self.start_processing)
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.clicked.connect(self.stop_processing)

        self.fps_label = QLabel("FPS: 0.0")
        self.status_label = QLabel("Idle")

        self.progress_bar = QProgressBar()

        controls_layout = QVBoxLayout()
        controls_layout.addWidget(self.load_btn)
        controls_layout.addWidget(self.webcam_btn)
        controls_layout.addWidget(self.start_btn)
        controls_layout.addWidget(self.stop_btn)
        controls_layout.addWidget(self.fps_label)
        controls_layout.addWidget(self.status_label)
        controls_layout.addWidget(self.progress_bar)
        controls_layout.addStretch()

        main_layout = QHBoxLayout()
        main_layout.addWidget(self.display_label, stretch=4)
        main_layout.addLayout(controls_layout, stretch=1)

        self.setLayout(main_layout)

    def load_video(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Video File")
        if file_path:
            self.cap = cv2.VideoCapture(file_path)
            total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.progress_bar.setRange(0, total_frames)
            self.progress_bar.setValue(0)
            self.using_webcam = False
            self.status_label.setText("Video Loaded")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.video_writer = cv2.VideoWriter("output_stabilized.mp4", fourcc, fps, (width, height))

    def start_webcam(self):
        self.cap = cv2.VideoCapture(0)
        self.status_label.setText("Webcam Active")
        self.using_webcam = True

    def start_processing(self):
        if self.cap and self.cap.isOpened():
            self.timer.start(30)
            self.status_label.setText("Stabilizing...")

    def stop_processing(self):
        self.timer.stop()
        if self.cap:
            self.cap.release()
            self.cap = None
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None
        self.prev_frame_full = None
        self.status_label.setText("Stopped")
        self.progress_bar.setValue(0)

    def update_frame(self):
        if self.cap is None or not self.cap.isOpened():
            return

        ret, frame_full = self.cap.read()
        if not self.using_webcam:
            current_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
            self.progress_bar.setValue(current_frame)
        if not ret:
            print("✅ Video finished. Saving and closing.")
            self.stop_processing()
            self.status_label.setText("Finished")
            return

        if self.prev_frame_full is None:
            self.prev_frame_full = frame_full.copy()
            stabilized = frame_full.copy()
        else:
            frame_small = cv2.resize(frame_full, (256, 256))
            prev_small = cv2.resize(self.prev_frame_full, (256, 256))
            flow_np = compute_optical_flow(prev_small, frame_small)
            flow_tensor = torch.from_numpy(flow_np).permute(2, 0, 1).unsqueeze(0).float().to(self.device)

            with torch.no_grad():
                pred = self.coarse_net(flow_tensor).cpu().numpy()[0]
                print(f"Predicted transform: dx={pred[0]:.4f}, dy={pred[1]:.4f}, theta={pred[2]:.4f}")

                amplify = 3.0  # Amplify weak webcam motion
                pred[0] *= amplify
                pred[1] *= amplify
                pred[2] *= amplify
                dx = self.kalman_filters_params[0].update(pred[0])
                dy = self.kalman_filters_params[1].update(pred[1])
                theta = self.kalman_filters_params[2].update(pred[2])

                crop_ratio = 0.94
                h, w = frame_full.shape[:2]
                ch = int(h * crop_ratio)
                cw = int(w * crop_ratio)
                y1 = (h - ch) // 2
                x1 = (w - cw) // 2
                warped = cv2.warpAffine(frame_full, cv2.getRotationMatrix2D((w // 2, h // 2), theta, 1), (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
                cropped = warped[y1:y1+ch, x1:x1+cw]
                stabilized = cv2.resize(cropped, (w, h))

        self.prev_frame_full = frame_full.copy()

        

        if self.video_writer:
            self.video_writer.write(stabilized)

        combined = np.hstack((frame_full, stabilized))
        rgb_image = cv2.cvtColor(combined, cv2.COLOR_BGR2RGB)
        h, w, _ = rgb_image.shape
        q_img = QImage(rgb_image.data, w, h, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img).scaled(self.display_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.display_label.setPixmap(pixmap)

        now = time.time()
        fps = 1.0 / (now - self.last_time)
        self.fps_label.setText(f"FPS: {fps:.1f}")
        self.last_time = now

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = StabilizerApp()
    window.show()
    sys.exit(app.exec_())

