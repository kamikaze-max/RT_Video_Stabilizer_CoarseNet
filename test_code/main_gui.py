import sys
import cv2
import torch
import numpy as np
from PyQt5.QtWidgets import (QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout)
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap
from models.coarse_net import CoarseNet
from models.fine_net import FineNet
from utils.flow import compute_optical_flow
from realtime.inference import warp_image_global, warp_image_with_flow

class VideoStabilizerGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Deep Learning Video Stabilizer")
        self.setGeometry(100, 100, 900, 350)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.coarse_net = CoarseNet().to(self.device).eval()
        try:
            self.coarse_net.load_state_dict(torch.load("coarsenet_trained.pth", map_location=self.device))
            print("Loaded trained CoarseNet model.")
        except Exception as e:
            print(f"Warning: Failed to load CoarseNet model. {e}")

        self.fine_net = FineNet().to(self.device).eval()
        try:
            self.fine_net.load_state_dict(torch.load("finenet_trained.pth", map_location=self.device))
            print("Loaded trained FineNet model.")
        except Exception as e:
            print(f"Warning: Failed to load FineNet model. {e}")

        self.cap = None
        self.prev_frame = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        self.init_ui()

    def init_ui(self):
        self.label_original = QLabel("Original")
        self.label_coarse = QLabel("Coarse")
        self.label_final = QLabel("Final")

        for label in [self.label_original, self.label_coarse, self.label_final]:
            label.setFixedSize(256, 256)
            label.setStyleSheet("border: 1px solid black")
            label.setAlignment(Qt.AlignCenter)

        self.start_btn = QPushButton("Start Webcam")
        self.start_btn.clicked.connect(self.start_webcam)

        self.stop_btn = QPushButton("Stop")
        self.stop_btn.clicked.connect(self.stop_webcam)
        self.stop_btn.setEnabled(False)

        h_layout = QHBoxLayout()
        h_layout.addWidget(self.label_original)
        h_layout.addWidget(self.label_coarse)
        h_layout.addWidget(self.label_final)

        btn_layout = QHBoxLayout()
        btn_layout.addWidget(self.start_btn)
        btn_layout.addWidget(self.stop_btn)

        v_layout = QVBoxLayout()
        v_layout.addLayout(h_layout)
        v_layout.addLayout(btn_layout)

        self.setLayout(v_layout)

    def start_webcam(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Webcam not accessible")
            return

        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.timer.start(30)

    def stop_webcam(self):
        self.timer.stop()
        if self.cap:
            self.cap.release()
            self.cap = None
        self.prev_frame = None
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        frame_resized = cv2.resize(frame, (256, 256))

        if self.prev_frame is not None:
            flow_np = compute_optical_flow(self.prev_frame, frame_resized)
            flow_tensor = torch.from_numpy(flow_np).permute(2, 0, 1).unsqueeze(0).float().to(self.device)

            with torch.no_grad():
                coarse_params = self.coarse_net(flow_tensor).cpu().numpy()[0]
                coarse_stabilized = warp_image_global(frame_resized, coarse_params)

                smoothed_flow_tensor = self.fine_net(flow_tensor)
                smoothed_flow_np = smoothed_flow_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
                final_stabilized = warp_image_with_flow(coarse_stabilized, smoothed_flow_np)

            self.display_frame(self.label_original, frame_resized)
            self.display_frame(self.label_coarse, coarse_stabilized)
            self.display_frame(self.label_final, final_stabilized)
        else:
            self.display_frame(self.label_original, frame_resized)
            self.display_frame(self.label_coarse, frame_resized)
            self.display_frame(self.label_final, frame_resized)

        self.prev_frame = frame_resized.copy()

    def display_frame(self, label, frame):
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        q_img = QImage(rgb_image.data, rgb_image.shape[1], rgb_image.shape[0], QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        label.setPixmap(pixmap)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = VideoStabilizerGUI()
    window.show()
    sys.exit(app.exec_())
