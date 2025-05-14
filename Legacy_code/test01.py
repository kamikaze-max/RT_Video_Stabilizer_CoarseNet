# gui.py

import sys
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QLabel, QVBoxLayout, QFileDialog
from PyQt5.QtCore import QThread, pyqtSignal
from realtime.inference import run_real_time_stabilization


class StabilizationThread(QThread):
    finished = pyqtSignal()

    def run(self):
        run_real_time_stabilization()
        self.finished.emit()


class VideoStabilizerApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RT Video Stabilizer")
        self.setGeometry(100, 100, 300, 150)

        self.init_ui()
        self.thread = None

    def init_ui(self):
        layout = QVBoxLayout()

        self.label = QLabel("Deep Learning Real-Time Video Stabilizer", self)
        layout.addWidget(self.label)

        self.start_btn = QPushButton("Start Webcam", self)
        self.start_btn.clicked.connect(self.start_stabilization)
        layout.addWidget(self.start_btn)

        self.stop_btn = QPushButton("Stop", self)
        self.stop_btn.clicked.connect(self.stop_stabilization)
        self.stop_btn.setEnabled(False)
        layout.addWidget(self.stop_btn)

        self.setLayout(layout)

    def start_stabilization(self):
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)

        self.thread = StabilizationThread()
        self.thread.finished.connect(self.reset_buttons)
        self.thread.start()

    def stop_stabilization(self):
        # GUI stop is currently manual via 'q' in OpenCV window
        self.label.setText("Please press 'q' in the video window.")
    
    def reset_buttons(self):
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = VideoStabilizerApp()
    window.show()
    sys.exit(app.exec_())

