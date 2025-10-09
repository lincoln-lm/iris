"""QWidget window for the main program"""

from PySide6.QtWidgets import (
    QVBoxLayout,
    QWidget,
    QLabel,
    QPushButton,
    QDialog,
    QComboBox,
)
from PySide6.QtGui import QPixmap
from araviq6.array2qvideoframe import array2qvideoframe
import numpy as np
import cv2
from .camera_selector import CameraSelector
from .camera import MunchlaxCamera


class CapturedEyeDialog(QDialog):
    def __init__(self, base_frame, contours) -> None:
        super().__init__()
        self.base_frame = base_frame
        self.contours = contours
        self.setup_widgets()
        self.initial_size = (self.width(), self.height())
        self.draw_contours(0)

    def draw_contours(self, index) -> None:
        frame = self.base_frame.copy()
        cv2.drawContours(frame, [self.contours[index]], -1, (0, 255, 0), 3)
        if self.initial_size[0] < self.initial_size[1]:
            ratio = frame.shape[0] / frame.shape[1]
            frame = cv2.resize(
                frame,
                (self.initial_size[0], int(self.initial_size[0] * ratio)),
            )
        else:
            ratio = frame.shape[1] / frame.shape[0]
            frame = cv2.resize(
                frame,
                (int(self.initial_size[1] * ratio), self.initial_size[1]),
            )
        video_frame = array2qvideoframe(frame)
        self.base_frame_label.setPixmap(QPixmap.fromImage(video_frame.toImage()))

    def get_selected_contour(self) -> np.ndarray:
        return self.contours[int(self.contour_selection.currentIndex())]

    def setup_widgets(self) -> None:
        self.widget_layout = QVBoxLayout(self)
        self.contour_selection = QComboBox()
        self.contour_selection.addItems([str(i) for i in range(len(self.contours))])
        self.base_frame_label = QLabel()
        self.contour_selection.currentIndexChanged.connect(self.draw_contours)
        self.confirm_button = QPushButton("Confirm")
        self.confirm_button.clicked.connect(self.accept)
        self.widget_layout.addWidget(self.contour_selection)
        self.widget_layout.addWidget(self.confirm_button)
        self.widget_layout.addWidget(self.base_frame_label)


class MainWindow(QWidget):
    """QWidget window for the main program"""

    def __init__(self) -> None:
        super().__init__()
        self.setup_widgets()

        self.show()

    def toggle_camera(self) -> None:
        if self.munchlax_camera.running:
            self.munchlax_camera.stop_camera()
            self.start_camera_button.setText("Start Camera")
        else:
            self.munchlax_camera.start_camera(self.camera_selector)
            self.start_camera_button.setText("Stop Camera")
        self.capture_eye_button.setEnabled(self.munchlax_camera.running)

    def capture_eye(self) -> None:
        def callback(base_frame, contours):
            dialog = CapturedEyeDialog(base_frame.copy(), contours)
            if dialog.exec():
                contour = dialog.get_selected_contour()
                self.munchlax_camera.set_target_contour(contour)

        self.munchlax_camera.request_contours(callback)

    def setup_widgets(self) -> None:
        """Construct main window widgets"""
        self.main_layout = QVBoxLayout(self)
        self.camera_selector = CameraSelector()
        self.start_camera_button = QPushButton("Start Camera")
        self.start_camera_button.clicked.connect(self.toggle_camera)
        self.munchlax_camera = MunchlaxCamera()
        self.capture_eye_button = QPushButton("Capture Eye")
        self.capture_eye_button.setEnabled(False)
        self.capture_eye_button.clicked.connect(self.capture_eye)
        self.reset_contour_button = QPushButton("Reset Contour")
        self.reset_contour_button.clicked.connect(
            lambda: self.munchlax_camera.reset_blink_tracker()
        )
        self.main_layout.addWidget(self.camera_selector)
        self.main_layout.addWidget(self.start_camera_button)
        self.main_layout.addWidget(self.munchlax_camera)
        self.main_layout.addWidget(self.capture_eye_button)
        self.main_layout.addWidget(self.reset_contour_button)

    def closeEvent(self, event):
        self.munchlax_camera.closeEvent(event)
        super().closeEvent(event)
