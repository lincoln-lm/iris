from PySide6.QtMultimedia import (
    QMediaDevices,
    QCamera,
)
from PySide6.QtWidgets import QComboBox


class CameraSelector(QComboBox):
    def __init__(self) -> None:
        super().__init__()
        for camera in QMediaDevices.videoInputs():
            self.addItem(camera.description(), camera)

    def selected_camera(self) -> QCamera:
        return QCamera(self.currentData())
