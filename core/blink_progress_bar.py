from PySide6.QtWidgets import QProgressBar
from PySide6.QtCore import QTimer
from time import perf_counter


class BlinkProgressBar(QProgressBar):
    MAXIMUM = 1000
    UPDATE_RATE = 1

    def __init__(self):
        super().__init__()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_bar)
        self.timer.start(self.UPDATE_RATE)
        self.start_timestamp = 0
        self.target_timestamp = 0
        self.setMaximum(self.MAXIMUM)
        self.setMinimum(0)
        self.setFormat("Next Blink: 0s")

    def set_target_timestamp(self, value):
        self.timer.stop()
        self.target_timestamp = value
        self.start_timestamp = perf_counter()
        self.setValue(0)
        self.timer.start(self.UPDATE_RATE)

    def update_bar(self):
        current_time = perf_counter()
        if current_time > self.target_timestamp:
            self.setValue(self.maximum())
            self.timer.stop()
        else:
            self.setValue(
                int(
                    (current_time - self.start_timestamp)
                    / (self.target_timestamp - self.start_timestamp)
                    * self.maximum()
                )
            )
            self.setFormat(f"Next Blink: {self.target_timestamp - current_time:.2f}s")
