from dataclasses import dataclass
from enum import IntEnum
import logging
from time import perf_counter
from araviq6 import CameraProcessWidget, VideoFrameWorker
import cv2
from PySide6.QtCore import Signal, Qt
from PySide6.QtWidgets import (
    QWidget,
    QHBoxLayout,
    QVBoxLayout,
    QLabel,
    QSpinBox,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QProgressBar,
    QLineEdit,
)
from numba_pokemon_prngs.xorshift import Xorshift128

from core.util import save_config_json, load_config_json
from core.blink_tracker import MunchlaxBlinkTracker
from core.blink_predictor import MunchlaxBlinkPredictor
from core.blink_progress_bar import BlinkProgressBar


@dataclass
class BoundingBox:
    x: int
    y: int
    width: int
    height: int


class ProcessStep(IntEnum):
    NONE = 0
    BLUR = 1
    GRAY = 2
    GRADIENT = 3
    THRESHOLD = 4


class MunchlaxCameraWorker(VideoFrameWorker):
    def __init__(
        self, contour_receiver, blink_receiver, rng_state_receiver, progress_reciever
    ):
        super().__init__()
        self.contour_receiver = contour_receiver
        self.blink_receiver = blink_receiver
        self.blink_predictor = None
        self.gaussian_size = 1
        self.pixel_threshold = 80
        self.crop = BoundingBox(0, 0, 0, 0)
        self.target_contour = None
        self.contour_threshold = 0.07
        self.full_screen = False
        self.process_step = ProcessStep.GRADIENT
        self.blink_tracker = MunchlaxBlinkTracker(rng_state_receiver, progress_reciever)

    def processArray(self, array):
        timestamp = perf_counter()
        if 0 in array.shape:
            return array
        crop = self.crop
        if crop.width == 0:
            crop.width = array.shape[1] - crop.x
        if crop.height == 0:
            crop.height = array.shape[0] - crop.y

        cropped = array[crop.y : crop.y + crop.height, crop.x : crop.x + crop.width]
        if 0 in cropped.shape:
            return array
        gaussian_size = (self.gaussian_size - 1) * 2 + 1
        # blur to reduce noise
        blurred = cv2.GaussianBlur(cropped, (gaussian_size, gaussian_size), 0)
        gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
        # compute sobel gradient to detect edges
        gradient_x = cv2.Sobel(
            gray,
            cv2.CV_16S,
            1,
            0,
            ksize=3,
            scale=1,
            delta=0,
            borderType=cv2.BORDER_DEFAULT,
        )
        gradient_y = cv2.Sobel(
            gray,
            cv2.CV_16S,
            0,
            1,
            ksize=3,
            scale=1,
            delta=0,
            borderType=cv2.BORDER_DEFAULT,
        )
        gradient = cv2.addWeighted(
            cv2.convertScaleAbs(gradient_x),
            0.5,
            cv2.convertScaleAbs(gradient_y),
            0.5,
            0,
        )
        # threshold to limit to high intensity edges
        _, thresholded = cv2.threshold(
            gradient, self.pixel_threshold, 255, cv2.THRESH_BINARY
        )
        # find potential shapes
        contours, _ = cv2.findContours(
            thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        biggest_contour = (
            max(cv2.contourArea(c) for c in contours) if contours else None
        )
        if biggest_contour is not None:
            # remove contours that are too small relative to the biggest
            # TODO: does this ever limit things? should it be configurable?
            contours = [
                c for c in contours if cv2.contourArea(c) > 0.25 * biggest_contour
            ]

        # report all contours back to the main thread
        self.contour_receiver.emit((cropped, contours))
        # determine which step of the process to display
        match self.process_step:
            case ProcessStep.NONE:
                colored = cropped.copy()
            case ProcessStep.BLUR:
                colored = blurred.copy()
            case ProcessStep.GRAY:
                colored = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            case ProcessStep.GRADIENT:
                colored = cv2.cvtColor(gradient, cv2.COLOR_GRAY2BGR)
            case ProcessStep.THRESHOLD:
                colored = cv2.cvtColor(thresholded, cv2.COLOR_GRAY2BGR)

        cv2.rectangle(
            colored, (0, 0), (crop.width - 1, crop.height - 1), (0, 0, 255), 2
        )
        # target contour existing = currently tracking
        if self.target_contour is not None:
            any_matches = False
            for contour in contours:
                # if any of the contours are close enough to the target
                difference = cv2.matchShapes(
                    self.target_contour, contour, cv2.CONTOURS_MATCH_I1, 0.0
                )
                # draw over the contour and report a match
                if difference < self.contour_threshold:
                    cv2.drawContours(colored, [contour], -1, (0, 255, 0), 3)
                    any_matches = True
            # report if the target contour is detected
            self.blink_receiver.emit((any_matches, timestamp))

        if self.full_screen:
            copied = array.copy()
            copied[crop.y : crop.y + crop.height, crop.x : crop.x + crop.width] = (
                colored
            )
            return copied
        return colored

    def set_gaussian_size(self, value):
        self.gaussian_size = value

    def set_pixel_threshold(self, value):
        self.pixel_threshold = value

    def set_crop(self, value):
        self.crop = value

    def set_target_contour(self, value):
        if value is not None:
            self.blink_tracker.reset()
        self.target_contour = value

    def set_contour_threshold(self, value):
        self.contour_threshold = value

    def set_full_screen(self, value):
        self.full_screen = value

    def set_process_step(self, value):
        self.process_step = ProcessStep(value)

    def set_epsilon(self, value):
        self.blink_tracker.set_epsilon(value)

    def set_offset(self, value):
        self.blink_tracker.set_offset(value)

    def set_leeway(self, value):
        self.blink_tracker.set_leeway(value)


class MunchlaxCamera(QWidget):
    received_contours = Signal(tuple)
    received_blink_state = Signal(tuple)
    received_rng_state = Signal(tuple)
    received_predicted_blink = Signal(tuple)
    received_progress = Signal(int)

    @staticmethod
    def labeled_widget(label, widget_type, *args, **kwargs):
        layout = QHBoxLayout()
        label = QLabel(label)
        layout.addWidget(label)
        widget = widget_type(*args, **kwargs)
        layout.addWidget(widget)
        return layout, widget, label

    def __init__(self) -> None:
        super().__init__()
        self.running = False
        self.camera = None
        self.blink_predictor = None
        self.received_rng_state.connect(self.on_rng_state_found)
        self.received_predicted_blink.connect(self.on_predicted_blink)
        self.received_progress.connect(self.on_entropy_progress)

        self.full_layout = QVBoxLayout(self)
        self.columns_layout = QHBoxLayout()
        self.left_column = QVBoxLayout()
        self.right_column = QVBoxLayout()
        self.display = CameraProcessWidget()
        self.worker = MunchlaxCameraWorker(
            self.received_contours,
            self.received_blink_state,
            self.received_rng_state,
            self.received_progress,
        )
        self.display.setWorker(self.worker)
        self.full_screen_checkbox = QCheckBox()
        self.full_screen_checkbox.setText("Full Screen")
        self.full_screen_checkbox.setChecked(False)
        self.full_screen_checkbox.stateChanged.connect(self.worker.set_full_screen)
        gs_layout, self.gaussian_size_input, _ = self.labeled_widget(
            "Gaussian Radius", QSpinBox
        )
        process_step_layout, self.process_step_combobox, _ = self.labeled_widget(
            "Process Step", QComboBox
        )
        self.process_step_combobox.addItems([step.name for step in ProcessStep])
        self.process_step_combobox.setCurrentIndex(ProcessStep.THRESHOLD.value)
        self.process_step_combobox.currentIndexChanged.connect(
            self.worker.set_process_step
        )

        self.gaussian_size_input.setRange(1, 100)
        self.gaussian_size_input.setValue(1)
        self.gaussian_size_input.valueChanged.connect(self.worker.set_gaussian_size)
        pt_layout, self.pixel_threshold_input, _ = self.labeled_widget(
            "Pixel Threshold", QSpinBox
        )
        self.pixel_threshold_input.setRange(0, 255)
        self.pixel_threshold_input.setValue(80)
        self.pixel_threshold_input.valueChanged.connect(self.worker.set_pixel_threshold)
        ct_layout, self.contour_threshold_input, _ = self.labeled_widget(
            "Contour Threshold", QDoubleSpinBox
        )
        self.contour_threshold_input.setRange(0.0, 1.0)
        self.contour_threshold_input.setValue(0.07)
        self.contour_threshold_input.setSingleStep(0.01)
        self.contour_threshold_input.valueChanged.connect(
            self.worker.set_contour_threshold
        )
        cx_layout, self.crop_x_input, _ = self.labeled_widget("Crop X", QSpinBox)
        self.crop_x_input.setRange(0, 16384)
        cy_layout, self.crop_y_input, _ = self.labeled_widget("Crop Y", QSpinBox)
        self.crop_y_input.setRange(0, 16384)
        cw_layout, self.crop_width_input, _ = self.labeled_widget(
            "Crop Width", QSpinBox
        )
        self.crop_width_input.setRange(0, 16384)
        ch_layout, self.crop_height_input, _ = self.labeled_widget(
            "Crop Height", QSpinBox
        )
        self.crop_height_input.setRange(0, 16384)

        def set_crop():
            self.worker.set_crop(
                BoundingBox(
                    self.crop_x_input.value(),
                    self.crop_y_input.value(),
                    self.crop_width_input.value(),
                    self.crop_height_input.value(),
                )
            )

        self.crop_x_input.valueChanged.connect(set_crop)
        self.crop_y_input.valueChanged.connect(set_crop)
        self.crop_width_input.valueChanged.connect(set_crop)
        self.crop_height_input.valueChanged.connect(set_crop)

        be_layout, self.blink_epsilon_input, _ = self.labeled_widget(
            "Blink Epsilon", QDoubleSpinBox
        )
        self.blink_epsilon_input.setDecimals(5)
        self.blink_epsilon_input.setRange(0.0, 10.0)
        self.blink_epsilon_input.setValue(0.1)
        self.blink_epsilon_input.setSingleStep(0.01)
        self.blink_epsilon_input.valueChanged.connect(self.worker.set_epsilon)

        bo_layout, self.blink_offset_input, _ = self.labeled_widget(
            "Blink Offset", QDoubleSpinBox
        )
        self.blink_offset_input.setDecimals(5)
        self.blink_offset_input.setRange(0.0, 10.0)
        self.blink_offset_input.setValue(0.287155)
        self.blink_offset_input.setSingleStep(0.01)
        self.blink_offset_input.valueChanged.connect(self.worker.set_offset)

        bl_layout, self.blink_leeway_input, _ = self.labeled_widget(
            "Blink Leeway", QDoubleSpinBox
        )
        self.blink_leeway_input.setDecimals(5)
        self.blink_leeway_input.setRange(0.0, 10.0)
        self.blink_leeway_input.setValue(0.1)
        self.blink_leeway_input.setSingleStep(0.01)
        self.blink_leeway_input.valueChanged.connect(self.worker.set_leeway)

        self.info_display = QLabel()
        self.info_display.setAlignment(Qt.AlignCenter)
        self.info_display.setTextInteractionFlags(Qt.TextSelectableByMouse)

        seed_layout, self.seed_input, _ = self.labeled_widget(
            "Initial State", QLineEdit
        )

        self.blink_progress_bar = BlinkProgressBar()
        self.blink_progress_bar.set_target_timestamp(0)
        self.blink_progress_bar.hide()
        self.entropy_progress_bar = QProgressBar()
        self.entropy_progress_bar.setValue(0)
        self.entropy_progress_bar.setMaximum(128)
        self.entropy_progress_bar.hide()

        self.full_layout.addWidget(self.full_screen_checkbox)
        self.full_layout.addLayout(self.columns_layout)
        self.columns_layout.addLayout(self.left_column)
        self.columns_layout.addLayout(self.right_column)

        self.left_column.addLayout(process_step_layout)
        self.left_column.addLayout(gs_layout)
        self.left_column.addLayout(pt_layout)
        self.left_column.addLayout(ct_layout)
        self.right_column.addLayout(cx_layout)
        self.right_column.addLayout(cy_layout)
        self.right_column.addLayout(cw_layout)
        self.right_column.addLayout(ch_layout)
        self.full_layout.addLayout(be_layout)
        self.full_layout.addLayout(bo_layout)
        self.full_layout.addLayout(bl_layout)
        self.full_layout.addLayout(seed_layout)
        self.full_layout.addWidget(self.info_display)
        self.full_layout.addWidget(self.blink_progress_bar)
        self.full_layout.addWidget(self.entropy_progress_bar)
        self.full_layout.addWidget(self.display, 1)

        config = load_config_json()
        if config is not None:
            if "full_screen" in config:
                self.full_screen_checkbox.setChecked(config["full_screen"])
                self.worker.set_full_screen(config["full_screen"])
            if "gaussian_size" in config:
                self.gaussian_size_input.setValue(config["gaussian_size"])
                self.worker.set_gaussian_size(config["gaussian_size"])
            if "pixel_threshold" in config:
                self.pixel_threshold_input.setValue(config["pixel_threshold"])
                self.worker.set_pixel_threshold(config["pixel_threshold"])
            if "process_step" in config:
                self.process_step_combobox.setCurrentIndex(config["process_step"])
                self.worker.set_process_step(config["process_step"])
            if "contour_threshold" in config:
                self.contour_threshold_input.setValue(config["contour_threshold"])
                self.worker.set_contour_threshold(config["contour_threshold"])
            if "crop_x" in config:
                self.crop_x_input.setValue(config["crop_x"])
            if "crop_y" in config:
                self.crop_y_input.setValue(config["crop_y"])
            if "crop_width" in config:
                self.crop_width_input.setValue(config["crop_width"])
            if "crop_height" in config:
                self.crop_height_input.setValue(config["crop_height"])
            self.worker.set_crop(
                BoundingBox(
                    self.crop_x_input.value(),
                    self.crop_y_input.value(),
                    self.crop_width_input.value(),
                    self.crop_height_input.value(),
                )
            )
            if "blink_epsilon" in config:
                self.blink_epsilon_input.setValue(config["blink_epsilon"])
                self.worker.set_epsilon(config["blink_epsilon"])
            if "blink_offset" in config:
                self.blink_offset_input.setValue(config["blink_offset"])
                self.worker.set_offset(config["blink_offset"])
            if "blink_leeway" in config:
                self.blink_leeway_input.setValue(config["blink_leeway"])
                self.worker.set_leeway(config["blink_leeway"])

    def request_contours(self, callback):
        def contours_callback(data):
            self.received_contours.disconnect(contours_callback)
            (cropped, contours) = data
            self.reset_blink_tracker()
            callback(cropped, contours)
            self.received_blink_state.disconnect()
            self.received_blink_state.connect(self.worker.blink_tracker.process_data)
            self.entropy_progress_bar.show()

        self.received_contours.connect(contours_callback)

    def on_entropy_progress(self, entropy):
        if entropy >= 128:
            self.entropy_progress_bar.setFormat(
                "(Extra entropy required) Entropy: %v/128(?)"
            )
        else:
            self.entropy_progress_bar.setFormat("Entropy: %v/128")
        self.entropy_progress_bar.setValue(entropy)

    def reset_blink_tracker(self):
        self.set_target_contour(None)
        self.entropy_progress_bar.setValue(0)
        self.entropy_progress_bar.setFormat("Entropy: %v/128")
        self.blink_progress_bar.set_target_timestamp(0)
        self.blink_progress_bar.hide()
        self.entropy_progress_bar.hide()
        self.info_display.setText("")

    def on_rng_state_found(self, data):
        rng_state, last_blink = data
        self.reset_blink_tracker()
        self.blink_predictor = MunchlaxBlinkPredictor(
            self.received_predicted_blink,
            rng_state,
            last_blink.timestamp,
        )
        rng = Xorshift128(*rng_state)
        rng.next()
        self.seed_input.setText(
            f"{rng.state[0]:08X} {rng.state[1]:08X} {rng.state[2]:08X} {rng.state[3]:08X}"
        )
        self.blink_predictor.start()
        self.blink_progress_bar.show()
        self.entropy_progress_bar.hide()

    def on_predicted_blink(self, data):
        (next_blink, timestamp, advance) = data
        self.blink_progress_bar.set_target_timestamp(next_blink)
        logging.info(" Predicted blink! Next: %.3fs", next_blink - timestamp)
        logging.info(" Advance: %d", advance)
        self.info_display.setText(f" Advance: {advance}")

    def set_target_contour(self, contour):
        if self.blink_predictor is not None:
            self.blink_predictor.running = False
            self.blink_predictor = None
        self.worker.set_target_contour(contour)

    def start_camera(self, camera_selector) -> None:
        if self.camera is not None:
            self.camera.stop()
        self.reset_blink_tracker()
        self.camera = camera_selector.selected_camera()
        self.display.setCamera(self.camera)
        self.camera.start()
        self.running = True

    def stop_camera(self) -> None:
        self.reset_blink_tracker()
        if self.camera is None:
            return
        self.camera.stop()
        self.running = False

    def closeEvent(self, event):
        config = {
            "full_screen": self.full_screen_checkbox.isChecked(),
            "process_step": self.process_step_combobox.currentIndex(),
            "gaussian_size": self.gaussian_size_input.value(),
            "pixel_threshold": self.pixel_threshold_input.value(),
            "contour_threshold": self.contour_threshold_input.value(),
            "crop_x": self.crop_x_input.value(),
            "crop_y": self.crop_y_input.value(),
            "crop_width": self.crop_width_input.value(),
            "crop_height": self.crop_height_input.value(),
            "blink_epsilon": self.blink_epsilon_input.value(),
            "blink_offset": self.blink_offset_input.value(),
            "blink_leeway": self.blink_leeway_input.value(),
        }
        save_config_json(config)
        self.stop_camera()
        super().closeEvent(event)
