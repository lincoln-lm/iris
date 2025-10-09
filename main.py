"""Main script for iris"""

import sys
from core.main_window import MainWindow
from qtpy.QtWidgets import QApplication
import logging

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    window = MainWindow()
    window.show()
    window.setFocus()

    sys.exit(app.exec())
