from PyQt6.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout
import sys

def run_gui():
    app = QApplication(sys.argv)
    w = QWidget(); w.setWindowTitle("DreamGlass — Lumaeth")
    layout = QVBoxLayout()
    layout.addWidget(QLabel("Rooms: Atrium · Observatory · Garden · Mirror Hall · Gallery of Shadows"))
    layout.addWidget(QLabel("Status: Online"))
    w.setLayout(layout); w.show()
    sys.exit(app.exec())
