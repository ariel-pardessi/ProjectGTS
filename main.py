import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QLabel
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt

from ShowData import DataVisualization
from capture import VideoCaptureWindow
from uploaded import ImageClassificationWindow


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Garbage Classification")
        self.setGeometry(100, 100, 400, 300)

        title_label = QLabel("Garbage Classification")
        title_label.setAlignment(Qt.AlignCenter)
        title_font = QFont("Arial", 16, QFont.Bold)
        title_label.setFont(title_font)
        title_label.setText("<u>Garbage Classification</u>")

        smaller_title_label = QLabel("By: Ariel Pardessi")
        smaller_title_label.setAlignment(Qt.AlignCenter)
        smaller_title_font = QFont("Arial", 12, QFont.Bold)
        smaller_title_label.setFont(smaller_title_font)

        data_visualization_button = QPushButton("Data Visualization")
        video_capture_button = QPushButton("Video Capture")
        photo_upload_button = QPushButton("Upload Photo")

        button_font = QFont("Arial", 12)
        data_visualization_button.setFont(button_font)
        video_capture_button.setFont(button_font)
        photo_upload_button.setFont(button_font)

        layout = QVBoxLayout()
        layout.addWidget(title_label)
        layout.addWidget(smaller_title_label)
        layout.addWidget(data_visualization_button)
        layout.addWidget(video_capture_button)
        layout.addWidget(photo_upload_button)
        layout.setSpacing(20)

        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

        self.data_visualization_window = DataVisualization()
        self.video_capture_window = VideoCaptureWindow()
        self.uploaded_photo_window = ImageClassificationWindow()

        data_visualization_button.clicked.connect(self.data_visualization_window.visualize_data)
        video_capture_button.clicked.connect(self.video_capture_window.show)
        photo_upload_button.clicked.connect(self.show_image_classification_window)

    def show_image_classification_window(self):
        self.uploaded_photo_window.show()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())
