from PyQt5.QtWidgets import QMainWindow, QLabel, QPushButton, QVBoxLayout, QWidget
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import cv2
from PIL import Image
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import QTimer

from NewModel import GarbageClassifier

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
data_dir = 'C:/Users/Ariel/Desktop/project/garbage_classification'


class VideoCaptureWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # Load the trained model
        self.model = self.load_model()

        # Create widgets
        self.video_label = QLabel()
        self.result_label = QLabel()
        self.capture_button = QPushButton("Capture")

        # Set up the layout
        layout = QVBoxLayout()
        layout.addWidget(self.video_label)
        layout.addWidget(self.capture_button)
        layout.addWidget(self.result_label)

        # Set the central widget and layout
        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

        # Connect the capture button click event
        self.capture_button.clicked.connect(self.capture_and_predict)

        # Start the video capture
        self.capture = cv2.VideoCapture(0)
        self.start_video_capture()

    def load_model(self):
        """
        Load the trained model.

        Returns:
            The loaded model.
        """
        # Define the ResNet34 model architecture
        model = models.resnet34(pretrained=False)
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Linear(num_ftrs, len(GarbageClassifier(data_dir).get_labels()))

        # Load the trained weights
        model.load_state_dict(torch.load('resnet_model.pth', map_location=device))
        model.eval()
        model.to(device=device)

        return model

    def start_video_capture(self):
        # Create a timer to update the video frame
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_video_frame)
        self.timer.start(30)  # Update every 30 milliseconds (33 fps)

    def update_video_frame(self):
        # Capture frame-by-frame
        ret, frame = self.capture.read()
        if not ret:
            print("ERROR: Failed to capture frame")
            return

        # Convert the captured frame to a PIL Image object
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Resize the image to fit the video_label dimensions
        image = image.resize((640, 480))

        # Convert the PIL Image to a QPixmap object
        pixmap = self.convert_image_to_pixmap(image)

        # Set the pixmap as the image for the video_label
        self.video_label.setPixmap(pixmap)

    def convert_image_to_pixmap(self, image):
        """
        Convert a PIL Image to QPixmap.

        Args:
            image: The PIL Image to convert.

        Returns:
            The converted QPixmap object.
        """
        image = image.convert("RGB")
        image_data = image.tobytes()
        pixmap = QPixmap.fromImage(QImage(image_data, image.size[0], image.size[1], QImage.Format_RGB888))
        return pixmap

    def predict_bin(self, predicted_class):
        """
        Predict the bin based on the predicted class.

        Args:
            predicted_class: The predicted class.

        Returns:
            The bin the item should be thrown into.
        """
        if predicted_class == "biological":
            return 'Brown Bin'
        elif predicted_class == "plastic":
            return 'Orange Bin'
        elif predicted_class == "brown_glass" or predicted_class == "white_glass" or predicted_class == "green_glass":
            return 'Purple Bin'
        elif predicted_class == "metal":
            return 'Grey Bin'
        elif predicted_class == "cardboard":
            return "Cardboard Bin"
        elif predicted_class == "battery":
            return 'Electronics Waste Bin'
        elif predicted_class == "paper":
            return 'Blue Bin'
        elif predicted_class == "clothes" or predicted_class == "shoes":
            return 'Green Bin (But you can always donate ;) )'
        else:
            return 'Green Bin'

    def capture_and_predict(self):
        # Capture a frame from the live video
        ret, frame = self.capture.read()
        if not ret:
            print("ERROR: Failed to capture frame")
            return

        # Save the captured frame as an image
        cv2.imwrite('captured_frame.jpg', frame)

        # Preprocess the captured image
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])
        image = Image.open('captured_frame.jpg')
        preprocessed_image = transform(image)

        # Add batch dimension
        preprocessed_image = preprocessed_image.unsqueeze(0)

        # Move the preprocessed image to the appropriate device
        preprocessed_image = preprocessed_image.to(device=device)

        # Perform inference using the model
        with torch.no_grad():
            output = self.model(preprocessed_image)

        # Get the predicted class index
        _, predicted_idx = torch.max(output, dim=1)
        predicted_class = GarbageClassifier(data_dir).get_labels()[predicted_idx.item()]
        predicted_bin = self.predict_bin(predicted_class)

        # Display the predicted class in the result label
        self.result_label.setText(
            f"Predicted Class: {predicted_class}, The bin it should be thrown to is: {predicted_bin}")
