from PyQt5.QtWidgets import QMainWindow, QLabel, QPushButton, QVBoxLayout, QWidget, QFileDialog
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
from PyQt5.QtGui import QPixmap, QImage

from NewModel import GarbageClassifier

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
data_dir = 'C:/Users/Ariel/Desktop/project/garbage_classification'


class ImageClassificationWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # Load the trained model
        self.model = self.load_model()

        # Create widgets
        self.image_label = QLabel()
        self.result_label = QLabel()
        self.upload_button = QPushButton("Upload Image")

        # Set up the layout
        layout = QVBoxLayout()
        layout.addWidget(self.image_label)
        layout.addWidget(self.upload_button)
        layout.addWidget(self.result_label)

        # Set the central widget and layout
        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

        # Connect the upload button click event
        self.upload_button.clicked.connect(self.upload_image)

    def load_model(self):
        # Define the ResNet34 model architecture
        model = models.resnet34(pretrained=False)
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Linear(num_ftrs, len(GarbageClassifier(data_dir).get_labels()))

        # Load the trained weights
        model.load_state_dict(torch.load('resnet_model.pth', map_location=device))
        model.eval()
        model.to(device=device)

        return model

    def convert_image_to_pixmap(self, image):
        # Convert PIL Image to QPixmap
        image = image.convert("RGB")
        image_data = image.tobytes()
        pixmap = QPixmap.fromImage(QImage(image_data, image.size[0], image.size[1], QImage.Format_RGB888))
        return pixmap

    def predict_bin(self, predicted_class):
        if predicted_class == "biological":
            return 'Brown Bin'
        elif predicted_class == "plastic":
            return 'Orange Bin'
        elif predicted_class == "brown_glass" or predicted_class == "white_glass" or predicted_class == "green_glass":
            return 'Purple Bin'
        elif predicted_class == "metal":
            return 'Grey Bin'
        elif predicted_class == "cardboard":
            return 'Cardboard Bin'
        elif predicted_class == "battery":
            return 'Electronics Waste Bin'
        elif predicted_class == "paper":
            return 'Blue Bin'
        elif predicted_class == "clothes" or predicted_class == "shoes":
            return 'Green Bin (But you can always donate ;) )'
        else:
            return 'Green Bin'

    def upload_image(self):
        # Open file dialog to select an image
        file_dialog = QFileDialog()
        image_path, _ = file_dialog.getOpenFileName(self, "Select Image", "", "Image Files (*.png *.jpg *.jpeg)")

        if image_path:
            # Load and preprocess the image
            image = Image.open(image_path)
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
            ])
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

            # Display the uploaded image in the image label
            pixmap = self.convert_image_to_pixmap(image)
            self.image_label.setPixmap(pixmap)

            # Display the predicted class in the result label
            self.result_label.setText(
                f"Predicted Class: {predicted_class}, The bin it should be thrown to is: {predicted_bin}")

