import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
import torchvision.models as models
import random
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GarbageClassifier():
    def __init__(self, data_dir, batch_size=32, num_epochs=10, learning_rate=0.001):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.class_labels = ['battery', 'biological', 'brown_glass', 'cardboard', 'clothes', 'green_glass', 'metal',
                             'paper', 'plastic', 'shoes', 'trash', 'white_glass']

    def get_labels(self):
        return self.class_labels

    def prepare_data(self):
        # Define transforms for data preprocessing
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

        # Define the path to your dataset directory
        dataset_dir = self.data_dir

        # Load all the images in the dataset
        dataset = ImageFolder(dataset_dir, transform=transform)

        # Shuffle the dataset indices
        random.seed(42)
        indices = list(range(len(dataset)))
        random.shuffle(indices)

        # Split the indices into train and test sets
        train_size = int(0.8 * len(dataset))
        train_indices = indices[:train_size]
        test_indices = indices[train_size:]

        # Create samplers for train and test sets
        train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indices)
        test_sampler = torch.utils.data.sampler.SubsetRandomSampler(test_indices)

        # Create dataloaders for training and testing
        train_dataloader = DataLoader(dataset, batch_size=self.batch_size, sampler=train_sampler)
        test_dataloader = DataLoader(dataset, batch_size=self.batch_size, sampler=test_sampler)

        return train_dataloader, test_dataloader

    def build_model(self):
        # Define the ResNet34 model architecture
        model = models.resnet34(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, len(self.class_labels))
        model.to(device=device)

        # Freeze early layers for fine-tuning
        for param in model.parameters():
            param.requires_grad = False

        # Enable gradient calculation for the last few layers
        for param in model.fc.parameters():
            param.requires_grad = True

        return model

    @staticmethod
    def get_accuracy(outputs, labels):
        _, predicted = torch.max(outputs, dim=1)
        correct = (predicted == labels).sum().item()
        accuracy = correct / labels.size(0)
        return accuracy

    def train(self):
        train_dataloader, test_dataloader = self.prepare_data()

        model = self.build_model()

        # Define the loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.fc.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

        train_losses = []
        train_accuracies = []

        # Create the figure and subplots outside the training loop
        figure = plt.figure(figsize=(10, 5))
        loss_subplot = figure.add_subplot(1, 2, 1)
        accuracy_subplot = figure.add_subplot(1, 2, 2)

        for epoch in range(self.num_epochs):
            running_loss = 0.0
            epoch_accuracy = 0.0

            # Training phase
            model.train()
            for images, labels in train_dataloader:
                images, labels = images.to(device=device), labels.to(device=device)
                optimizer.zero_grad()

                # Forward pass
                outputs = model(images)
                loss = criterion(outputs, labels)

                # Backward pass and optimization
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                epoch_accuracy += self.get_accuracy(outputs, labels)

            # Calculate average loss and accuracy for the epoch
            epoch_loss = running_loss / len(train_dataloader)
            epoch_accuracy = epoch_accuracy / len(train_dataloader)

            train_losses.append(epoch_loss)
            train_accuracies.append(epoch_accuracy)

            # Update learning rate
            scheduler.step()

        # Plot loss and accuracy outside the training loop
        loss_subplot.plot(range(epoch + 1), train_losses)
        loss_subplot.set_xlabel('Epoch')
        loss_subplot.set_ylabel('Loss')
        loss_subplot.set_title('Training Loss')

        accuracy_subplot.plot(range(epoch + 1), train_accuracies)
        accuracy_subplot.set_xlabel('Epoch')
        accuracy_subplot.set_ylabel('Accuracy')
        accuracy_subplot.set_title('Training Accuracy')

        plt.tight_layout()
        plt.show()

        # Evaluation on the test set
        model.eval()
        test_accuracy = 0.0

        # Iterate over the test dataset
        with torch.no_grad():
            for images, labels in test_dataloader:
                images, labels = images.to(device=device), labels.to(device=device)
                test_outputs = model(images)
                test_accuracy += self.get_accuracy(test_outputs, labels)

            # Calculate the average test accuracy
            test_accuracy = test_accuracy / len(test_dataloader)

            # Print the test accuracy
            print(f'Test Accuracy: {test_accuracy * 100:.2f}%')

            # Save the trained model
            torch.save(model.state_dict(), 'resnet_model.pth')


if __name__ == "__main__":
    # Set the path to your dataset directory
    data_dir = r'C:\Users\Ariel\Desktop\project\garbage_classification'

    # Create an instance of the GarbageClassifier class
    classifier = GarbageClassifier(data_dir)

    # Start the training process
    classifier.train()
