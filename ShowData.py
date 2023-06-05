import torch
import matplotlib.pyplot as plt
import numpy as np
import random

from NewModel import GarbageClassifier


class DataVisualization:
    def __init__(self):
        """
        Class for visualizing data samples.

        Initializes the device, the garbage classifier, and the labels list.
        """
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.is_classifier = GarbageClassifier('C:/Users/Ariel/Desktop/project/garbage_classification')
        self.labels_list = self.is_classifier.get_labels()

    def visualize_data(self):
        """
        Visualize a subset of data samples.

        Retrieves a subset of training data, applies normalization, and displays the images.
        """
        train_dataloader, _ = self.is_classifier.prepare_data()

        # Shuffle the dataset
        dataset = train_dataloader.dataset
        indices = list(range(len(dataset)))
        random.shuffle(indices)

        mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
        std = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))

        figure = plt.figure(figsize=(8, 8))
        cols, rows = 5, 3
        for i in range(1, cols * rows + 1):
            sample_idx = indices[i]
            img, label = train_dataloader.dataset[sample_idx]
            img = img.numpy() * std + mean
            img = np.transpose(img, (1, 2, 0))  # Transpose the tensor to (224, 224, 3)
            figure.add_subplot(rows, cols, i)
            plt.title(self.labels_list[label])
            plt.axis("off")
            plt.imshow(img)
        plt.show()

