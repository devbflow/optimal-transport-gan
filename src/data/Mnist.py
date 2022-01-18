import matplotlib
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms


from data.DatasetABC import DatasetABC

matplotlib.use('Agg')



class Mnist32(DatasetABC):
    """
    Class for the upscaled 32x32 MNIST dataset.
    """

    def __init__(self, batch_size, dataset_size):
        """
        :param batch_size: The default batch size.
        """
        # Load data.
        super().__init__()
        self.shape = (1, 32, 32)
        data, labels = self.generate_data()
        # Initialize class.
        self.name = "Mnist"
        self.data = data
        self.labels = labels
        self.dataset_size = dataset_size
        self.batch_size = batch_size
        self.dataset_iterator = self.get_iterator()

    @staticmethod
    def generate_data():
        transform = transforms.Compose([
            transforms.Resize((32,32)),
            transforms.ToTensor(),

            ])
        self.mnist = datasets.MNIST(root='./data', download=True, transform=transform)
        data = self.mnist.data
        label = self.mnist.targets
        #data = data.reshape(data.shape[0], -1)
        return data, label

    def get_iterator(self):
        """
        Returns DataLoader for the dataset.
        """
        return DataLoader(self.mnist, batch_size=self.batch_size, shuffle=True)

    def __getitem__(self, index):
        data, label = self.dataset_iterator
