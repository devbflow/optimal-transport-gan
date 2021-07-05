import matplotlib
import numpy as np
import torch
from torchvision import datasets

from data.DatasetABC import DatasetABC

matplotlib.use('Agg')
import scipy.misc


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
        self.shape = (32, 32, 1)
        data, labels = self.generate_data()
        # Initialize class.
        self.name = "Mnist"
        self.data = data
        self.labels = labels
        self.dataset_size = dataset_size
        self.batch_size = batch_size

    @staticmethod
    def generate_data():
        shape = (32, 32, 1)
        mnist = datasets.MNIST(root='./data', download=True)
        data = mnist.data
        label = mnist.targets
        imgs_32 = [scipy.misc.imresize(data[idx], shape)
                   for idx in range(data.shape[0])]
        data = np.expand_dims(np.asarray(imgs_32).astype(np.float32), axis=3)
        data = data / 255
        data = (data - 0.5) / 0.5
        data = data.reshape(data.shape[0], -1)
        return data, label
