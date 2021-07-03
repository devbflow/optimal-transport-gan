from abc import ABC, abstractmethod

import torch
import numpy as np

class Latent(ABC):
    """
    Abstract class for latent spaces.
    Subclasses only need to implement the sample() method.
    """

    @abstractmethod
    def __init__(self):
        """
        :param shape: The shape of a *single* sample.
        :param batch_size: The default batch size.
        """
        self.shape = None
        self.batch_size = None

    @abstractmethod
    def sample(self):
        """
        Returns several samples from the latent space.
        """
        pass


class GaussianLatent(Latent):

    def __init__(self, shape=None, batch_size=None):
        super().__init__()
        self.batch_size = batch_size
        self.shape = shape
        self.name = "GaussianLatent"

    def sample(self, batch_size):
        return torch.randn((self.batch_size, self.shape))



class MultiGaussianLatent(Latent):

    def __init__(self, shape=None, batch_size=None, N=10000, sigma=0.1):
        super().__init__()
        self.batch_size = batch_size
        self.shape = shape
        self.name = "MultiGaussianLatent"
        self.initial_points = torch.randn((N, self.shape))
        self.sigma = sigma

    def sample(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        samples = np.random.default_rng().choice(self.initial_points, batch_size)
        samples = torch.Tensor(samples)
        return samples + self.sigma * torch.randn((self.batch_size, self.shape))

