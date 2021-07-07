import torch
import torchvision.transforms as T
from torchvision.datasets import MNIST


class MNIST32(MNIST):
    """
    Wrapper class for slightly modified MNIST dataset.
    """
    def __init__(self, *args, **kwargs):
        transform = T.Compose([
            T.Resize((32,32)),
            T.ToTensor()
            ])
        super(MNIST32, self).__init__(transform=transform, *args, **kwargs)
        self.data = ((self.data / 255) - 0.5) / 0.5
        #self.data = torch.unsqueeze(self.data, dim=1)
        self.name = "MNIST32"
        self.data_shape = (1,32,32)
