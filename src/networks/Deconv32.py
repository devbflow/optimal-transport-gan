import torch
import torch.nn as nn
import torch.nn.functional as F

class Deconv32(nn.Module):
    def __init__(self, name: str, lr: float, channels: int = 1):
        self.name = name
        self.lr = lr
        self.channels = channels

        # flattened -> 4x4
        self.layer0 = nn.Sequential(
                        nn.Linear(1, 4096), # (256*4*4) = 4096
                        nn.Unflatten(1, (256, 4, 4))
                    )
        # 4x4 -> 8x8
        self.layer1 = nn.Sequential(
                        nn.ConvTranspose2d(256, 128, kernel_size=5, stride=2, padding=2, output_padding=1),
                        nn.ReLU()
                    )
        # 8x8 -> 16x16
        self.layer2 = nn.Sequential(
                        nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2, padding=2, output_padding=1),
                        nn.ReLU()
                    )
        # 16x16 -> 32x32
        self.layer3 = nn.ConvTranspose2d(64, self.channels, kernel_size=5, stride=2, padding=2, output_padding=1)


    def forward(self, X):
        output = self.layer0(X)
        output = self.layer1(output)
        output = self.layer2(output)
        output = self.layer3(output)
        output = F.tanh(torch.reshape(output, (-1, self.channels*32*32)))
        return output
