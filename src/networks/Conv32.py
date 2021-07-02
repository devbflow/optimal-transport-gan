import torch
import torch.nn as nn


class Conv32(nn.Module):
    def __init__(self, name: str, lr: float, channels=1):
        self.name = name
        self.lr = lr
        self.channels = channels
        self.shape = (32, 32) # shape of one data point, without channels

        # 32x32 -> 16x16
        self.layer0 = nn.Sequential(
                        nn.Conv2d(self.channels, 64, kernel_size=5, stride=2, padding=2),
                        nn.LayerNorm(self.channels),
                        nn.LeakyReLU()
                    )

        # 16x16 -> 8x8
        self.layer1 = nn.Sequential(
                        nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
                        nn.LayerNorm(self.channels),
                        nn.LeakyReLU
                    )

        # 8x8 -> 4x4
        self.layer2 = nn.Sequential(
                        nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2),
                        nn.LayerNorm(self.channels),
                        nn.LeakyReLU()
                    )
        # flatten and dense
        self.layer3 = nn.Sequential(
                        nn.Flatten(),
                        nn.Linear(4096, 1) # (256*4*4) = 4096, from layer2 output
                    )

    def forward(self, X: torch.Tensor):
        assert X.shape[1:] == (self.channels, *self.shape), "input shape not (N,C,H,W)"
        output = self.layer0(X)
        output = self.layer1(output)
        output = self.layer2(output)
        output = self.layer3(output)
        return output
