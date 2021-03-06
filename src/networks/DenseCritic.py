import torch
import torch.nn as nn
import torch.nn.functional as F

class DenseCritic(nn.Module):

    def __init__(self, name, lr, layer_dim, xdim):
        super().__init__()
        # variables
        self.name = name
        self.lr = lr
        self.layer_dim = layer_dim
        self.xdim = xdim

        # architecture
        self.dense0 = nn.Linear(self.xdim, self.layer_dim)
        self.dense1 = nn.Linear(self.layer_dim, self.layer_dim)
        self.dense2 = nn.Linear(self.layer_dim, self.layer_dim)
        self.dense3 = nn.Linear(self.layer_dim, self.layer_dim)
        self.dense4 = nn.Linear(self.layer_dim, 1)

    def forward(self, X):

        X = torch.reshape(X, shape=(-1, self.xdim))
        output = self.dense0(X)
        output = F.leaky_relu(output, negative_slope=0.2)
        # standard PT slope == 0.01, different from TF (0.2), thus we change it here, respecting the TF code

        output = self.dense1(output)
        output = F.leaky_relu(output, negative_slope=0.2)

        output = self.dense2(output)
        output = F.leaky_relu(output, negative_slope=0.2)

        output = self.dense3(output)
        output = F.leaky_relu(output, negative_slope=0.2)

        output = self.dense4(output)

        output = torch.reshape(output, shape=(-1, 1))
        return output
