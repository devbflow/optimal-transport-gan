import torch
import torch.nn as nn
import torch.nn.functional as F

class DenseGenerator(nn.Module):

    def __init__(self, channels, name=None, lr=None, layer_dim=1024, xdim=20):
        # variables
        self.name = name
        self.lr = lr
        self.layer_dim = layer_dim
        self.xdim = xdim

        # architecture
        self.dense0 = nn.LazyLinear(self.layer_dim) # infer in_features
        self.ln0 = nn.LayerNorm(self.layer_dim)

        self.dense1 = nn.Linear(self.layer_dim, self.layer_dim)
        self.ln1 = nn.LayerNorm(self.layer_dim)

        self.dense2 = nn.Linear(self.layer_dim, self.layer_dim)
        self.ln2 = nn.LayerNorm(self.layer_dim, self.layer_dim)

        self.dense3 = nn.Linear(self.layer_dim, self.xdim)

    def forward(self, X):
        output = self.dense0(X)
        output = F.leaky_relu(output)
        output = self.ln0(output)

        output = self.dense1(output)
        output = F.leaky_relu(output)
        output = self.ln1(output)

        output = self.dense2(output)
        output = F.leaky_relu(output)
        output = self.ln2(output)

        output = self.dense3(output)
        output = F.tanh(output)
        return output
