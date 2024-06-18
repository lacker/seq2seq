import torch
import torch.nn as nn


class MLP(nn.Module):
    """
    Multi-layer perceptron. Has num_features for both input and output.
    No dropout but maybe we could stick some at the end.
    """
    def __init__(self, num_features):
        super().__init__()
        self.expansion    = nn.Linear(num_features, 4 * num_features)
        self.gelu    = nn.GELU()
        self.projection  = nn.Linear(4 * num_features, num_features)

    def forward(self, x):
        x = self.expansion(x)
        x = self.gelu(x)
        x = self.projection(x)
        return x