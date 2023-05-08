import torch
from torch import nn


class Embedder(nn.Module):
    def __init__(self, outputDimension):
        super().__init__()

        self.outputDimension = outputDimension

        # Structure of the net
        self.net = nn.Sequential(nn.Conv2d(3, 32, (3, 3), stride=1, padding=1), nn.ReLU(),
                                 nn.Conv2d(32, 32, (3, 3), stride=1, padding=1), nn.ReLU(),
                                 nn.MaxPool2d((2, 2), stride=2),
                                 nn.Conv2d(32, 32, (3, 3), stride=1, padding=1), nn.ReLU(),
                                 nn.Conv2d(32, 32, (3, 3), stride=1, padding=1), nn.ReLU(),
                                 nn.MaxPool2d((2, 2), stride=2),
                                 nn.Conv2d(32, 64, (3, 3), stride=1, padding=1), nn.ReLU(),
                                 nn.Conv2d(64, 64, (3, 3), stride=1, padding=1), nn.ReLU(),
                                 nn.MaxPool2d((2, 2), stride=2),
                                 nn.Conv2d(64, 128, (3, 3), stride=1, padding=1), nn.ReLU(),
                                 nn.Conv2d(128, 128, (3, 3), stride=1, padding=1), nn.ReLU(),
                                 nn.MaxPool2d((2, 2), stride=2),
                                 nn.Flatten(), nn.Linear(128 * 6 * 6, self.outputDimension)
                                 )

        # we can also try average pooling as last layer

    def forward(self, x):

        out = self.net(x)

        return out

