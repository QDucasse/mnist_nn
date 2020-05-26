# -*- coding: utf-8 -*-

# mnist_nn
# author - Quentin Ducasse
# https://github.com/QDucasse
# quentin.ducasse@ensta-bretagne.org

# MLP network: Three Fully Connected layers

import torch.nn as nn
import torch.nn.functional as F

class MLPNet(nn.Module):
    '''This neural network consists of the following flow:
        -                             Input:  [1000,  1, 28, 28]
        - Layer    | FullyConnected | Output: [1000, 20, 8, 8]
        - Function | ReLU           | Output: [1000, 10, 12, 12]
        - Layer    | FullyConnected | Output: [1000, 50]
        - Function | ReLU           | Output: [1000, 10, 12, 12]
        - Layer    | FullyConnected | Output: [1000, 50]'''

    def __init__(self):
        super(MLPNet, self).__init__()
        self.fc1 = nn.Linear(28*28,128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
        self.name = "MLP"

    def forward(self, x):
        x = x.view(-1, 28*28)
        # Layer 1
        x = F.relu(self.fc1(x))
        # Layer 2
        x = F.relu(self.fc2(x))
        # Layer 3
        x = self.fc3(x)
        return F.log_softmax(x,dim=1)
