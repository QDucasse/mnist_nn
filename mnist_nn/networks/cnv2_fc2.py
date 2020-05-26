# -*- coding: utf-8 -*-

# mnist_nn
# author - Quentin Ducasse
# https://github.com/QDucasse
# quentin.ducasse@ensta-bretagne.org

# Two Conv-2D followed by two Fully-Connected

import torch.nn as nn
import torch.nn.functional as F

class Cnv2_FC2(nn.Module):
    '''This neural network consists of the following flow:
        - Layer    - Conv2D:
        - Layer    - MaxPool2D:
        - Function - ReLU:
        - Layer    - Conv2D:
        - Layer    - Dropout2D:
        - Layer    - MaxPool2D:
        - Function - ReLU:
        - Layer    - FullyConnected:
        - Function - Dropout:
        - Layer    - FullyConnected:
        - Function - LogSoftMax:'''
        
    def __init__(self):
        super(Cnv2_FC2, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)