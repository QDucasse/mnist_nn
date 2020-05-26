# -*- coding: utf-8 -*-

# mnist_nn
# author - Quentin Ducasse
# https://github.com/QDucasse
# quentin.ducasse@ensta-bretagne.org

# Trained model loading and inference

import torch
import torchvision
import matplotlib.pyplot as plt

from mnist_nn.networks import Cnv2_FC2


if __name__ == "__main__":
    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('files', train=False, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
        batch_size=1000, shuffle=True)
    examples = enumerate(test_loader)
    batch_idx, (example_data, example_targets) = next(examples)
    # Load the trained network
    network = Cnv2_FC2()
    network.load_state_dict(torch.load('results/model.pth'))
    network.eval()

    with torch.no_grad():
        output = network(example_data)

    fig = plt.figure()
    for i in range(6):
        plt.subplot(2,3,i+1)
        plt.tight_layout()
        plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
        plt.title("Prediction: {}".format(
        output.data.max(1, keepdim=True)[1][i].item()))
        plt.xticks([])
        plt.yticks([])
    plt.show()
