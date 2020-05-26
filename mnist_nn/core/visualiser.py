# -*- coding: utf-8 -*-

# mnist_nn
# author - Quentin Ducasse
# https://github.com/QDucasse
# quentin.ducasse@ensta-bretagne.org

# Dataset exploration and visualisation

import torch
import torchvision
import matplotlib.pyplot as plt


class Visualiser(object):
    def __init__(self,batch_size=1000):
        self.loader = torch.utils.data.DataLoader(
            torchvision.datasets.MNIST('./files', download=True,
                                        transform=torchvision.transforms.Compose([
                                        torchvision.transforms.ToTensor(),
                                        torchvision.transforms.Normalize((0.1307,), (0.3081,))])),
                            batch_size=batch_size, shuffle=True)

    def display_6items(self):
        '''Shows six items along with their labels out of the dataset'''
        examples = enumerate(self.loader)
        batch_idx, (example_data, example_targets) = next(examples)
        fig = plt.figure()
        for i in range(6):
            plt.subplot(2,3,i+1)
            plt.tight_layout()
            plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
            plt.title("Ground Truth: {}".format(example_targets[i]))
            plt.xticks([])
            plt.yticks([])
        plt.show()

# Display 6 images of the dataset

if __name__ == "__main__":
    # Visualiser initialization and image display
    visualiser = Visualiser()
    visualiser.display_6items()
