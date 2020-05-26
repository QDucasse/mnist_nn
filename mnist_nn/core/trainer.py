# -*- coding: utf-8 -*-

# mnist_nn
# author - Quentin Ducasse
# https://github.com/QDucasse
# quentin.ducasse@ensta-bretagne.org

# Training routine for a network

import torch
import torchvision
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

from mnist_nn.networks import Cnv2_FC2


class Trainer(object):
    def __init__(self, network, optimizer, n_epochs, learning_rate, momentum,
                 batch_size_test, batch_size_train, log_interval):
        # Number of epochs
        self.n_epochs         = n_epochs
        # Batch arguments for both train and test
        self.batch_size_train = batch_size_train
        self.batch_size_test  = batch_size_test
        # Learning hyperparameters
        self.learning_rate    = learning_rate
        self.momentum         = momentum
        # Info presentation interval
        self.log_interval     = log_interval
        # Network and optimizer definition
        self.network   = network
        self.optimizer = optimizer
        # Dataset holders
        self.train_loader = None
        self.test_loader  = None
        # Losses and counter holders for test and train routines
        self.train_losses  = []
        self.train_counter = []
        self.test_losses   = []
        self.test_counter  = []
        # Initialize seed
        self.initialize_randomness()
        # Load dataset
        self.load_train_test()

    def initialize_randomness(self):
        '''Set the seed for the randomness, disable cudnn backend'''
        random_seed = 1
        torch.backends.cudnn.enabled = False
        torch.manual_seed(random_seed)


    def load_train_test(self):
        '''Loads the train/test dataset in their corresponding DataLoaders. Normalize
            the inputs and shuffles them.'''
        self.train_loader = torch.utils.data.DataLoader(
                torchvision.datasets.MNIST('./files', train=True, download=True,
                                     transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize(
                                         (0.1307,), (0.3081,))
                                     ])),
                batch_size=self.batch_size_train,
                shuffle=True)

        self.test_loader = torch.utils.data.DataLoader(
                torchvision.datasets.MNIST('./files', train=False, download=True,
                                     transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize(
                                         (0.1307,), (0.3081,))
                                     ])),
                batch_size=self.batch_size_test,
                shuffle=True)

        self.test_counter  = [i*len(self.train_loader.dataset) for i in range(self.n_epochs + 1)]


    def train(self,epoch):
        '''Launches the training routine that consists of:
            - Initialization of the info holders
            - Actual training:
                - Call the nn.train() function
                - Perform the backpropagation with optimizer and loss function
                - Print '''
        # Aliases to lighten the code
        network       = self.network
        optimizer     = self.optimizer
        train_loader  = self.train_loader
        train_losses  = self.train_losses
        train_counter = self.train_counter
        log_interval  = self.log_interval

        # First call to the torch.nn.train() (and therefore forward() defined in the network)
        # Forward propagation
        network.train()

        # Backpropagation
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = network(data) # Output of the network given the input data

            # Loss function definition
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % log_interval == 0:
                  print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                            epoch, batch_idx * len(data), len(train_loader.dataset),
                            100. * batch_idx / len(train_loader), loss.item()))
                  train_losses.append(loss.item())
                  train_counter.append((batch_idx*64) + ((epoch-1)*len(train_loader.dataset)))
                  # Save the state_dict of the network and optimizer if the training has to stop so it can be resumed
                  torch.save(network.state_dict(), 'results/model.pth')
                  torch.save(optimizer.state_dict(), 'results/optimizer.pth')


    def test(self):
        '''Launches the training routine that consists of:
            - Initialization of the info holders
            - Actual training:
                - Call the nn.train() function
                - Perform the backpropagation with optimizer and loss function
                - Print '''
        # Aliases to lighten the code
        network       = self.network
        optimizer     = self.optimizer
        test_loader   = self.test_loader
        test_losses   = self.test_losses
        test_counter  = self.test_counter

        # Evaluation with the toch.nn function
        network.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                output = network(data) # Actual output of the network given the input data
                test_loss += F.nll_loss(output, target, size_average=False).item()
                pred = output.data.max(1, keepdim=True)[1]
                correct += pred.eq(target.data.view_as(pred)).sum()
                test_loss /= len(test_loader.dataset)
            test_losses.append(test_loss)
            print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                test_loss, correct, len(test_loader.dataset),
                100. * correct / len(test_loader.dataset)))

    def start_train_test(self):
        # Training Routine
        self.test()
        for epoch in range(1, self.n_epochs + 1):
            # Train and test for each epoch
            self.train(epoch)
            self.test()
        self.plot_results()


    def plot_results(self):
        fig = plt.figure()
        plt.plot(self.train_counter, self.train_losses, color='blue')
        plt.scatter(self.test_counter, self.test_losses, color='red')
        plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
        plt.xlabel('number of training examples seen')
        plt.ylabel('negative log likelihood loss')
        plt.show()



if __name__ == "__main__":
    # Learning hyperparameters
    learning_rate = 0.01
    momentum = 0.5
    # Network and Optimizer initialization
    network = Cnv2_FC2()
    optimizer = optim.SGD(network.parameters(), lr=learning_rate,
                                                momentum=momentum)
    # Trainer initialization
    trainer = Trainer(network          = network,
                      optimizer        = optimizer,
                      n_epochs         = 3,
                      batch_size_train = 64,
                      batch_size_test  = 1000,
                      learning_rate    = learning_rate,
                      momentum         = momentum,
                      log_interval     = 10)

    # Launch training routine
    trainer.start_train_test()
