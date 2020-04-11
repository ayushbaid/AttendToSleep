'''

CNN + transformer acrchitecture: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
'''
import math

import torch

import torch.nn as nn


class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()

        self.num_temporal = 10
        self.num_labels = 5

        self.cnn_left = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=64,
                      kernel_size=50, stride=6, padding=8),
            nn.MaxPool1d(kernel_size=8, stride=8, padding=1),
            nn.ReLU(),
            nn.Dropout(),
            nn.Conv1d(in_channels=64, out_channels=128,
                      kernel_size=8, stride=1, padding=7),
            nn.ReLU(),
            nn.Conv1d(in_channels=128, out_channels=128,
                      kernel_size=8, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv1d(in_channels=128, out_channels=128,
                      kernel_size=8, stride=1, padding=0),
            nn.MaxPool1d(kernel_size=4, stride=4),
            nn.Dropout(),  # moving dropout to the front
        )

        self.cnn_right = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=64,
                      kernel_size=400, stride=50, padding=7),
            nn.MaxPool1d(kernel_size=4, stride=4),
            nn.ReLU(),
            nn.Dropout(),
            nn.Conv1d(in_channels=64, out_channels=128,
                      kernel_size=6, stride=1, padding=5),
            nn.ReLU(),
            nn.Conv1d(in_channels=128, out_channels=128,
                      kernel_size=6, stride=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=128, out_channels=128,
                      kernel_size=6, stride=1),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(),  # moving dropout to the front
        )

        self.fc = nn.Sequential(
            nn.ReLU(),
            nn.Linear(2176, 1024),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(1024, 5)
        )

    def forward(self, x):
        '''
        The input is stacked as a separate batch for each time step
        '''

        # merge the batch and temporal data
        cnn_input = x.view(x.shape[0]*x.shape[1], x.shape[2], x.shape[3])

        # pass in through the 2 CNNs

        # print(x.shape)

        cnn1 = self.cnn_left(cnn_input)
        cnn2 = self.cnn_right(cnn_input)

        # print(cnn1.shape)
        # print(cnn2.shape)

        cnn_output = torch.cat(
            (cnn1.view(cnn_input.shape[0], -1), cnn2.view(cnn_input.shape[0], -1)), axis=1)

        # reshape it according to x
        cnn_output = cnn_output.view(x.shape[0], x.shape[1], -1)

        output = self.fc(cnn_output)
        # print('Transformer output shape: ', output.shape)

        # flip the axis convention to the input convention
        return output.view(x.shape[0], x.shape[1], -1)
