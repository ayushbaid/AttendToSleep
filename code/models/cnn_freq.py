'''

CNN + transformer acrchitecture: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
'''
import math

import torch

import torch.nn as nn

from torchaudio.transforms import Spectrogram

from torchvision.models import alexnet


class CNNFreqModel(nn.Module):
    def __init__(self):
        super(CNNFreqModel, self).__init__()

        self.num_temporal = 10
        self.num_labels = 5

        self.eeg_to_freq = Spectrogram(n_fft=128)

        # self.cnn = alexnet(pretrained=False).features[:11]
        # replace first layer and add batchnorm after first layer
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=11, stride=4, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0,
                         dilation=1, ceil_mode=False),
            nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0,
                         dilation=1, ceil_mode=False),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

        # print(self.cnn)

        self.fc = nn.Sequential(
            nn.Dropout(),
            nn.Linear(384, 50),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(50, 5)
        )

    def forward(self, x):
        '''
        The input is stacked as a separate batch for each time step
        '''

        # merge the batch and temporal data
        cnn_input = x.view(x.shape[0]*x.shape[1], x.shape[2], x.shape[3])

        # pass in through the 2 CNNs

        # print(x.shape)

        # cnn1 = self.cnn_left(cnn_input)
        # cnn2 = self.cnn_right(cnn_input)

        # print(cnn1.shape)
        # print(cnn2.shape)

        cnn_output = self.cnn(self.eeg_to_freq(cnn_input))

        # reshape it according to x
        cnn_output = cnn_output.view(x.shape[0], x.shape[1], -1)

        output = self.fc(cnn_output)
        # print('Transformer output shape: ', output.shape)

        # flip the axis convention to the input convention
        return output.view(x.shape[0], x.shape[1], -1)
