'''

CNN + transformer acrchitecture: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
'''
import math

import torch

import torch.nn as nn


class CNNSeq2SeqModel(nn.Module):
    def __init__(self, num_temporal, cnn_weights=None):
        super(CNNSeq2SeqModel, self).__init__()

        self.num_temporal = num_temporal
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

        if cnn_weights is not None:
            print('[Log] Loading CNN model weights from pretraining')

            self.cnn_left.load_state_dict(cnn_weights['model_cnnLeft'])
            self.cnn_right.load_state_dict(cnn_weights['model_cnnRight'])

            self.freeze_cnn()

        self.fc_layers = nn.Sequential(
            nn.ReLU(),
            nn.Linear(2176, 512),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(512, 256),
        )

        self.transformer = TransformerModel(
            ntoken=self.num_labels, ninp=256, nhead=4, nhid=256, nlayers=3, dropout=0.05
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

        # pass through the fc layers
        cnn_output = self.fc_layers(cnn_output)

        # reshape it according to x
        cnn_output = cnn_output.view(x.shape[0], x.shape[1], -1)

        # switch batch axis to second
        cnn_output = torch.transpose(cnn_output, 1, 0)

        # convert the CNN output to temporal inputs, with time being the first dimenstion and batch being the second)
        # cnn_concat = cnn_concat.view(x.)

        # pass the data through the transformer
        # print('Transformer input shape: ', cnn_output.shape)
        output = self.transformer(cnn_output)
        # print('Transformer output shape: ', output.shape)

        # flip the axis convention to the input convention
        return torch.transpose(output, 1, 0)

    def freeze_cnn(self):
        self.cnn_left[0].weight.requires_grad = False
        self.cnn_left[0].bias.requires_grad = False

        self.cnn_left[4].weight.requires_grad = False
        self.cnn_left[4].bias.requires_grad = False

        self.cnn_left[6].weight.requires_grad = False
        self.cnn_left[6].bias.requires_grad = False

        self.cnn_left[8].weight.requires_grad = False
        self.cnn_left[8].bias.requires_grad = False

        # take care to turn off gradients for both weight and bias
        self.cnn_right[0].weight.requires_grad = False
        self.cnn_right[0].bias.requires_grad = False

        self.cnn_right[4].weight.requires_grad = False
        self.cnn_right[4].bias.requires_grad = False

        self.cnn_right[6].weight.requires_grad = False
        self.cnn_right[6].bias.requires_grad = False

        self.cnn_right[8].weight.requires_grad = False
        self.cnn_right[8].bias.requires_grad = False

    def unfreeze_cnn(self):
        self.cnn_left[0].weight.requires_grad = True
        self.cnn_left[0].bias.requires_grad = True

        self.cnn_left[4].weight.requires_grad = True
        self.cnn_left[4].bias.requires_grad = True

        self.cnn_left[6].weight.requires_grad = True
        self.cnn_left[6].bias.requires_grad = True

        self.cnn_left[8].weight.requires_grad = True
        self.cnn_left[8].bias.requires_grad = True

        # take care to turn off gradients for both weight and bias
        self.cnn_right[0].weight.requires_grad = True
        self.cnn_right[0].bias.requires_grad = True

        self.cnn_right[4].weight.requires_grad = True
        self.cnn_right[4].bias.requires_grad = True

        self.cnn_right[6].weight.requires_grad = True
        self.cnn_right[6].bias.requires_grad = True

        self.cnn_right[8].weight.requires_grad = True
        self.cnn_right[8].bias.requires_grad = True


class TransformerModel(nn.Module):

    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = nn.TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers, nlayers)
        # self.encoder = nn.Embedding(ntoken, ninp) # moved this to a linear layer after the cnn concat
        self.ninp = ninp
        self.decoder = nn.Linear(ninp, ntoken)

        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float(
            '-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        # self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        # src = self.encoder(src) * math.sqrt(self.ninp)
        src = src * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        # output = self.transformer_encoder(src, self.src_mask) # with mask
        output = self.transformer_encoder(src)  # no mask
        output = self.decoder(output)
        return output


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
