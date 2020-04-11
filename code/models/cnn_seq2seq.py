'''

CNN + transformer acrchitecture: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
'''
import math

import torch

import torch.nn as nn


class CNNSeq2SeqModel(nn.Module):
    def __init__(self, num_hidden=128, num_rnn_layers=2, num_temporal=10):
        super(CNNSeq2SeqModel, self).__init__()

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

        self.transformer = TransformerModel(
            ntoken=self.num_labels, ninp=2176, nhead=1, nhid=128, nlayers=1)

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


class TransformerModel(nn.Module):

    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = nn.TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers, nlayers)
        # self.encoder = nn.Embedding(ntoken, ninp)
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
        output = self.transformer_encoder(src, self.src_mask)
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


# class EncoderRNN(nn.Module):
#     def __init__(self, input_size, hidden_size, device, num_layers):
#         super(EncoderRNN, self).__init__()
#         self.hidden_size = hidden_size
#         self.device = device

#         self.LSTM = nn.LSTM(input_size=100,
#                             hidden_size=hidden_size,
#                             num_layers=num_layers,
#                             bidirectional=True,
#                             batch_first=True
#                             )

#     def forward(self, input, hidden):
#         embedded = self.embedding(input).view(1, 1, -1)
#         output = embedded
#         output, hidden = self.gru(output, hidden)
#         return output, hidden

#     def initHidden(self):
#         return torch.zeros(1, 1, self.hidden_size, device=self.device)


# class AttnDecoderRNN(nn.Module):
#     def __init__(self, hidden_size, device, num_layers, output_size, dropout_p=0.1, max_length=MAX_LENGTH):
#         super(AttnDecoderRNN, self).__init__()
#         self.hidden_size = hidden_size
#         self.output_size = output_size
#         self.dropout_p = dropout_p
#         self.max_length = max_length

#         self.embedding = nn.Embedding(self.output_size, self.hidden_size)
#         self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
#         self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
#         self.dropout = nn.Dropout(self.dropout_p)
#         self.gru = nn.GRU(self.hidden_size, self.hidden_size)
#         self.out = nn.Linear(self.hidden_size, self.output_size)

#     def forward(self, input, hidden, encoder_outputs):
#         embedded = self.embedding(input).view(1, 1, -1)
#         embedded = self.dropout(embedded)

#         attn_weights = F.softmax(
#             self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
#         attn_applied = torch.bmm(attn_weights.unsqueeze(0),
#                                  encoder_outputs.unsqueeze(0))

#         output = torch.cat((embedded[0], attn_applied[0]), 1)
#         output = self.attn_combine(output).unsqueeze(0)

#         output = F.relu(output)
#         output, hidden = self.gru(output, hidden)

#         output = F.log_softmax(self.out(output[0]), dim=1)
#         return output, hidden, attn_weights

#     def initHidden(self):
#         return torch.zeros(1, 1, self.hidden_size, device=device)
