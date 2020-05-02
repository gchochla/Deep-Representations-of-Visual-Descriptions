'''Text encoder architectures.'''

# pylint: disable=no-member

__all__ = ['HybridCNN']

import torch
import torch.nn as nn

class HybridCNN(nn.Module):
    '''Architecture used in `Learning Deep Representations of Fine-Grained Visual Descriptions`.
    However, note that the hidden state of only the last layer hidden of the RNN is used
    to compute the embedding.'''

    # pylint: disable=arguments-differ
    # pylint: disable=invalid-name

    def __init__(self, vocab_dim: int, conv_channels, conv_kernels, conv_strides,
                 rnn_hidden_size: int, rnn_num_layers: int, emb_dim=1024, conv_dropout=0.0,
                 rnn_dropout=0.0, lin_dropout=0.0, rnn_bidir=False, lstm=False, map_to_emb=True):

        # pylint: disable=too-many-arguments

        '''Initialize HybridCNN.'''

        super().__init__()

        # sanity checks
        assert hasattr(conv_channels, '__len__')
        assert hasattr(conv_kernels, '__len__')
        assert hasattr(conv_strides, '__len__')
        assert len(conv_channels) == len(conv_kernels)
        assert len(conv_channels) == len(conv_strides)
        # make sure embedding dim is correct whether mapper is used or not
        assert map_to_emb or (rnn_hidden_size * (1 + int(rnn_bidir)) == emb_dim)

        self.map_to_emb = map_to_emb

        conv_channels = [vocab_dim] + conv_channels

        self.conv_layers = nn.ModuleList()
        for in_ch, out_ch, k, s in zip(conv_channels[:-1], conv_channels[1:],
                                       conv_kernels, conv_strides):
            self.conv_layers.append(
                nn.Sequential(
                    nn.Conv1d(in_ch, out_ch, k),
                    nn.MaxPool1d(s),
                    nn.ReLU(),
                    nn.Dropout(conv_dropout)
                )
            )

        if lstm:
            self.rnn = nn.LSTM(conv_channels[-1], hidden_size=rnn_hidden_size,
                               num_layers=rnn_num_layers, batch_first=True,
                               dropout=rnn_dropout if rnn_num_layers > 1 else 0,
                               bidirectional=rnn_bidir)
        else:
            self.rnn = nn.RNN(conv_channels[-1], hidden_size=rnn_hidden_size,
                              num_layers=rnn_num_layers, batch_first=True,
                              dropout=rnn_dropout if rnn_num_layers > 1 else 0,
                              bidirectional=rnn_bidir, nonlinearity='relu')

        if map_to_emb:
            self.emb_mapper = nn.Sequential(
                nn.Dropout(lin_dropout),
                nn.Linear(rnn_hidden_size * (1 + int(rnn_bidir)), emb_dim)
            )

    def _forward(self, x):
        '''Forward propagate through CNN and RNN.'''

        for convl in self.conv_layers:
            x = convl(x)

        # transpose so input_dim and sequence length of rnn
        # become channels and width of the 1d conv result respectively
        x = self.rnn(x.transpose(1, 2))
        return x

    def compute_mean_hidden(self, x):
        '''Compute mean of all the hidden layer activations
        of the final layer through every step.'''

        if self.rnn.bidirectional:
            direction_size = x.size(-1) // 2
            # reverse backward direction
            # to compute aligned mean
            x_front = x[..., :direction_size]
            x_back = x[..., torch.arange(direction_size*2-1, direction_size-1, -1)]
            x_ = torch.cat((x_front, x_back), dim=2)
            return x_.mean(dim=1)

        return x.mean(dim=1)

    def forward(self, x):
        '''Forward propagation.'''

        assert torch.is_tensor(x)
        assert x.size(1) == next(self.conv_layers[0].children()).in_channels

        x = self._forward(x)
        x = self.compute_mean_hidden(x[0])
        if self.map_to_emb:
            x = self.emb_mapper(x)
        return x

class TextCNN(nn.Module):

    # pylint: disable=arguments-differ
    # pylint: disable=invalid-name

    '''Text-based CNN.'''

    def __init__(self, vocab_dim, text_width, conv_channels, conv_kernels,
                 conv_strides, emb_dim=1024):
        '''Init TextCNN.'''

        super().__init__()

        conv_channels = [vocab_dim] + conv_channels

        self.conv_layers = nn.ModuleList()
        for in_ch, out_ch, k, s in zip(conv_channels[:-1], conv_channels[1:],
                                       conv_kernels, conv_strides):
            self.conv_layers.append(
                nn.Sequential(
                    nn.Conv1d(in_ch, out_ch, k),
                    nn.MaxPool1d(s),
                    nn.ReLU()
                )
            )

            text_width = (text_width - k + 1) // s

        self.emb_mapper = nn.Linear(conv_channels[-1] * text_width, emb_dim)

    def forward(self, x):
        '''Forward prop through text-based CNN.'''
        for convl in self.conv_layers:
            x = convl(x)
        x = x.view(x.size(0), -1)
        x = self.emb_mapper(x)
        return x
