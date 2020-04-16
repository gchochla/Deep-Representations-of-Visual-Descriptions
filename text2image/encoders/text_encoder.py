'''Text encoder architectures.'''

# pylint: disable=no-member

__all__ = ['ConvolutionalLSTM']

import torch
import torch.nn as nn

class ConvolutionalLSTM(nn.Module):

    # pylint: disable=invalid-name
    # pylint: disable=arguments-differ

    '''Convolutional Long Short-Term Memory.'''

    def __init__(self, vocab_dim: int, conv_channels, conv_kernels, conv_strides,
                 rnn_hidden_size: int, rnn_num_layers: int, conv_dropout=0.0,
                 rnn_dropout=0.0, rnn_bidir=False):

        # pylint: disable=too-many-arguments

        '''Initialize ConvolutionalLSTM.'''

        super().__init__()

        assert vocab_dim > 0
        assert hasattr(conv_channels, '__len__')
        for i in conv_channels:
            assert isinstance(i, int) and i > 0
        assert hasattr(conv_kernels, '__len__')
        for i in conv_kernels:
            assert isinstance(i, int) and i > 0
        assert len(conv_channels) == len(conv_kernels)
        assert len(conv_channels) == len(conv_strides)
        assert rnn_hidden_size > 0
        assert rnn_num_layers > 0
        assert 0 <= conv_dropout < 1
        assert 0 <= rnn_dropout < 1

        conv_channels = [vocab_dim] + conv_channels

        self.conv_layers = nn.ModuleList()
        for in_ch, out_ch, k, s in zip(conv_channels[:-1], conv_channels[1:],
                                       conv_kernels, conv_strides):
            self.conv_layers.append(
                nn.Sequential(
                    nn.Conv1d(in_ch, out_ch, k, bias=False),
                    nn.BatchNorm1d(out_ch),
                    nn.ReLU(),
                    nn.MaxPool1d(s),
                    nn.Dropout(conv_dropout)
                )
            )

        self.rnn = nn.LSTM(conv_channels[-1], hidden_size=rnn_hidden_size,
                           num_layers=rnn_num_layers, batch_first=True,
                           dropout=rnn_dropout if rnn_num_layers > 1 else 0,
                           bidirectional=rnn_bidir)

    def _forward(self, x):
        '''Forward propagation helper.'''

        for convl in self.conv_layers:
            x = convl(x)

        # sequence length is width
        # and input_dim is channels
        x = self.rnn(x.transpose(1, 2))
        return x

    def compute_embedding(self, x):
        '''Copmute final embedding from all the hidden layer activations
        of the final layer through every step.'''

        if self.rnn.bidirectional:
            direction_size = x.size(-1) // 2
            # reverse backward direction
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
        return self.compute_embedding(x[0])
