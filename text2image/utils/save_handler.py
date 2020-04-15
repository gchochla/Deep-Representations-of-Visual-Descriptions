'''Save handling utilities.'''

__all__ = ['model_name', 'hyperparameters', 'get_hyperparameters_from_entry']

import os
import types
from copy import deepcopy

ATTRS = [
    'batches', 'level', 'img_px', 'text_cutoff', 'conv_channels',
    'conv_kernels', 'conv_maxpool', 'rnn_num_layers', 'rnn_bidir',
    'learning_rate', 'conv_dropout', 'rnn_dropout'
]

def model_name(parser_args):
    '''Include training hyperparameters in saved model filename.'''
    return os.path.join(parser_args.model_dir,
                        'text_encoder_' + hyperparameters(parser_args, delim='_') + '.pt')

def hyperparameters(parser_args, delim=','):
    '''Provide string of hyperparameters, separated by `delim`.'''
    args = deepcopy(parser_args)
    args.conv_channels = '-'.join(map(str, parser_args.conv_channels))
    args.conv_kernels = '-'.join(map(str, parser_args.conv_kernels))
    name = delim.join([str(getattr(args, attr)) for attr in ATTRS])
    return name

def get_hyperparameters_from_entry(name: str):
    '''Get hyperparameters from row entry of summary/experiment file.'''
    obj = types.SimpleNamespace()

    values = name.split(',')
    for attr, value in zip(ATTRS, values):
        if attr in ('batches', 'img_px', 'text_cutoff', 'conv_maxpool',
                    'rnn_num_layers'):
            value = int(value)
        elif attr in ('learning_rate', 'conv_dropout', 'rnn_dropout'):
            value = float(value)
        elif attr in ('conv_channels', 'conv_kernels'):
            value = list(map(int, value.split('-')))
        elif attr == 'rnn_bidir':
            value = bool(value)

        setattr(obj, attr, value)

    return obj
