'''Load pretrained net with one function'''

import torch

from crnns4captions.encoders import HybridCNN
from crnns4captions.utils import get_hyperparameters_from_entry, model_name

def load_best_model(model_dir, summary, device):
    '''Load best model from `model_dir` based on summary of experiments in `summary`
    to device `device`.'''
    with open(summary, 'r') as fp:
        best = (-1, '')
        while True:
            row = fp.readline()
            if not row:
                break
            score_ind = row.index(',')
            if best[0] < float(row[:score_ind]):
                best = (float(row[:score_ind]), row[score_ind+1:])

    margs = get_hyperparameters_from_entry(best[1])
    setattr(margs, 'model_dir', model_dir)


    txt_encoder = HybridCNN(vocab_dim=70, conv_channels=margs.conv_channels,
                            conv_kernels=margs.conv_kernels, conv_strides=margs.conv_strides,
                            rnn_num_layers=margs.rnn_num_layers, rnn_bidir=margs.rnn_bidir,
                            rnn_hidden_size=margs.rnn_hidden_size//(1+int(margs.rnn_bidir)),
                            lstm=margs.lstm).to(device).eval()
    txt_encoder.load_state_dict(torch.load(model_name(margs), map_location=device))

    return txt_encoder

def captions_to_tensor(captions, device):
    '''Properly transform input iterable of strings for the model.'''

    alphabet = 'abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:\'"/\\|_@#$%^&*~`+-=<>()[]{} '
    # pylint: disable=no-member
    text_t = torch.zeros(len(captions), len(alphabet), 201, device=device)
    for i, caption in enumerate(captions):
        for j, tok in enumerate(caption.lower()):
            text_t[i, alphabet.index(tok), j] = 1

    return text_t
