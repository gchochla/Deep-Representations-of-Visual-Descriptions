'''Evaluate text encoder.'''

# pylint: disable=no-member

import os
import argparse

import torch

from crnns4captions.utils import CUBDataset, Fvt, hyperparameters, model_name
from crnns4captions.encoders import HybridCNN

def evaluate_text_encoder():
    '''Main'''

    parser = argparse.ArgumentParser()

    parser.add_argument('-d', '--dataset_dir', required=True, type=str,
                        help='dataset root directory')

    parser.add_argument('-avc', '--avail_class_fn', required=True, type=str,
                        help='txt containing classes used w.r.t dataset directory')

    parser.add_argument('-i', '--image_dir', required=True, type=str,
                        help='directory of images w.r.t dataset directory')

    parser.add_argument('-t', '--text_dir', required=True, type=str,
                        help='directory of descriptions w.r.t dataset directory')

    parser.add_argument('-cut', '--text_cutoff', required=True, type=int,
                        help='fixed dimension of tokens of text')

    parser.add_argument('-lvl', '--level', default='char', type=str, choices=['char', 'word'],
                        help='level of temporal resolution')

    parser.add_argument('-v', '--vocab_fn', type=str,
                        help='vocabulary filename w.r.t dataset directory.' + \
                            'Used only when level=word')

    parser.add_argument('-ch', '--conv_channels', nargs='*', type=int, required=True,
                        help='convolution channels')

    parser.add_argument('-k', '--conv_kernels', nargs='*', type=int, required=True,
                        help='convolution kernel sizes')

    parser.add_argument('-cs', '--conv_strides', nargs='*', type=int, required=True,
                        help='convolution kernel strides')

    parser.add_argument('-rn', '--rnn_num_layers', type=int, required=True,
                        help='number of layers in rnn')

    parser.add_argument('-rh', '--rnn_hidden_size', type=int, default=256,
                        help='size of rnn hidden state (including bidirectionality)')

    parser.add_argument('-rb', '--rnn_bidir', default=False, action='store_true',
                        help='whether to use bidirectional rnn')

    parser.add_argument('--lstm', default=False, action='store_true',
                        help='whether to use lstm instead of vanilla rnn')

    parser.add_argument('-cd', '--conv_dropout', type=float, default=0.,
                        help='dropout in convolutional layers')

    parser.add_argument('-rd', '--rnn_dropout', type=float, default=0.,
                        help='dropout in rnn cells')

    parser.add_argument('-ld', '--lin_dropout', type=float, default=0.,
                        help='dropout in final embedding mapper')

    parser.add_argument('-b', '--batches', type=int, required=True,
                        help='batches the model was trained on')

    parser.add_argument('-mbs', '--minibatch_size', type=int, default=-1,
                        help='minibatch size, <=1 fetches all classes')

    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-4,
                        help='learning rate')

    parser.add_argument('-lrd', '--lr_decay', default=False, action='store_true',
                        help='whether to use learning rate decay')

    parser.add_argument('-md', '--model_dir', type=str, required=True,
                        help='where to retrieve model\'s parameters')

    parser.add_argument('-dev', '--device', type=str, default='cuda:0',
                        help='device to execute on')

    parser.add_argument('-s', '--summary', type=str, help='where to save resulting metrics')

    args = parser.parse_args()

    evalset = CUBDataset(dataset_dir=args.dataset_dir, avail_class_fn=args.avail_class_fn,
                         image_dir=args.image_dir, text_dir=args.text_dir,
                         text_cutoff=args.text_cutoff, level=args.level, vocab_fn=args.vocab_fn,
                         device=args.device)

    txt_encoder = HybridCNN(vocab_dim=evalset.vocab_len, conv_channels=args.conv_channels,
                            conv_kernels=args.conv_kernels, conv_strides=args.conv_strides,
                            rnn_bidir=args.rnn_bidir, rnn_num_layers=args.rnn_num_layers,
                            rnn_hidden_size=args.rnn_hidden_size//(1 + int(args.rnn_bidir)),
                            lstm=args.lstm).to(args.device).eval()
    txt_encoder.load_state_dict(torch.load(model_name(args), map_location=args.device))

    mean_txt_embs = torch.empty(len(evalset.avail_classes), 1024, device=args.device)
    with torch.no_grad():
        for i, (captions, _lbl) in enumerate(evalset.get_captions()):
            mean_txt_embs[i] = txt_encoder(captions.view(-1, *captions.size()[-2:])).mean(dim=0)

    corr, outa = 0, 0
    for i, (img_embs, _lbl) in enumerate(evalset.get_images()):
        preds = Fvt(img_embs, mean_txt_embs).max(dim=1)[1]
        corr += (preds == i).sum().item()
        outa += len(preds)

    print(f'Validation set Accuracy={corr/outa*100:5.2f}%')

    if args.summary:
        if not os.path.exists(os.path.split(args.summary)[0]):
            os.makedirs(os.path.split(args.summary)[0])
        with open(args.summary, 'a') as fp:
            fp.write(f'{corr/outa},{hyperparameters(args)}\n')

if __name__ == '__main__':
    evaluate_text_encoder()
