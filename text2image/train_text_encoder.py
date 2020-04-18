'''Train text encoder.'''

# pylint: disable=no-member

import os
import argparse

import torch
import torch.optim as optim

from text2image.utils import CUBDataset, joint_embedding_loss, model_name, Fvt
from text2image.encoders import ConvolutionalLSTM

def train_text_encoder():
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

    parser.add_argument('-px', '--img_px', required=True, type=int,
                        help='pixels for image to be resized to')

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

    parser.add_argument('-rb', '--rnn_bidir', default=False, action='store_true',
                        help='whether to use bidirectional rnn')

    parser.add_argument('-cd', '--conv_dropout', type=float, default=0.,
                        help='dropout in convolutional layers')

    parser.add_argument('-rd', '--rnn_dropout', type=float, default=0.,
                        help='dropout in lstm cells')

    parser.add_argument('-b', '--batches', required=True, type=int,
                        help='number of batches')

    parser.add_argument('-mbs', '--minibatch_size', type=int, default=-1,
                        help='minibatch size, <=0 fetches all classes')

    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-4,
                        help='learning rate')

    parser.add_argument('-md', '--model_dir', type=str, help='where to save model\'s parameters')

    parser.add_argument('-dev', '--device', type=str, default='cuda:0',
                        help='device to execute on')

    parser.add_argument('-pe', '--print_every', type=int,
                        help='print accuracy of batch in regular intervals')

    args = parser.parse_args()

    trainset = CUBDataset(dataset_dir=args.dataset_dir, avail_class_fn=args.avail_class_fn,
                          image_dir=args.image_dir, text_dir=args.text_dir, img_px=args.img_px,
                          text_cutoff=args.text_cutoff, level=args.level, vocab_fn=args.vocab_fn,
                          device=args.device, minibatch_size=args.minibatch_size)

    txt_encoder = ConvolutionalLSTM(vocab_dim=trainset.vocab_len, conv_channels=args.conv_channels,
                                    conv_kernels=args.conv_kernels, conv_strides=args.conv_strides,
                                    rnn_bidir=args.rnn_bidir, conv_dropout=args.conv_dropout,
                                    rnn_dropout=args.rnn_dropout, rnn_num_layers=args.rnn_num_layers,
                                    rnn_hidden_size=1024 if not args.rnn_bidir else 512)\
                                        .to(args.device).train()

    optimizer = optim.Adam(txt_encoder.parameters(), lr=args.learning_rate)

    for batch in range(args.batches):
        img_embs, txts, lbls = trainset.get_next_minibatch()
        txt_embs = txt_encoder(txts)

        loss = joint_embedding_loss(img_embs, txt_embs, lbls, batched=False, device=args.device)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if args.print_every is not None and \
            ((batch+1) % args.print_every == 0 or batch == 0):
            comp = Fvt(img_embs, txt_embs)
            corr = (comp.max(dim=-1)[1] == torch.arange(comp.size(0), device=args.device))\
                .sum().item()
            print(f'Batch {batch+1} loss {loss.item():.4f}, accuracy: {corr}/{comp.size(0)}')

    print('Done training')

    if args.model_dir:
        if not os.path.exists(args.model_dir):
            os.makedirs(args.model_dir)
        torch.save(txt_encoder.state_dict(), model_name(args))

if __name__ == '__main__':
    train_text_encoder()
