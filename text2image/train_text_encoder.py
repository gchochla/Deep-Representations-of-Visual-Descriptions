'''Train text encoder.'''

# pylint: disable=no-member

import os
import argparse

import torch
import torch.optim as optim

from text2image.utils import CUBDataset, joint_embedding_loss, model_name, Fvt
from text2image.encoders import HybridCNN

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

    parser.add_argument('-b', '--batches', required=True, type=int,
                        help='number of batches')

    parser.add_argument('-mbs', '--minibatch_size', type=int, default=-1,
                        help='minibatch size, <=0 fetches all classes')

    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-4,
                        help='learning rate')

    parser.add_argument('-lrd', '--lr_decay', default=False, action='store_true',
                        help='whether to use learning rate decay')

    parser.add_argument('-md', '--model_dir', type=str, help='where to save model\'s parameters')

    parser.add_argument('-dev', '--device', type=str, default='cuda:0',
                        help='device to execute on')

    parser.add_argument('-pe', '--print_every', type=int,
                        help='print accuracy of batch periodically')

    parser.add_argument('-se', '--save_every', type=int,
                        help='save model periodically. Note that' + \
                            ' all copies are preserved, so use with caution')

    args = parser.parse_args()

    assert args.save_every is None or args.model_dir is not None

    trainset = CUBDataset(dataset_dir=args.dataset_dir, avail_class_fn=args.avail_class_fn,
                          image_dir=args.image_dir, text_dir=args.text_dir,
                          text_cutoff=args.text_cutoff, level=args.level, vocab_fn=args.vocab_fn,
                          device=args.device, minibatch_size=args.minibatch_size)

    txt_encoder = HybridCNN(vocab_dim=trainset.vocab_len, conv_channels=args.conv_channels,
                            conv_kernels=args.conv_kernels, conv_strides=args.conv_strides,
                            rnn_bidir=args.rnn_bidir, conv_dropout=args.conv_dropout,
                            lin_dropout=args.lin_dropout, rnn_dropout=args.rnn_dropout,
                            rnn_hidden_size=args.rnn_hidden_size//(1 + int(args.rnn_bidir)),
                            rnn_num_layers=args.rnn_num_layers, lstm=args.lstm)\
                                .to(args.device).train()

    optimizer = optim.Adam(txt_encoder.parameters(), lr=args.learning_rate)
    if args.lr_decay:
        # keep value closer to 1 than usual because of no solid def of epoch
        lr_decay = optim.lr_scheduler.MultiplicativeLR(optimizer, lambda b: 0.9997)

    if args.model_dir:
        if not os.path.exists(args.model_dir):
            os.makedirs(args.model_dir)

    total_batches = args.batches
    for batch in range(total_batches):
        img_embs, txts, lbls = trainset.get_next_minibatch()
        txt_embs = txt_encoder(txts)

        loss = joint_embedding_loss(img_embs, txt_embs, lbls, batched=False, device=args.device)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if args.lr_decay:
            lr_decay.step()

        if args.print_every is not None and \
            ((batch+1) % args.print_every == 0 or batch == 0):
            comp = Fvt(img_embs, txt_embs)
            corr = (comp.max(dim=-1)[1] == torch.arange(comp.size(0), device=args.device))\
                .sum().item()
            print(f'Batch {batch+1} loss {loss.item():.4f}, accuracy: {corr}/{comp.size(0)}')

        if args.save_every is not None and (batch + 1) % args.save_every == 0:
            # note that assertion ensures model_dir
            args.batches = batch + 1 # NOTE: remove this line to save to the same file each time
            torch.save(txt_encoder.state_dict(), model_name(args))

    print('Done training')

    if args.model_dir:
        args.batches = total_batches
        torch.save(txt_encoder.state_dict(), model_name(args))

if __name__ == '__main__':
    train_text_encoder()
