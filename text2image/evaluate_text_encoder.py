'''Train text encoder.'''

# pylint: disable=no-member

import os
import argparse

import torch

from text2image.utils import CUBDataset, Fvt
from text2image.encoders import googlenet_feature_extractor, ConvolutionalLSTM

def main():
    '''Main'''

    parser = argparse.ArgumentParser()

    parser.add_argument('-d', '--dataset_dir', required=True, type=str,
                        help='dataset root directory')

    parser.add_argument('-avc', '--avail_class_fn', required=True, type=str,
                        help='txt containing classes used')

    parser.add_argument('-i', '--image_dir', required=True, type=str,
                        help='directory of images w.r.t dataset directory')

    parser.add_argument('-t', '--text_dir', required=True, type=str,
                        help='directory of descriptions w.r.t detaset directory')

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

    parser.add_argument('-rn', '--rnn_num_layers', type=int, required=True,
                        help='number of layers in rnn')

    parser.add_argument('-m', '--conv_maxpool', type=int, default=3,
                        help='maxpool parameter')

    parser.add_argument('-rb', '--rnn_bidir', default=False, action='store_true',
                        help='whether to use bidirectional rnn')

    parser.add_argument('-b', '--batched', type=int, help='batches the model was trained on')

    parser.add_argument('-mfn', '--model_fn', type=str, required=True,
                        help='where to retrieve model\'s parameters')

    parser.add_argument('-s', '--summary', type=str, help='where to save resulting metrics')

    args = parser.parse_args()

    assert args.summary is None or args.batches is not None

    evalset = CUBDataset(dataset_dir=args.dataset_dir, avail_class_fn=args.avail_class_fn,
                         image_dir=args.image_dir, text_dir=args.text_dir, img_px=args.img_px,
                         text_cutoff=args.text_cutoff, level=args.level, vocab_fn=args.vocab_fn)

    img_encoder = googlenet_feature_extractor().eval()
    txt_encoder = ConvolutionalLSTM(vocab_dim=evalset.vocab_len, conv_channels=args.conv_channels,
                                    conv_kernels=args.conv_kernels, conv_maxpool=args.conv_maxpool,
                                    rnn_num_layers=args.rnn_num_layers, rnn_bidir=args.rnn_bidir,
                                    rnn_hidden_size=1024 if not args.rnn_bidir else 512).eval()
    txt_encoder.load_state_dict(torch.load(args.model_fn))

    mean_txt_embs = torch.empty(len(evalset.avail_classes), 1024)
    for i, (captions, _lbl) in enumerate(evalset.get_captions()):
        mean_txt_embs[i] = txt_encoder(captions.view(-1, *captions.size()[-2:])).mean(dim=-1)

    corr, outa = 0, 0
    for i, (images, _lbl) in enumerate(evalset.get_images()):
        img_embs = img_encoder(images)
        preds = Fvt(img_embs, mean_txt_embs).max(dim=1)[1]
        corr += (preds == i).sum().item()
        outa += len(preds)

    print(f'Accuracy={corr/outa*100:5.2f}%')

    if args.summary:
        if not os.path.exists(os.path.split(args.summary)[0]):
            os.makedirs(os.path.split(args.summary)[0])
        with open(args.summary, 'a') as fp:
            fp.write(f'{corr/outa},{args.batches},{args.level},{args.img_px},{args.text_cutoff},' +
                     f'{args.conv_channels},{args.conv_kernels},{args.conv_maxpool},' +
                     f'{args.rnn_num_layers},{args.rnn_bidir}')


if __name__ == '__main__':
    main()