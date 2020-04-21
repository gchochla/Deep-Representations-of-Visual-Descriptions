'''Fetch and evaluate best text encoder.'''

# pylint: disable=no-member

import os
import argparse

import torch

from text2image.utils import CUBDataset, Fvt, model_name, get_hyperparameters_from_entry
from text2image.encoders import HybridCNN

def test_best():
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

    parser.add_argument('-v', '--vocab_fn', type=str,
                        help='vocabulary filename w.r.t dataset directory.' + \
                            'Used only when level=word')

    parser.add_argument('-md', '--model_dir', type=str, required=True,
                        help='where to retrieve model\'s parameters')

    parser.add_argument('-dev', '--device', type=str, default='cuda:0',
                        help='device to execute on')

    parser.add_argument('-s', '--summary', type=str, help='where resulting metrics are saved')

    parser.add_argument('-c', '--clear', default=False, action='store_true',
                        help='whether to clean models folder')

    args = parser.parse_args()

    with open(args.summary, 'r') as fp:
        best = (-1, '')
        while True:
            row = fp.readline()
            if not row:
                break
            score_ind = row.index(',')
            if best[0] < float(row[:score_ind]):
                best = (float(row[:score_ind]), row[score_ind+1:])

    margs = get_hyperparameters_from_entry(best[1])
    setattr(margs, 'model_dir', args.model_dir)

    evalset = CUBDataset(dataset_dir=args.dataset_dir, avail_class_fn=args.avail_class_fn,
                         image_dir=args.image_dir, text_dir=args.text_dir, img_px=args.img_px,
                         text_cutoff=args.text_cutoff, level=margs.level, vocab_fn=args.vocab_fn,
                         device=args.device)

    txt_encoder = HybridCNN(vocab_dim=evalset.vocab_len, conv_channels=margs.conv_channels,
                            conv_kernels=margs.conv_kernels, conv_strides=margs.conv_strides,
                            rnn_num_layers=margs.rnn_num_layers, rnn_bidir=margs.rnn_bidir,
                            rnn_hidden_size=margs.rnn_hidden_size//(1+int(margs.rnn_bidir)),
                            lstm=margs.lstm).to(args.device).eval()
    txt_encoder.load_state_dict(torch.load(model_name(margs)))

    mean_txt_embs = torch.empty(len(evalset.avail_classes), 1024, device=args.device)
    with torch.no_grad():
        for i, (captions, _lbl) in enumerate(evalset.get_captions()):
            mean_txt_embs[i] = txt_encoder(captions.view(-1, *captions.size()[-2:])).mean(dim=0)

    corr, outa = 0, 0
    for i, (img_embs, _lbl) in enumerate(evalset.get_images()):
        preds = Fvt(img_embs, mean_txt_embs).max(dim=1)[1]
        corr += (preds == i).sum().item()
        outa += len(preds)

    print(f'Test set Accuracy={corr/outa*100:5.2f}%')

    if args.clear:
        os.system(f'rm -rf {margs.model_dir}/*.pt')
        torch.save(txt_encoder.state_dict(), model_name(margs))

if __name__ == '__main__':
    test_best()
