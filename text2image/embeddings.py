'''Embed images prior to training text encoder.'''

# pylint: disable=no-member

import os
import argparse
from PIL import Image

import torch
import torchfile
import h5py

from torchvision.transforms import transforms

from text2image.encoders import googlenet_feature_extractor

def zeropad(digit, width=4):
    '''Zero pad digit to keep sorted
    in directory view'''
    return '0' * (width - len(str(digit))) + str(digit)

def embed(dataset_dir, image_dir, img_px, image_emb_dir, device, classes, train):
    '''Calculate the embeddings of images (given that the image encoder
    remains frozen during training) and save in .h5 format, where each image's
    embedding can be accessed at 'emb#'.'''

    image_dir = os.path.join(dataset_dir, image_dir)
    image_emb_dir = os.path.join(dataset_dir, image_emb_dir)

    # get names of available classes
    avail_classes = []
    with open(os.path.join(dataset_dir, classes), 'r') as avcls:
        while True:
            line = avcls.readline()
            if not line:
                break
            avail_classes.append(line.strip())

    img_encoder = googlenet_feature_extractor().to(device).eval()
    pil2tensor = transforms.ToTensor()
    with torch.no_grad():
        for clas_dir in os.listdir(image_dir):

            if clas_dir not in avail_classes: # if not instructed to meddle with class
                continue

            # get file name of embeddings of class, e.g. bluh/001.Black_footed_Albatross.h5
            clas_embs_fn = os.path.join(image_emb_dir, clas_dir) + '.h5'

            clas_ims = os.listdir(os.path.join(image_dir, clas_dir))
            active_cnt = 0 # images could be grayscale, keep count of RGB ones
            with h5py.File(clas_embs_fn, 'w') as h5fp:
                for clas_im in clas_ims:
                    img = Image.open(os.path.join(image_dir, clas_dir, clas_im))
                    img_name = os.path.splitext(clas_im)[0] # get name to keep corresp with text
                    if train: # if train, include embedding of crops
                        embs = torch.empty(1024, 10, device=device)
                        wdt, hgt = img.size
                        crops = [
                            (0, 0, int(wdt*0.8), int(hgt*0.8)),
                            (int(wdt*0.2), 0, wdt, int(hgt*0.8)),
                            (0, int(hgt*0.2), int(wdt*0.8), hgt),
                            (int(wdt*0.2), int(hgt*0.2), wdt, hgt),
                            (int(wdt*0.1), int(hgt*0.1), int(wdt*0.9), int(hgt*0.9))
                        ]

                        rgb = True
                        for j, crop in enumerate(crops):
                            tens = pil2tensor(img.crop(crop).resize((img_px,)*2))
                            if tens.size(0) != 3:
                                # if image grayscale. should be activate first time it's checked
                                rgb = False
                                break
                            embs[..., 2*j] = img_encoder(tens.unsqueeze(0))
                            tens = pil2tensor(
                                img.transpose(Image.FLIP_LEFT_RIGHT).crop(crop).resize((img_px,)*2)
                            )
                            embs[..., 2*j+1] = img_encoder(tens.unsqueeze(0))
                        if not rgb:
                            continue

                    else: # if testing, only original images are used
                        img = pil2tensor(img.resize((img_px,)*2))
                        if img.size(0) != 3:
                            continue
                        embs = img_encoder(img.unsqueeze(0)).squeeze()
                        active_cnt += 1

                    if device.startswith('cuda'):
                        embs = embs.cpu()
                    embs = embs.detach().numpy()

                    h5fp[img_name] = embs

def transform(dataset_dir, image_dir, image_emb_dir, classes, clear):
    '''Transform embeddings from .t7 to .h5'''

    image_dir = os.path.join(dataset_dir, image_dir)
    image_emb_dir = os.path.join(dataset_dir, image_emb_dir)

    if not os.path.exists(image_emb_dir):
        os.makedirs(image_emb_dir)

    avail_classes = []
    with open(os.path.join(dataset_dir, classes), 'r') as avcls:
        while True:
            line = avcls.readline()
            if not line:
                break
            avail_classes.append(line.strip())

    for clas_embs in os.listdir(image_dir):
        # get name of class
        clas_name = os.path.splitext(clas_embs)[0]

        if clas_name not in avail_classes:
            continue # if not instructed to meddle with class

        # get read and write filenames
        clas_embs_fn = os.path.join(image_dir, clas_embs)
        new_clas_embs_fn = os.path.join(image_emb_dir, clas_name + '.h5')
        # n_images x 1024 x 10
        embs = torchfile.load(clas_embs_fn)

        with h5py.File(new_clas_embs_fn, 'w') as h5fp:
            for img in range(embs.shape[0]):
                h5fp.create_dataset(f'img{img}', data=embs[img])

        if clear:
            os.system(f'rm {image_dir}/*.t7')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-d', '--dataset_dir', required=True, type=str,
                        help='root directory of dataset')

    parser.add_argument('-i', '--image_dir', required=True, type=str,
                        help='directory of images')

    parser.add_argument('-ied', '--image_emb_dir', required=True, type=str,
                        help='directory to save embeddings')

    parser.add_argument('-px', '--img_px', default=224, type=int,
                        help='pixels for image to be resized to')

    parser.add_argument('-dev', '--device', default='cuda:0', type=str,
                        help='device to execute on')

    parser.add_argument('-c', '--clear', default=False, action='store_true',
                        help='whether to delete .t7 after transform')

    parser.add_argument('-cls', '--class_fn', type=str, required=True,
                        help='txt of classes to manipulate')

    parser.add_argument('--train', default=False, action='store_true',
                        help='whether')

    parser.add_argument('-emb', '--embed', default=False, action='store_true',
                        help='If set, creates embeddings, else transforms them')

    args = parser.parse_args()

    if args.embed:
        embed(args.dataset_dir, args.image_dir, args.img_px, args.image_emb_dir,
              args.device, args.class_fn, args.train)
    else:
        transform(args.dataset_dir, args.image_dir, args.image_emb_dir,
                  args.class_fn, args.clear)
