'''CUB-200-2011 Pytorch dataset.'''

# pylint: disable=no-member

__all__ = ['CUBDataset']

import os
import torch
import torchvision.transforms as transforms
import torchfile
import h5py

from PIL import Image

class CUBDataset(torch.utils.data.Dataset):

    # pylint: disable=abstract-method
    # pylint: disable=too-many-instance-attributes

    '''CUB-200-2011 dataset.'''
    def __init__(self, dataset_dir: str, avail_class_fn: str, image_dir: str,
                 text_dir: str, img_px: int, text_cutoff: int, level='char',
                 device='cuda:0', **kwargs):

        # pylint: disable=too-many-arguments

        '''Initialize dataset.'''

        super().__init__()

        assert level in ('word', 'char')

        if level == 'word':
            assert 'vocab_fn' in kwargs

            vocab_fn = kwargs['vocab_fn']

            vocab = torchfile.load(os.path.join(dataset_dir, vocab_fn))
            # keys / vocab words are bytes, values start from 1
            self.vocab = {k.decode('utf-8'): vocab[k]-1 for k in vocab}
            self.vocab_len = len(self.vocab)
            self.split = lambda s: s.split()

        else:
            alphabet = 'abcdefghijklmnopqrstuvwxyz0123456789 -,;.!?:\'"/\\|_@#$%^&*~`+-=<>()[]{}'
            self.vocab = {k: i for i, k in enumerate(alphabet)}
            self.vocab['ï'] = self.vocab['i']
            self.vocab['¿'] = self.vocab['?']
            self.vocab['½'] = self.vocab[' ']
            self.vocab_len = len(self.vocab) - 3 # do not include 'ï', '¿', '½'
            self.split = list

        self.avail_classes = []
        with open(os.path.join(dataset_dir, avail_class_fn), 'r') as avcls:
            while True:
                line = avcls.readline()
                if not line:
                    break
                self.avail_classes.append(line.strip())

        self.dataset_dir = dataset_dir
        self.image_dir = image_dir
        self.text_dir = text_dir
        self.image_px = img_px
        self.text_cutoff = text_cutoff
        self.device = device

        if 'minibatch_size' in kwargs and kwargs['minibatch_size'] > 1:
            self.minibatch_size = kwargs['minibatch_size']
        else:
            self.minibatch_size = len(self.avail_classes)

    def get_captions(self):
        '''Creates generator that yields one class' captions at a time in a
        `torch.Tensor` of size `images`x`10`x`vocabulary_size`x`caption_max_size`.
        Label is also returned.'''

        for clas in self.avail_classes:
            lbl = int(clas.split('.')[0])

            txt_fns = os.listdir(os.path.join(self.dataset_dir, self.text_dir, clas))
            txt_fns = list(filter(lambda s: os.path.splitext(s)[1] == '.h5', txt_fns))
            clas_txts = torch.empty(len(txt_fns), 10, self.vocab_len,
                                    self.text_cutoff, device=self.device)

            for i, txt_fn in enumerate(txt_fns):
                txtvals = h5py.File(os.path.join(self.dataset_dir, self.text_dir,
                                                 clas, txt_fn), 'r').values()
                for j, txt in enumerate(txtvals):
                    clas_txts[i, j] = self.process_text(txt)

            yield clas_txts, lbl

    def get_images(self):
        '''Creates generator that yields one class' images at a time in a
        `torch.Tensor`. Label is also returned.'''

        for clas in self.avail_classes:
            lbl = int(clas.split('.')[0])

            img_fns = os.listdir(os.path.join(self.dataset_dir, self.image_dir, clas))
            clas_imgs = torch.empty(len(img_fns), 3, self.image_px, self.image_px,
                                    device=self.device)

            for i, img_fn in enumerate(img_fns):
                img = Image.open(os.path.join(self.dataset_dir, self.image_dir,
                                              clas, img_fn)).resize((self.image_px,)*2)
                clas_imgs[i] = transforms.ToTensor()(img)

            yield clas_imgs, lbl

    def get_next_minibatch(self, n_txts=1):
        '''Get next training batch as suggested in
        `Learning Deep Representations of Fine-Grained Visual Descriptions`, i.e.
        one image with `n_txts` matching descriptions is returned from every class along
        with their labels.'''

        assert 1 <= n_txts <= 10

        imgs = torch.empty(self.minibatch_size, 3, self.image_px, self.image_px,
                           device=self.device)
        txts = torch.empty(self.minibatch_size, n_txts, self.vocab_len,
                           self.text_cutoff, device=self.device)
        lbls = torch.empty(self.minibatch_size, dtype=int, device=self.device)

        rand_class_ind = torch.randperm(len(self.avail_classes))[:self.minibatch_size]
        for i, class_ind in enumerate(rand_class_ind):
            clas = self.avail_classes[class_ind]

            lbl = int(clas.split('.')[0])

            img_fns = os.listdir(os.path.join(self.dataset_dir, self.image_dir, clas))
            rand_im = torch.randint(len(img_fns), (1,)).item()
            rand_txts = torch.randperm(10)[:n_txts] + 1

            sample_fn = img_fns[rand_im]
            txt_fn = os.path.splitext(sample_fn)[0] + '.h5'

            img = Image.open(os.path.join(self.dataset_dir, self.image_dir,
                                          clas, sample_fn)).resize((self.image_px,)*2)
            txtobj = h5py.File(os.path.join(self.dataset_dir, self.text_dir,
                                            clas, txt_fn), 'r')
            for j, rand_txt in enumerate(rand_txts):
                txt = txtobj['txt' + str(rand_txt.item())]
                txt = self.process_text(txt)
                txts[i, j] = txt

            imgs[i] = transforms.ToTensor()(img)
            lbls[i] = lbl

        return imgs, txts.squeeze(), lbls

    def process_text(self, text):
        '''Transform array of ascii codes to one-hot sequence.'''

        # get ords from h5py object, trans to iter of chars,
        # trans to str and split
        text = self.split(''.join(map(chr, text[:self.text_cutoff].astype(int))))

        res = torch.zeros(self.vocab_len, self.text_cutoff, device=self.device)
        res[[[self.vocab[tok] for tok in text], range(len(text))]] = 1

        return res
