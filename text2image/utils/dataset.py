'''CUB-200-2011 Pytorch dataset.'''

# pylint: disable=no-member
# pylint: disable=not-callable

__all__ = ['CUBDataset']

import os
import torch
# import torchvision.transforms as transforms
import torchfile
import h5py

# from PIL import Image

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
            self.split = lambda s: s.split() # split text by spaces -> words

        else:
            alphabet = 'abcdefghijklmnopqrstuvwxyz0123456789 -,;.!?:\'"/\\|_@#$%^&*~`+-=<>()[]{}'
            self.vocab = {k: i for i, k in enumerate(alphabet)}
            self.vocab['ï'] = self.vocab['i']
            self.vocab['¿'] = self.vocab['?']
            self.vocab['½'] = self.vocab[' ']
            self.vocab_len = len(self.vocab) - 3 # do not include 'ï', '¿', '½'
            self.split = list # splits text into characters

        self.avail_classes = [] # classes to read from
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

        # number of classes to return
        if 'minibatch_size' in kwargs and kwargs['minibatch_size'] > 1:
            self.minibatch_size = min(kwargs['minibatch_size'], len(self.avail_classes))
        else:
            self.minibatch_size = len(self.avail_classes)

    def get_captions(self):
        '''Creates generator that yields one class' captions at a time in a
        `torch.Tensor` of size `images`x`10`x`vocabulary_size`x`caption_max_size`.
        Label is also returned.'''

        for clas in self.avail_classes:
            lbl = int(clas.split('.')[0])

            txt_fns = os.listdir(os.path.join(self.dataset_dir, self.text_dir, clas))
            # remove .txts, keep .h5
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
        '''Creates generator that yields one class' image embeddings
        at a time in a `torch.Tensor`. Label is also returned.'''

        for clas in self.avail_classes:
            lbl = int(clas.split('.')[0])

            img_fn = os.path.join(self.dataset_dir, self.image_dir, clas + '.h5')
            with h5py.File(img_fn, 'r') as h5fp:
                imgs = torch.empty(len(h5fp.keys()), 1024, device=self.device)
                for i, key in enumerate(h5fp.keys()):
                    imgs[i] = torch.tensor(h5fp[key], device=self.device).squeeze()

            yield imgs, lbl


    def get_next_minibatch(self, n_txts=1):
        '''Get next training batch as suggested in
        `Learning Deep Representations of Fine-Grained Visual Descriptions`, i.e.
        one image's embeddings with `n_txts` matching descriptions is returned from
        every class along with their labels.'''

        assert 1 <= n_txts <= 10

        imgs = torch.empty(self.minibatch_size, 1024, device=self.device)
        txts = torch.empty(self.minibatch_size, n_txts, self.vocab_len,
                           self.text_cutoff, device=self.device)
        lbls = torch.empty(self.minibatch_size, dtype=int, device=self.device)

        rand_class_ind = torch.randperm(len(self.avail_classes))[:self.minibatch_size]
        for i, class_ind in enumerate(rand_class_ind):
            clas = self.avail_classes[class_ind]

            lbl = int(clas.split('.')[0])

            img_fn = os.path.join(self.dataset_dir, self.image_dir, clas + '.h5')
            with h5py.File(img_fn, 'r') as h5fp:
                keys = list(h5fp.keys()) # make subscriptable
                rand_img = torch.randint(len(keys), (1,)).item()
                rand_key = keys[rand_img]
                rand_crop = torch.randint(10, (1,)).item()
                img = h5fp[rand_key][:, rand_crop]

            # keys of images where named so that text can be retrieved
            txt_fn = os.path.join(self.dataset_dir, self.text_dir, clas, rand_key + '.h5')
            rand_txts = torch.randperm(10)[:n_txts] + 1 # txt keys start at 1
            with h5py.File(txt_fn, 'r') as txtobj:
                for j, rand_txt in enumerate(rand_txts):
                    txt = txtobj['txt' + str(rand_txt.item())]
                    txt = self.process_text(txt)
                    txts[i, j] = txt

            imgs[i] = torch.tensor(img, device=self.device)
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

    # def get_next_minibatch(self, n_txts=1):
    #     '''Get next training batch as suggested in
    #     `Learning Deep Representations of Fine-Grained Visual Descriptions`, i.e.
    #     one image with `n_txts` matching descriptions is returned from every class along
    #     with their labels.'''

    #     def jitter(img):
    #         '''Randomly flip and crop image `PIL.Image img`.'''
    #         wdt, hgt = img.size
    #         flip = bool(torch.randint(2, (1,)).item())
    #         crop = torch.randint(5, (1,)).item()
    #         crop = [
    #             (0, 0, int(wdt*0.8), int(hgt*0.8)), (int(wdt*0.2), 0, wdt, int(hgt*0.8)),
    #             (0, int(hgt*0.2), int(wdt*0.8), hgt), (int(wdt*0.2), int(hgt*0.2), wdt, hgt),
    #             (int(wdt*0.1), int(hgt*0.1), int(wdt*0.9), int(hgt*0.9))
    #         ][crop]

    #         if flip:
    #             img = img.transpose(Image.FLIP_LEFT_RIGHT)
    #         return img.crop(crop).resize((self.image_px,)*2)

    #     assert 1 <= n_txts <= 10

    #     imgs = torch.empty(self.minibatch_size, 3, self.image_px, self.image_px,
    #                        device=self.device)
    #     txts = torch.empty(self.minibatch_size, n_txts, self.vocab_len,
    #                        self.text_cutoff, device=self.device)
    #     lbls = torch.empty(self.minibatch_size, dtype=int, device=self.device)

    #     pil2tensor = transforms.ToTensor()
    #     rand_class_ind = torch.randperm(len(self.avail_classes))[:self.minibatch_size]
    #     for i, class_ind in enumerate(rand_class_ind):
    #         clas = self.avail_classes[class_ind]

    #         lbl = int(clas.split('.')[0])

    #         img_fns = os.listdir(os.path.join(self.dataset_dir, self.image_dir, clas))
    #         rand_im = torch.randint(len(img_fns), (1,)).item()
    #         rand_txts = torch.randperm(10)[:n_txts] + 1

    #         sample_fn = img_fns[rand_im]
    #         txt_fn = os.path.splitext(sample_fn)[0] + '.h5'

    #         img = Image.open(os.path.join(self.dataset_dir, self.image_dir,
    #                                       clas, sample_fn))
    #         txtobj = h5py.File(os.path.join(self.dataset_dir, self.text_dir,
    #                                         clas, txt_fn), 'r')
    #         for j, rand_txt in enumerate(rand_txts):
    #             txt = txtobj['txt' + str(rand_txt.item())]
    #             txt = self.process_text(txt)
    #             txts[i, j] = txt

    #         imgs[i] = pil2tensor(jitter(img))
    #         lbls[i] = lbl

    #     return imgs, txts.squeeze(), lbls

    # def get_images(self):
    #     '''Creates generator that yields one class' images at a time in a
    #     `torch.Tensor`. Label is also returned.'''

    #     for clas in self.avail_classes:
    #         lbl = int(clas.split('.')[0])

    #         img_fns = os.listdir(os.path.join(self.dataset_dir, self.image_dir, clas))
    #         clas_imgs = torch.empty(len(img_fns), 3, self.image_px, self.image_px,
    #                                 device=self.device)

    #         for i, img_fn in enumerate(img_fns):
    #             img = Image.open(os.path.join(self.dataset_dir, self.image_dir,
    #                                           clas, img_fn)).resize((self.image_px,)*2)
    #             clas_imgs[i] = transforms.ToTensor()(img)

    #         yield clas_imgs, lbl
