'''CUB-200-2011 Pytorch dataset.'''

# pylint: disable=no-member
# pylint: disable=not-callable

__all__ = ['CUBDataset', 'CUBDatasetLazy']

import os
import string
import torch
# import torchvision.transforms as transforms
import torchfile
import h5py

class CUBDataset(torch.utils.data.Dataset):

    # pylint: disable=abstract-method
    # pylint: disable=too-many-instance-attributes

    '''CUB-200-2011 dataset.'''
    def __init__(self, dataset_dir: str, avail_class_fn: str, image_dir: str,
                 text_dir: str, text_cutoff: int, level='char', device='cuda:0', **kwargs):

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
            self.split = lambda s: list(filter(
                lambda ss: ss, # remove empty strings, Nones, etc...
                map(
                    lambda ss: ss.translate(str.maketrans('', '', string.punctuation)),
                    s.split()
                )
            )) # split text by spaces, remove punctuation and blanks -> words

        else:
            alphabet = 'abcdefghijklmnopqrstuvwxyz0123456789 ,;.!?:\'"/\\|_@#$%^&*~`+-=<>()[]{}'
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

class CUBDatasetLazy(torch.utils.data.Dataset):

    # pylint: disable=abstract-method
    # pylint: disable=too-many-instance-attributes

    '''CUB-200-2011 dataset.'''
    def __init__(self, dataset_dir: str, avail_class_fn: str, image_dir: str,
                 text_dir: str, device='cuda:0', **kwargs):

        # pylint: disable=too-many-arguments

        '''Initialize dataset. Note that non lazy dirs are expected,
        they will be used wherever necessary.'''

        super().__init__()

        # alphabet = 'abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:\'"/\\|_@#$%^&*~`+-=<>()[]{} '
        self.vocab_len = 70

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

            txt_fn = os.path.join(self.dataset_dir, self.text_dir, clas + '.t7')
            txt_np = torchfile.load(txt_fn)
            txt_t = self.process_text(txt_np)

            yield txt_t, lbl

    def get_images(self):
        '''Creates generator that yields one class' image embeddings
        at a time in a `torch.Tensor`. Label is also returned.'''

        for clas in self.avail_classes:
            lbl = int(clas.split('.')[0])

            imgs_fn = os.path.join(self.dataset_dir, self.image_dir, clas + '.t7')
            imgs_np = torchfile.load(imgs_fn)
            # the original image is used during inference -> index 0
            imgs_t = torch.tensor(imgs_np[..., 0], dtype=torch.float, device=self.device)

            yield imgs_t, lbl


    def get_next_minibatch(self, n_txts=1):
        '''Get next training batch as suggested in
        `Learning Deep Representations of Fine-Grained Visual Descriptions`, i.e.
        one image's embeddings with `n_txts` matching descriptions is returned from
        every class along with their labels.'''

        assert 1 <= n_txts <= 10

        imgs = torch.empty(self.minibatch_size, 1024, device=self.device)
        txts = torch.empty(self.minibatch_size, n_txts, self.vocab_len,
                           201, device=self.device)
        lbls = torch.empty(self.minibatch_size, dtype=int, device=self.device)

        rand_class_ind = torch.randperm(len(self.avail_classes))[:self.minibatch_size]
        for i, class_ind in enumerate(rand_class_ind):
            clas = self.avail_classes[class_ind]

            lbl = int(clas.split('.')[0])

            img_fn = os.path.join(self.dataset_dir, self.image_dir + '_lazy', clas + '.h5')
            with h5py.File(img_fn, 'r') as h5fp:
                # pick an image from the class at rand
                rand_img = str(torch.randint(len(h5fp), (1,)).item())
                # pick a crop at rand
                rand_crop = torch.randint(10, (1,)).item()
                imgs[i] = torch.tensor(h5fp[rand_img][..., rand_crop], device=self.device)


            txt_fn = os.path.join(self.dataset_dir, self.text_dir + '_lazy', clas + '.h5')
            with h5py.File(txt_fn, 'r') as h5fp:
                # get n_txts random texts
                rand_txts = torch.randperm(10)[:n_txts]
                # reshape because process text expects 3d
                txts[i] = self.process_text(h5fp[rand_img][..., rand_txts].reshape(1, 201, len(rand_txts)))

            lbls[i] = lbl

        return imgs, txts.squeeze(), lbls

    def process_text(self, text):
        '''Transform np array of ascii codes to one-hot sequence.'''

        ohvec = torch.zeros(text.shape[0], text.shape[2], self.vocab_len, 
                            text.shape[1], device=self.device)

        for corr_img in range(text.shape[0]):
            for cap in range(text.shape[2]):
                for tok in range(text.shape[1]):
                    # -1 because of lua indexing
                    ohvec[corr_img, cap, int(text[corr_img, tok, cap])-1, tok] = 1

        return ohvec
