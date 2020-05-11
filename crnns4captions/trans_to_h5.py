import os
import argparse

import torchfile
import h5py

def transform(dataset_dir, image_dir, text_dir, new_image_dir, new_text_dir):

    if not os.path.exists(os.path.join(dataset_dir, new_image_dir)):
        os.makedirs(os.path.join(dataset_dir, new_image_dir))
    if not os.path.exists(os.path.join(dataset_dir, new_text_dir)):
        os.makedirs(os.path.join(dataset_dir, new_text_dir))

    for ofn in os.listdir(os.path.join(dataset_dir, image_dir)):
        # note that names in image dir and text dir as supposed to be the same

        # class name
        clas = os.path.splitext(ofn)[0]

        # old t7 image file
        t7fn = os.path.join(dataset_dir, image_dir, ofn)

        # new h5 image file
        h5fn = os.path.join(dataset_dir, new_image_dir, clas + '.h5')

        t7_np = torchfile.load(t7fn)
        with h5py.File(h5fn, 'w') as h5fp:
            for i in range(t7_np.shape[0]):
                h5fp[str(i)] = t7_np[i]

        # old t7 text file
        t7fn = os.path.join(dataset_dir, text_dir, ofn)
        # new h5 text file
        h5fn = os.path.join(dataset_dir, new_text_dir, clas + '.h5')

        t7_np = torchfile.load(t7fn)
        with h5py.File(h5fn, 'w') as h5fp:
            for i in range(t7_np.shape[0]):
                h5fp[str(i)] = t7_np[i]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-d', '--dataset_dir', default='../../datasets/cvpr2016_cub')
    parser.add_argument('-i', '--image_dir', default='images')
    parser.add_argument('-t', '--text_dir', default='text_c10')
    parser.add_argument('-ni', '--new_image_dir', default='images_lazy')
    parser.add_argument('-nt', '--new_text_dir', default='text_c10_lazy')

    args = parser.parse_args()

    transform(dataset_dir=args.dataset_dir, image_dir=args.image_dir, text_dir=args.text_dir,
              new_image_dir=args.new_image_dir, new_text_dir=args.new_text_dir)
