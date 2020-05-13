# [Learning Deep Representations of Fine-Grained Visual Descriptions](http://openaccess.thecvf.com/content_cvpr_2016/html/Reed_Learning_Deep_Representations_CVPR_2016_paper.html)

![Imgs&Captions](https://user-images.githubusercontent.com/43141476/80866419-37108000-8c97-11ea-83ef-8c39a59616c1.png)

Implementation of Convolutional Recurrent Neural Nets for zero-shot retrieval of images based on corresponding captions.

![CRNN](https://user-images.githubusercontent.com/43141476/80866252-54911a00-8c96-11ea-9247-a8bbb900130c.png)

CRNNs consist of 1D Convolutional blocks followed by a RNN. Convolutions decrease the sequence length of the captions, allowing the RNN to learn efficiently.

## Installation

Follow [these](./docs/installation.md) instructions.

## Usage

To load pretrained and use character-level model on CUB:

```python
from crnns4captions.utils import load_best_model, captions_to_tensor

# model is returned in eval mode
model = load_best_model('./models/', './models/experiments.txt', device='cuda:0')
captions = ['This bird has blue wings, a pointed red beak and long legs.', 'El pollo loco!']
captions_tensor = captions_to_tensor(captions, device='cuda:0')
reprs = model(captions_tensor) # torch.Size([2, 1024])
```

Alternatively, if you download the files locally without the rest of the repo, you can modify `crnns4captions/utils/deploy.py` by pasting the repo relative code in the file.

## Train & Evaluate

After installation:

1. Make `scripts` executables:

    ```console
    chmod +x scripts/*
    ```

1. Configure the paths in `scripts/to_h5py` and execute it to get h5 files for every t7 file in the CUB dataset (NOTE: do not overwrite the t7 files):

    ```console
    scripts/to_h5py
    ```

1. Change the hyperparameters configuration in `scripts/grid_search` (default: the ones suggested in original paper and the ones used in the pre-trained model) and run the grid_search:

    ```console
    scripts/grid_search
    ```
