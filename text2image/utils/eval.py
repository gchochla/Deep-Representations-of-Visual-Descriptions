'''Evaluation utilities.'''

# pylint: disable=no-member

__all__ = ['Fvt', 'modality_loss', 'joint_embedding_loss']

import torch
import torch.nn.functional as F

def Fvt(x, y):

    # pylint: disable=invalid-name

    '''Compatibility metric.

    Arguments:

    * `x`, `y`: `torch.Tensor` where `.size(-1)` are equal.

    Returns:

    Similarity metric between every pair of vectors.'''

    return torch.matmul(x, y.transpose(-2, -1))

def modality_loss(comp: torch.Tensor, dim: int, batched=False, device='cuda:0'):
    '''Compute loss across modality (image or text). The losses are defined as
    (`Learning Deep Representations of Fine-Grained Visual Descriptions`):

    l_v(v_n, t_n, y_n) = max_{y in labels}(0, D(y, y_n) + E_{t from T(y)}[F(v_n,t) - F[v_n,t_n]])

    l_v(v_n, t_n, y_n) = max_{y in labels}(0, D(y, y_n) + E_{v from V(y)}[F(v,t_n) - F[v_n,t_n]])

    As suggested, only on description pre class is used, so the Expectation is approximated
    by the value of one sample. The implementation is fully vectorized. Notice that the labels
    are not used as the D loss is zero in the diagonal if the class of an index between tensors
    is the same. `dim` essentially controls the broadcasting dimensions, e.g. if `dim==0`, the
    diagonal is unsqueezed to a row-vector and is therefore broadcasted by replicating the diagonal
    elements column-wise, i.e. the "iteration" of classes concerns the modality differing between
    rows.

    Arguments:

    * `comp`: square `torch.Tensor` (maybe excluding batch dimension) containing compatibilities.

    * `dim`: `int` dimension to check compatibility.

    * `batched`: `bool` indicating if 1st dimension is batch dimension.

    Returns:

    * `torch.Tensor` of loss.'''

    batched = int(batched) # 0 or 1
    # size(1) is not concerned with the existence of batch dimension
    # unsqueeze in diagonal to control
    return F.relu(
        1 - torch.eye(comp.size(1), device=device) + \
            comp - comp.diagonal(0, batched, batched+1).unsqueeze(dim+batched)
    ).max(dim=dim+batched)[0].mean(dim=-1).sum()

def joint_embedding_loss(im_enc, txt_enc, _lbls, batched=False, device='cuda:0'):
    '''Compute and return loss as defined in
    `Learning Deep Representations of Fine-Grained Visual Descriptions`.

    Arguments:

    * `im_enc`: `torch.Tensor` of vector representation of images.

    * `txt_enc`: `torch.Tensor` of vector representation of captions.
    Must have same dimensionality as `im_enc`.

    * `_lbls`: `torch.Tensor` of labels. Not used, included for possible
    compatibility with other such losses.

    `batched`: `bool`, whether 1st dimension is batch dimension.

    Returns:

    * `torch.Tensor` of losses.'''

    assert im_enc.size() == txt_enc.size()

    comp = Fvt(im_enc, txt_enc)

    loss = modality_loss(comp, 0, batched, device) + \
        modality_loss(comp, 1, batched, device)

    return loss
