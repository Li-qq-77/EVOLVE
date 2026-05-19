from typing import Tuple
import torch
import torch.nn.functional as F


def aggregate(prob: torch.Tensor, dim: int, keep_bg: bool = True) -> torch.Tensor:
    """Convert per-object foreground probabilities to bg+object logits.

    The model predicts independent object probabilities with shape such as
    B x N x H x W. CUTIE/XMem-style aggregation creates a background channel
    as product(1 - p_i), concatenates it with object channels, and returns
    logits so that a softmax over channels is meaningful.
    """
    eps = 1e-7
    prob = prob.clamp(eps, 1 - eps)
    bg = torch.prod(1 - prob, dim=dim, keepdim=True)
    new_prob = torch.cat([bg, prob], dim=dim) if keep_bg else prob
    logits = torch.log(new_prob / (1 - new_prob).clamp(eps))
    return logits


def cls_to_one_hot(cls_gt: torch.Tensor, num_objects: int) -> torch.Tensor:
    """Convert class-index masks to one-hot masks with background channel.

    Input is usually T x 1 x H x W or T x H x W. Output is T x (N+1) x H x W.
    """
    if cls_gt.dim() == 4 and cls_gt.shape[1] == 1:
        cls_gt = cls_gt[:, 0]
    cls_gt = cls_gt.long()
    return F.one_hot(cls_gt, num_classes=num_objects + 1).permute(0, 3, 1, 2).float()


def pad_divide_by(in_img: torch.Tensor, d: int) -> Tuple[torch.Tensor, Tuple[int, int, int, int]]:
    """Pad a tensor so H and W are divisible by d.

    Returns padded tensor and pad tuple `(left, right, top, bottom)` for unpad().
    """
    h, w = in_img.shape[-2:]
    new_h = h if h % d == 0 else h + d - h % d
    new_w = w if w % d == 0 else w + d - w % d
    lh = (new_h - h) // 2
    uh = new_h - h - lh
    lw = (new_w - w) // 2
    uw = new_w - w - lw
    pad = (lw, uw, lh, uh)
    return F.pad(in_img, pad), pad


def unpad(img: torch.Tensor, pad: Tuple[int, int, int, int]) -> torch.Tensor:
    lw, uw, lh, uh = pad
    h_end = None if uh == 0 else -uh
    w_end = None if uw == 0 else -uw
    return img[..., lh:h_end, lw:w_end]
