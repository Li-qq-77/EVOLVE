# Minimal PointRend-style point sampling utilities used by model/losses.py.
import torch
import torch.nn.functional as F


def calculate_uncertainty(logits: torch.Tensor) -> torch.Tensor:
    if logits.shape[1] == 1:
        return -torch.abs(logits)
    top2 = torch.topk(logits, k=2, dim=1).values
    return -(top2[:, 0:1] - top2[:, 1:2]).abs()


def point_sample(input: torch.Tensor, point_coords: torch.Tensor, **kwargs) -> torch.Tensor:
    add_dim = False
    if point_coords.dim() == 3:
        point_coords = point_coords.unsqueeze(2)
        add_dim = True
    output = F.grid_sample(input, 2.0 * point_coords - 1.0, **kwargs)
    if add_dim:
        output = output.squeeze(3)
    return output


def get_uncertain_point_coords_with_randomness(
    coarse_logits: torch.Tensor,
    uncertainty_func,
    num_points: int,
    oversample_ratio: float,
    importance_sample_ratio: float,
) -> torch.Tensor:
    assert oversample_ratio >= 1
    assert 0 <= importance_sample_ratio <= 1
    num_boxes = coarse_logits.shape[0]
    num_sampled = int(num_points * oversample_ratio)
    point_coords = torch.rand(num_boxes, num_sampled, 2, device=coarse_logits.device)
    point_logits = point_sample(coarse_logits, point_coords, align_corners=False)
    point_uncertainties = uncertainty_func(point_logits).squeeze(1)
    num_uncertain = int(importance_sample_ratio * num_points)
    num_random = num_points - num_uncertain
    idx = torch.topk(point_uncertainties, k=num_uncertain, dim=1)[1]
    shift = num_sampled * torch.arange(num_boxes, dtype=torch.long, device=coarse_logits.device)
    idx = idx + shift[:, None]
    point_coords = point_coords.reshape(-1, 2)[idx.reshape(-1), :].reshape(num_boxes, num_uncertain, 2)
    if num_random > 0:
        random_coords = torch.rand(num_boxes, num_random, 2, device=coarse_logits.device)
        point_coords = torch.cat([point_coords, random_coords], dim=1)
    return point_coords
