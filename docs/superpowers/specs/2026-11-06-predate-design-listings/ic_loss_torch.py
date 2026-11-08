"""Differentiable training loss: negative Pearson correlation over one cross-section (one trading day)."""
import torch


def pearson_ic_loss(pred: torch.Tensor, label: torch.Tensor, epsilon: float = 1e-8) -> torch.Tensor:
    """Return -corr(pred, label) over a 1-D cross-section; epsilon keeps zero-variance inputs finite."""
    if pred.ndim != 1 or pred.shape != label.shape:
        raise ValueError(f"pred and label must be 1-D with the same shape, got {tuple(pred.shape)} and {tuple(label.shape)}")
    if pred.numel() < 3:
        raise ValueError(f"a cross-section needs at least 3 names, got {pred.numel()}")
    pred_centred = pred - pred.mean()
    label_centred = label - label.mean()
    covariance = (pred_centred * label_centred).sum()
    spread = torch.sqrt((pred_centred**2).sum() + epsilon) * torch.sqrt((label_centred**2).sum() + epsilon)
    return -covariance / spread
