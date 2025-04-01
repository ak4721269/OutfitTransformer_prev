import torch
import torch.nn.functional as F
import torch.nn as nn

def bce_loss(
        y_prob: torch.Tensor,
        y_true: torch.Tensor,
        alpha: float = 0.75,
        reduction: str = "mean",
        ) -> torch.Tensor:
    """
    Implements Binary Cross-Entropy (BCE) Loss with optional alpha weighting for class imbalance.

    Args:
        y_prob (torch.Tensor): Predicted probabilities (logits).
        y_true (torch.Tensor): Ground truth labels (0 or 1).
        alpha (float): Weighting factor for positive class (default=0.75).
        reduction (str): Reduction mode - 'none', 'mean', or 'sum'.
    
    Returns:
        torch.Tensor: Computed BCE loss.
    """
    # Compute standard BCE loss
    ce_loss = F.binary_cross_entropy_with_logits(y_prob, y_true, reduction="none")

    # Apply alpha weighting for class imbalance
    if alpha >= 0:
        alpha_t = alpha * y_true + (1 - alpha) * (1 - y_true)
        ce_loss = alpha_t * ce_loss

    # Apply reduction method
    if reduction == "none":
        return ce_loss
    elif reduction == "mean":
        return ce_loss.mean()
    elif reduction == "sum":
        return ce_loss.sum()
    else:
        raise ValueError(f"Invalid Value for arg 'reduction': '{reduction}' \n Supported reduction modes: 'none', 'mean', 'sum'")

# Example usage
if __name__ == "__main__":
    y_prob = torch.tensor([0.8, 0.2, 0.6, 0.9])  # Predicted logits
    y_true = torch.tensor([1.0, 0.0, 1.0, 1.0])  # Ground truth labels

    loss = bce_loss(y_prob, y_true, alpha=0.75, reduction="mean")
    print("BCE Loss:", loss.item())
