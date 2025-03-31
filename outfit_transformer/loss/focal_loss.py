# import torch
# import torch.nn.functional as F
# import torch.nn as nn

# def focal_loss(
#         y_prob: torch.Tensor,
#         y_true: torch.Tensor,
#         alpha: float = 0.75,
#         gamma: float = 2,
#         reduction: str = "mean",
#         ) -> torch.Tensor:
#     ce_loss = F.binary_cross_entropy_with_logits(y_prob, y_true, reduction="none")
#     p_t = y_prob * y_true + (1 - y_prob) * (1 - y_true)
#     loss = ce_loss * ((1 - p_t) ** gamma)

#     if alpha >= 0:
#         alpha_t = alpha * y_true + (1 - alpha) * (1 - y_true)
#         loss = alpha_t * loss

#     if reduction == "none":
#         pass
#     elif reduction == "mean":
#         loss = loss.mean()
#     elif reduction == "sum":
#         loss = loss.sum()
#     else:
#         raise ValueError(f"Invalid Value for arg 'reduction': '{reduction} \n Supported reduction modes: 'none', 'mean', 'sum'")
    
#     return loss
import torch
import torch.nn.functional as F
import torch.nn as nn

class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.75, gamma: float = 2, reduction: str = "mean", multi_class: bool = False):
        """
        Focal Loss implementation with improved numerical stability.

        Args:
            alpha (float): Weighting factor for class imbalance (0 to 1). Set to None to disable.
            gamma (float): Focusing parameter to handle hard examples.
            reduction (str): Specifies reduction type: 'none', 'mean', or 'sum'.
            multi_class (bool): If True, uses softmax for multi-class classification.
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.multi_class = multi_class

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss.

        Args:
            y_pred (torch.Tensor): Logits (before activation) with shape (batch, num_classes) or (batch, 1).
            y_true (torch.Tensor): Ground truth labels with shape (batch, num_classes) or (batch, 1).

        Returns:
            torch.Tensor: Computed focal loss.
        """
        # Apply sigmoid for binary and softmax for multi-class classification
        if self.multi_class:
            y_prob = F.softmax(y_pred, dim=1)
            y_true_one_hot = F.one_hot(y_true.long(), num_classes=y_pred.shape[1]).float()
        else:
            y_prob = torch.sigmoid(y_pred)
            y_true_one_hot = y_true.float()

        # Compute cross-entropy loss
        ce_loss = F.binary_cross_entropy(y_prob, y_true_one_hot, reduction="none")

        # Compute probability of correct class
        p_t = y_prob * y_true_one_hot + (1 - y_prob) * (1 - y_true_one_hot)

        # Apply focal loss scaling factor
        loss = ce_loss * ((1 - p_t) ** self.gamma)

        # Apply alpha weighting if enabled
        if self.alpha is not None:
            alpha_t = self.alpha * y_true_one_hot + (1 - self.alpha) * (1 - y_true_one_hot)
            loss = alpha_t * loss

        # Apply reduction
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss
