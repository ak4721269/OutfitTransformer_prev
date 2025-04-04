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
# # import torch
# # import torch.nn.functional as F
# # import torch.nn as nn

# # class FocalLoss(nn.Module):
# #     def __init__(self, alpha: float = 0.75, gamma: float = 2, reduction: str = "mean", multi_class: bool = False):
# #         """
# #         Focal Loss implementation with improved numerical stability.

# #         Args:
# #             alpha (float): Weighting factor for class imbalance (0 to 1). Set to None to disable.
# #             gamma (float): Focusing parameter to handle hard examples.
# #             reduction (str): Specifies reduction type: 'none', 'mean', or 'sum'.
# #             multi_class (bool): If True, uses softmax for multi-class classification.
# #         """
# #         super(FocalLoss, self).__init__()
# #         self.alpha = alpha
# #         self.gamma = gamma
# #         self.reduction = reduction
# #         self.multi_class = multi_class

# #     def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
# #         """
# #         Compute focal loss.

# #         Args:
# #             y_pred (torch.Tensor): Logits (before activation) with shape (batch, num_classes) or (batch, 1).
# #             y_true (torch.Tensor): Ground truth labels with shape (batch, num_classes) or (batch, 1).

# #         Returns:
# #             torch.Tensor: Computed focal loss.
# #         """
# #         # Apply sigmoid for binary and softmax for multi-class classification
# #         if self.multi_class:
# #             y_prob = F.softmax(y_pred, dim=1)
# #             y_true_one_hot = F.one_hot(y_true.long(), num_classes=y_pred.shape[1]).float()
# #         else:
# #             y_prob = torch.sigmoid(y_pred)
# #             y_true_one_hot = y_true.float()

# #         # Compute cross-entropy loss
# #         ce_loss = F.binary_cross_entropy(y_prob, y_true_one_hot, reduction="none")

# #         # Compute probability of correct class
# #         p_t = y_prob * y_true_one_hot + (1 - y_prob) * (1 - y_true_one_hot)

# #         # Apply focal loss scaling factor
# #         loss = ce_loss * ((1 - p_t) ** self.gamma)

# #         # Apply alpha weighting if enabled
# #         if self.alpha is not None:
# #             alpha_t = self.alpha * y_true_one_hot + (1 - self.alpha) * (1 - y_true_one_hot)
# #             loss = alpha_t * loss

# #         # Apply reduction
# #         if self.reduction == "mean":
# #             return loss.mean()
# #         elif self.reduction == "sum":
# #             return loss.sum()
# #         return loss
# # import torch
# # import torch.nn.functional as F

# # def focal_loss(
# #         y_prob: torch.Tensor,
# #         y_true: torch.Tensor,
# #         alpha: float = 0.75,
# #         gamma: float = 2,
# #         reduction: str = "mean",
# #         multi_class: bool = False
# #         ) -> torch.Tensor:
# #     """
# #     Focal Loss for binary and multi-class classification.

# #     Args:
# #         y_prob (torch.Tensor): Logits (before activation) with shape (batch, num_classes) or (batch, 1).
# #         y_true (torch.Tensor): Ground truth labels with shape (batch, num_classes) or (batch, 1).
# #         alpha (float): Weighting factor for class imbalance (0 to 1). Set to -1 to disable.
# #         gamma (float): Focusing parameter to handle hard examples.
# #         reduction (str): Specifies reduction type: 'none', 'mean', or 'sum'.
# #         multi_class (bool): If True, uses softmax for multi-class classification.

# #     Returns:
# #         torch.Tensor: Computed focal loss.
# #     """
# #     if multi_class:
# #         # Apply softmax for multi-class
# #         y_prob = F.softmax(y_prob, dim=1)
# #         y_true_one_hot = F.one_hot(y_true.long(), num_classes=y_prob.shape[1]).float()
# #     else:
# #         # Apply sigmoid for binary classification
# #         y_prob = torch.sigmoid(y_prob)
# #         y_true_one_hot = y_true.float()

# #     # Compute cross-entropy loss
# #     ce_loss = F.binary_cross_entropy(y_prob, y_true_one_hot, reduction="none")

# #     # Compute probability of correct class
# #     p_t = y_prob * y_true_one_hot + (1 - y_prob) * (1 - y_true_one_hot)

# #     # Apply focal loss scaling factor
# #     loss = ce_loss * ((1 - p_t) ** gamma)

# #     # Apply alpha weighting if enabled
# #     if alpha >= 0:
# #         alpha_t = alpha * y_true_one_hot + (1 - alpha) * (1 - y_true_one_hot)
# #         loss = alpha_t * loss

# #     # Apply reduction
# #     if reduction == "mean":
# #         return loss.mean()
# #     elif reduction == "sum":
# #         return loss.sum()
# #     return loss
import torch
import torch.nn.functional as F
import torch.nn as nn

def focal_loss(
        y_prob: torch.Tensor,
        y_true: torch.Tensor,
        alpha: float = 0.75,
        gamma_pos: float = 2.0,  # Focusing factor for positives
        gamma_neg: float = 4.0,  # Focusing factor for negatives (hard negatives)
        reduction: str = "mean",
        ) -> torch.Tensor:
    """
    Asymmetric Focal Loss (AFL)
    
    - Applies different focusing factors for positive (gamma_pos) and negative (gamma_neg) samples.
    - Improves classification in highly imbalanced datasets.
    
    Args:
        y_prob (torch.Tensor): Model output logits (before sigmoid activation).
        y_true (torch.Tensor): Ground-truth binary labels (0 or 1).
        alpha (float, optional): Weighting factor for class imbalance (default: 0.75).
        gamma_pos (float, optional): Focusing factor for positive class (default: 2.0).
        gamma_neg (float, optional): Focusing factor for negative class (default: 4.0).
        reduction (str, optional): Reduction method - 'none', 'mean', or 'sum' (default: 'mean').
    
    Returns:
        torch.Tensor: Computed loss.
    """
    
    # Convert logits to probabilities
    y_prob = torch.sigmoid(y_prob)  # Convert raw logits to probability

    # Binary Cross Entropy Loss (without reduction)
    ce_loss = F.binary_cross_entropy(y_prob, y_true, reduction="none")
    
    # Compute p_t (probability assigned to the true class)
    p_t = y_true * y_prob + (1 - y_true) * (1 - y_prob)
    
    # Apply asymmetric focusing
    gamma_t = gamma_pos * y_true + gamma_neg * (1 - y_true)  # Different gamma for pos & neg
    loss = ce_loss * ((1 - p_t) ** gamma_t)

    # Apply class weighting (alpha)
    if alpha >= 0:
        alpha_t = alpha * y_true + (1 - alpha) * (1 - y_true)
        loss = alpha_t * loss

    # Reduction strategy
    if reduction == "none":
        pass
    elif reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()
    else:
        raise ValueError(f"Invalid Value for arg 'reduction': '{reduction}'\n Supported modes: 'none', 'mean', 'sum'")
    
    return loss
