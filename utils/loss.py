import torch
import torch.nn as nn
import torch.nn.functional as F


def binary_cross_entropy_loss(pred, gt):
    """
    INPUT:
        pred: predict result, [B, ...]
        gt: ground true, [B, ...]
    RETURN:
        binary cross entropy loss
    """
    return F.binary_cross_entropy_with_logits(pred, gt)


def dice_loss(pred, gt):
    """
    loss = \frac{2\sum^N_{i=1}y_i\hat{y}_i}{\sum^N_{i=1}y_i + \sum^N_{i=1}\hat{y}_i}
    INPUT:
        pred: predict result, [B, ...]
        gt: ground true, [B, ...]
    RETURN:
        dice loss
    """
    smooth = 1.
    B = pred.shape[0]
    term1, term2 = pred.view(B, -1), gt.view(B, -1) 
    intersection = torch.sum(term1 * term2, dim=-1)
 
    batch_loss = 1 - (2. * intersection + smooth) / (torch.sum(term1, dim=-1) + torch.sum(term2, dim=-1) + smooth)
    return torch.sum(batch_loss) / B


def kl_divergence_loss(mu, log_var):
    B = mu.shape[0]
    return 0.5 * torch.sum(torch.exp(log_var) + torch.pow(mu, 2) - 1. - log_var) / B


