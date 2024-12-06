import torch
import torch.nn as nn
import torch.nn.functional as F


from utils.utils import (
    get_close_joint_dist, 
    get_penetrate_dist)


def l2_loss(pred, gt):
    l2_loss_cal = nn.MSELoss()
    return l2_loss_cal(pred, gt)

def smooth_l1_loss(pred, true):
    return F.smooth_l1_loss(pred, true)

def binary_cross_entropy_loss(pred, gt):
    return F.binary_cross_entropy_with_logits(pred, gt)

class Loss:
    def __init__(self) -> None:
        self.smooth_l1_loss = F.smooth_l1_loss

criterion = Loss()

def dice_loss(pred, gt):
    """
    loss = \frac{2\sum^N_{i=1}y_i\hat{y}_i}{\sum^N_{i=1}y_i + \sum^N_{i=1}\hat{y}_i}
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


def dm_loss(lhand_dist_map_diff, rhand_dist_map_diff):
    return torch.sum((lhand_dist_map_diff + rhand_dist_map_diff))


def orient_loss(
    lhand_diff_pred, lhand_diff,
    rhand_diff_pred, rhand_diff,
    lhand_mask, rhand_mask
    ):
    lhand_mask = lhand_mask.unsqueeze(-1)
    rhand_mask = rhand_mask.unsqueeze(-1)

    return l2_norm(lhand_mask * (lhand_diff_pred - lhand_diff)) + \
           l2_norm(rhand_mask * (rhand_diff_pred - rhand_diff))

def l2_norm(x):
    return torch.norm(x, p=2)


def refine_loss(
    ref_motion_lhand, gt_motion_lhand,
    ref_motion_rhand, gt_motion_rhand
    ):
    return l2_norm(ref_motion_lhand - gt_motion_lhand) + \
           l2_norm(ref_motion_rhand - gt_motion_rhand)


def penetrate_loss(
    point_cloud, 
    verts_lhand, faces_lhand, mask_lhand,
    verts_rhand, faces_rhand, mask_rhand,
    frame_mask
    ):
    penet_dist_sum_lhand = get_penetrate_dist(
        point_cloud=point_cloud, 
        hand_verts=verts_lhand, hand_faces=faces_lhand,
        hand_mask=mask_lhand, frame_mask=frame_mask)
    penet_dist_sum_rhand = get_penetrate_dist(
        point_cloud=point_cloud, 
        hand_verts=verts_rhand, hand_faces=faces_rhand,
        hand_mask=mask_rhand, frame_mask=frame_mask)
    return l2_norm(penet_dist_sum_lhand) + l2_norm(penet_dist_sum_rhand)


def contect_loss(
    point_cloud, 
    joint_lhand, mask_lhand,
    joint_rhand, mask_rhand,
    frame_mask,
    tau
    ):
    joint_dist_sum_lhand = get_close_joint_dist(
        point_cloud=point_cloud, 
        hand_joint=joint_lhand,
        hand_mask=mask_lhand, frame_mask=frame_mask,
        tau=tau)
    joint_dist_sum_rhand = get_close_joint_dist(
        point_cloud=point_cloud, 
        hand_joint=joint_rhand,
        hand_mask=mask_rhand, frame_mask=frame_mask,
        tau=tau)
    return l2_norm(joint_dist_sum_lhand) + l2_norm(joint_dist_sum_rhand)