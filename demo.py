import torch

from utils.arguments import CFGS
from utils.loss import (
    binary_cross_entropy_loss, 
    dice_loss,
    joint_loss, 
    l2_loss, 
    dm_loss, 
    or_loss, 
    kl_divergence_loss,
    penetrate_loss,
    refine_loss,)
from utils.utils import (
    get_deformed_obj_point_cloud,
    get_hand_motion_mask, 
    get_padding_mask, 
    frame_len_original, 
    estimated_distance_maps,
    relative_3d_orientation_diff, 
    align_frame_for_component,)
from utils.models import (
    get_mano_result,
    mano_layer,
    frame_len_prediction,)

from models.contact_map_generator import ContactMapGenerator
from models.motion_generator import MotionGenerator
from models.hand_refiner import HandRefiner

# ==============================================================================

from demo_utils.params import DEVICE, inference, text, B, mesh, contact_map, gt_motion

# ==============================================================================

normal_repr = torch.Tensor.__repr__
torch.Tensor.__repr__ = lambda self: f"{self.shape}_{normal_repr(self)}"

# ==============================================================================

lhand_mano_model = mano_layer(CFGS.mano_model_path, False, B).to(DEVICE)
rhand_mano_model = mano_layer(CFGS.mano_model_path, True, B).to(DEVICE)


contact_map_generator = ContactMapGenerator(DEVICE, CFGS).to(DEVICE)


if not inference:
    contact_map_generator_result = contact_map_generator(text, mesh, inference, contact_map)

    contact_map_BCE_loss = binary_cross_entropy_loss(
        contact_map_generator_result["ref_contact_map"], 
        contact_map_generator_result["sample_contact_map"])
    
    contact_map_dice_loss = dice_loss(
        contact_map_generator_result["ref_contact_map"], 
        contact_map_generator_result["sample_contact_map"])

    Reparameterization_loss = kl_divergence_loss(
        contact_map_generator_result["mu"],
        contact_map_generator_result["log_var"])
else:
    contact_map_generator_result = contact_map_generator(text, mesh, inference, contact_map)

# ==============================================================================

after_pad_len = CFGS.max_frame

pred_frame_len = frame_len_prediction(
    contact_map_generator_result["text_feature"],
    CFGS.max_frame, B, DEVICE)

orig_frame_len = frame_len_original(gt_motion, DEVICE)

hand_motion_mask = get_hand_motion_mask(
    contact_map_generator_result["text_feature"], 
    DEVICE)

frame_mask, frame_mask_seq, frame_mask_seq_cond = get_padding_mask(
    batch_size=B, 
    frame_len=after_pad_len, 
    orig_frame_len=orig_frame_len, 
    pred_frame_len=pred_frame_len, 
    device=DEVICE)


motion_generator = MotionGenerator(
    cfgs=CFGS,
    device=DEVICE,
    hand_motion_mask=hand_motion_mask,
    frame_padding_mask=frame_mask_seq_cond,
    ).to(DEVICE)


obj_feat = torch.cat([
    contact_map_generator_result["global_feature"], 
    contact_map_generator_result["ref_contact_map"].squeeze(dim=2), 
    contact_map_generator_result["object_scale"].unsqueeze(dim=1)], dim=1)


if not inference:
    gt_motion_lhand = align_frame_for_component(
        x=gt_motion["lhand_motion"], x_type="lhand_motion", 
        frame_len=after_pad_len, 
        device=DEVICE) * hand_motion_mask["mask_lhand"]
    
    gt_motion_rhand = align_frame_for_component(
        x=gt_motion["rhand_motion"], x_type="rhand_motion", 
        frame_len=after_pad_len, 
        device=DEVICE) * hand_motion_mask["mask_lhand"]
    
    gt_motion_obj = align_frame_for_component(
        x=gt_motion["obj_motion"], x_type="obj_motion", 
        frame_len=after_pad_len, 
        device=DEVICE)
    
    pred_motion_lhand, pred_motion_rhand, pred_motion_obj = motion_generator(
        obj_feat=obj_feat, 
        text_feat=contact_map_generator_result["text_feature"], 
        frame_len=after_pad_len,
        inference=inference,
        motion_lhand=gt_motion_lhand, 
        motion_rhand=gt_motion_rhand, 
        motion_obj=gt_motion_obj)
    
    gt_mano_lhand = get_mano_result(
        gt_motion_lhand,
        lhand_mano_model,
        hand_motion_mask=hand_motion_mask["mask_lhand"],
        frame_padding_mask=frame_mask,
        frame_len=after_pad_len, batch_size=B, device=DEVICE)
    
    gt_mano_rhand = get_mano_result(
        gt_motion_rhand,
        rhand_mano_model,
        hand_motion_mask=hand_motion_mask["mask_rhand"],
        frame_padding_mask=frame_mask,
        frame_len=after_pad_len, batch_size=B, device=DEVICE)

else:
    pred_motion_lhand, pred_motion_rhand, pred_motion_obj = motion_generator(
        obj_feat=obj_feat, 
        text_feat=contact_map_generator_result["text_feature"], 
        frame_len=pred_frame_len,
        inference=inference,)
    

pred_mano_lhand = get_mano_result(
    pred_motion_lhand,
    lhand_mano_model,
    hand_motion_mask=hand_motion_mask["mask_lhand"],
    frame_padding_mask=frame_mask,
    frame_len=after_pad_len, batch_size=B, device=DEVICE)

pred_mano_rhand = get_mano_result(
    pred_motion_rhand,
    rhand_mano_model,
    hand_motion_mask=hand_motion_mask["mask_rhand"],
    frame_padding_mask=frame_mask,
    frame_len=after_pad_len, batch_size=B, device=DEVICE)
    

ref_point_cloud_pred = get_deformed_obj_point_cloud(
    pred_motion_obj, 
    contact_map_generator_result["point_cloud"])

if not inference:
    ref_point_cloud = get_deformed_obj_point_cloud(
        gt_motion_obj, 
        contact_map_generator_result["point_cloud"])
    
    ddpm_denoise_loss = \
        l2_loss(pred_motion_lhand, gt_motion_lhand) + \
        l2_loss(pred_motion_rhand, gt_motion_rhand) + \
        l2_loss(pred_motion_obj, gt_motion_obj)

    lhand_dist_map_diff = estimated_distance_maps(
        deformed_obj_point_cloud_pred=ref_point_cloud_pred, 
        deformed_obj_point_cloud=ref_point_cloud, 
        pred_joint=pred_mano_lhand["hand_joint"], 
        gt_joint=gt_mano_lhand["hand_joint"],
        hand_mask=hand_motion_mask["mask_lhand"], frame_mask=frame_mask,
        tau=CFGS.tau)

    rhand_dist_map_diff = estimated_distance_maps(
        deformed_obj_point_cloud_pred=ref_point_cloud_pred, 
        deformed_obj_point_cloud=ref_point_cloud, 
        pred_joint=pred_mano_rhand["hand_joint"], 
        gt_joint=gt_mano_rhand["hand_joint"],
        hand_mask=hand_motion_mask["mask_rhand"], frame_mask=frame_mask,
        tau=CFGS.tau)
    
    dist_map_loss = dm_loss(lhand_dist_map_diff, rhand_dist_map_diff)

    lhand_relative_orientation_diff_pred = relative_3d_orientation_diff(
        pred_motion_lhand, pred_motion_obj,
        hand_mask=hand_motion_mask["mask_lhand"], frame_mask=frame_mask)

    lhand_relative_orientation_diff = relative_3d_orientation_diff(
        gt_motion_lhand, gt_motion_obj,
        hand_mask=hand_motion_mask["mask_lhand"], frame_mask=frame_mask)

    rhand_relative_orientation_diff_pred = relative_3d_orientation_diff(
        pred_motion_rhand, pred_motion_obj,
        hand_mask=hand_motion_mask["mask_rhand"], frame_mask=frame_mask)

    rhand_relative_orientation_diff = relative_3d_orientation_diff(
        pred_motion_rhand, pred_motion_obj,
        hand_mask=hand_motion_mask["mask_rhand"], frame_mask=frame_mask)

    orientation_loss = or_loss(
        lhand_diff_pred=lhand_relative_orientation_diff_pred, 
        lhand_diff=lhand_relative_orientation_diff,
        rhand_diff_pred=rhand_relative_orientation_diff_pred, 
        rhand_diff=rhand_relative_orientation_diff,
        lhand_mask=hand_motion_mask["mask_lhand"], 
        rhand_mask=hand_motion_mask["mask_rhand"]
    )

# ==============================================================================

hand_refinement_network = HandRefiner(
    cfgs=CFGS,
    device=DEVICE,
    hand_motion_mask=hand_motion_mask,
    frame_padding_mask=frame_mask_seq,
    ).to(DEVICE)

ref_motion_lhand, ref_motion_rhand = hand_refinement_network(
    pred_motion_lhand, pred_motion_rhand, 
    pred_mano_lhand["hand_joint"], pred_mano_rhand["hand_joint"], 
    contact_map_generator_result["ref_contact_map"].unsqueeze(dim=2), 
    ref_point_cloud_pred
)

ref_mano_lhand = get_mano_result(
    ref_motion_lhand,
    lhand_mano_model,
    hand_motion_mask=hand_motion_mask["mask_lhand"],
    frame_padding_mask=frame_mask,
    frame_len=after_pad_len, batch_size=B, device=DEVICE)

ref_mano_rhand = get_mano_result(
    ref_motion_rhand,
    rhand_mano_model,
    hand_motion_mask=hand_motion_mask["mask_rhand"],
    frame_padding_mask=frame_mask,
    frame_len=after_pad_len, batch_size=B, device=DEVICE)

if not inference:
    refine_hand_motions_loss = refine_loss(
        ref_motion_lhand, gt_motion_lhand, 
        ref_motion_rhand, gt_motion_rhand)

    ref_penet_loss = penetrate_loss(
        ref_point_cloud_pred, 
        ref_mano_lhand["hand_verts"], 
        ref_mano_lhand["hand_faces"], 
        hand_motion_mask["mask_lhand"],
        ref_mano_rhand["hand_verts"], 
        ref_mano_rhand["hand_verts"], 
        hand_motion_mask["mask_rhand"],
        frame_mask
    )

    ref_joint_loss = joint_loss(
        ref_point_cloud_pred, 
        ref_mano_lhand["hand_joint"], 
        hand_motion_mask["mask_lhand"],
        ref_mano_rhand["hand_joint"], 
        hand_motion_mask["mask_rhand"],
        frame_mask,
        CFGS.tau
    )

pass