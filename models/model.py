import torch
import torch.nn as nn
from torch.nn import functional as F

from models.components.mano.utils import get_mano_result, mano_layer
from models.contact_map_generator import ContactMapGenerator
from models.frame_len_predictor import FrameLenPredictor
from models.hand_refiner import HandRefiner
from models.motion_generator import MotionGenerator
from utils.utils import get_deformed_obj_point_cloud, get_frame_mask, get_hand_motion_mask


class THOI(nn.Module):
    def __init__(self,
            DEVICE,
            CFGS,
            lhand_mano_layer,
            rhand_mano_layer
        ):
        super(THOI, self).__init__()

        self.DEVICE = DEVICE
        self.CFGS = CFGS

        # MANO layers for left and right hands.
        self.lhand_mano_layer = lhand_mano_layer
        self.rhand_mano_layer = rhand_mano_layer

        self.MODELS = {
            # Contact map generator.
            'contact_map_generator': ContactMapGenerator(CFGS, DEVICE).to(DEVICE),
            # Motion frame length predictor.
            'frame_len_predictor': FrameLenPredictor(DEVICE).to(DEVICE),
            # Motion generator.
            'motion_generator': MotionGenerator(CFGS, DEVICE).to(DEVICE),
            # Hand refinement.
            'hand_refinement_network': HandRefiner(CFGS, DEVICE).to(DEVICE),
        }

    
    def forward(self, 
            data, inferencing
        ):

        B = len(data["prompt"])

        contact_map_generator_result = self.MODELS["contact_map_generator"](
            data["prompt"], 
            data["obj_verts"], 
            inference=False, 
            contact_map=data["contact_map"])
        
        # contact_map_generator_result = clone_detach_dict_tensor(contact_map_generator_result)
        
        # Predict motion frame length.
        pred_frame_len = self.MODELS["frame_len_predictor"](
            contact_map_generator_result["text_feature"])
        # Ground-truth motion frame length.
        pred_frame_len = pred_frame_len.long()


        # Get left and right hand type masks base on input prompt.
        hand_motion_mask = get_hand_motion_mask(
            contact_map_generator_result["text_feature"], 
            self.DEVICE)
        # Get frame masks base to mask padding area.
        pred_frame_mask = get_frame_mask(B, self.CFGS.max_frame, pred_frame_len, self.DEVICE)

        obj_feat = torch.cat([
            contact_map_generator_result["global_feature"], 
            contact_map_generator_result["ref_contact_map"].squeeze(dim=2), 
            contact_map_generator_result["object_scale"].unsqueeze(dim=1)], dim=1)
        
        # Contact Motion Generator results
        pred_motion_lhand, pred_motion_rhand, pred_motion_obj = self.MODELS["motion_generator"](
            obj_feat, 
            contact_map_generator_result["text_feature"], 
            self.CFGS.max_frame,
            False,
            hand_motion_mask, pred_frame_mask,
            data["lh_motion"], 
            data["rh_motion"], 
            data["obj_motion"])
        
        # Deformed object point cloud based on ground-truth object motion
        ref_point_cloud = get_deformed_obj_point_cloud(
            data["obj_motion"], 
            contact_map_generator_result["point_cloud"])
        # Deformed object point cloud based on predict object motion
        ref_point_cloud_pred = get_deformed_obj_point_cloud(
            pred_motion_obj, 
            contact_map_generator_result["point_cloud"])
        
        pred_mano_lhand = get_mano_result(
            pred_motion_lhand, self.lhand_mano_layer, hand_motion_mask["mask_lhand"], pred_frame_mask,
            self.CFGS.max_frame, B, self.DEVICE)

        pred_mano_rhand = get_mano_result(
            pred_motion_rhand, self.rhand_mano_layer, hand_motion_mask["mask_rhand"], pred_frame_mask,
            self.CFGS.max_frame, B, self.DEVICE)


        ref_motion_lhand, ref_motion_rhand = self.MODELS["hand_refinement_network"](
            pred_motion_lhand, pred_motion_rhand, 
            pred_mano_lhand["hand_joint"], pred_mano_rhand["hand_joint"], 
            contact_map_generator_result["ref_contact_map"].unsqueeze(dim=2), 
            ref_point_cloud_pred,
            hand_motion_mask, pred_frame_mask
        )

        ref_mano_lhand = get_mano_result(
            ref_motion_lhand,
            self.lhand_mano_layer,
            hand_motion_mask["mask_lhand"],
            pred_frame_mask,
            self.CFGS.max_frame, B, self.DEVICE)

        ref_mano_rhand = get_mano_result(
            ref_motion_rhand,
            self.rhand_mano_layer,
            hand_motion_mask["mask_rhand"],
            pred_frame_mask,
            self.CFGS.max_frame, B, self.DEVICE)
        
        return contact_map_generator_result, pred_frame_len, hand_motion_mask, pred_frame_mask, pred_motion_lhand, pred_motion_rhand, pred_motion_obj, \
            ref_point_cloud_pred, pred_mano_lhand, pred_mano_rhand, ref_motion_lhand, ref_motion_rhand, ref_mano_lhand, ref_mano_rhand