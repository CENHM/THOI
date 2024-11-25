import torch
from torch.utils.data import DataLoader

import tqdm

from models.contact_map_generator import ContactMapGenerator
from models.frame_len_predictor import FrameLenPredictor
from models.hand_refiner import HandRefiner
from models.motion_generator import MotionGenerator
from models.components.mano.utils import get_mano_result, mano_layer

from utils.loss import (
    binary_cross_entropy_loss, 
    dice_loss, 
    dm_loss, 
    contect_loss, 
    kl_divergence_loss, 
    l2_loss, 
    orient_loss, 
    penetrate_loss, 
    refine_loss)
from utils.utils import (
    clone_detach_dict_tensor, 
    estimated_distance_maps, 
    get_deformed_obj_point_cloud,
    get_frame_mask, 
    get_hand_motion_mask, 
    get_padding_mask, 
    params_to_device, 
    relative_3d_orientation_diff)

from utils.arguments import CFGS
from utils.code import (
    ADD_OPTIMIZERS, 
    INITIALIZER, 
    MODELS_SET_MODE, 
    MODELS_SET_ZERO_GRAD, 
    OPTIMIZER_STEP, 
    log, 
    SAVE_LOG)
from utils.code import LOAD_WEIGHT, SAVE_CHECKPOINT, LOAD_CHECKPOINT
from datasets.dataset import Dataset




def tests():
    global DEVICE, MODELS, DATASET, DATALOADER
    global lhand_mano_layer, rhand_mano_layer

    MODELS = MODELS_SET_MODE(MODELS, train=False)
    
    loader_len = len(DATALOADER) 
    loader_iter = iter(DATALOADER)

    with torch.no_grad():
        for batch_idx in tqdm.tqdm(range(loader_len)):
            data = next(loader_iter)
            data = params_to_device(data, DEVICE)

            B = len(data["prompt"])

            MODELS = MODELS_SET_ZERO_GRAD(MODELS)

            # Contact Map Generator results
            contact_map_generator_result = MODELS["contact_map_generator"](
                data["prompt"], 
                data["obj_verts"], 
                inference=False, 
                contact_map=data["contact_map"])
            
            # Predict motion frame length.
            pred_frame_len = MODELS["frame_len_predictor"](
                contact_map_generator_result["text_feature"])
            pred_frame_len = pred_frame_len.long()

            # Get left and right hand type masks base on input prompt.
            hand_motion_mask = get_hand_motion_mask(
                contact_map_generator_result["text_feature"], 
                DEVICE)
            # Get frame masks base to mask padding area.
            pred_frame_mask = get_frame_mask(B, CFGS.max_frame, pred_frame_len, DEVICE)

            obj_feat = torch.cat([
                contact_map_generator_result["global_feature"], 
                contact_map_generator_result["ref_contact_map"].squeeze(dim=2), 
                contact_map_generator_result["object_scale"].unsqueeze(dim=1)], dim=1)
            
            # Contact Motion Generator results
            pred_motion_lhand, pred_motion_rhand, pred_motion_obj = MODELS["motion_generator"](
                obj_feat, 
                contact_map_generator_result["text_feature"], 
                CFGS.max_frame,
                True,
                hand_motion_mask, pred_frame_mask)

            # Deformed object point cloud based on predict object motion
            ref_point_cloud_pred = get_deformed_obj_point_cloud(
                pred_motion_obj, 
                contact_map_generator_result["point_cloud"])
            
            pred_mano_lhand = get_mano_result(
                pred_motion_lhand, lhand_mano_layer, hand_motion_mask["mask_lhand"], pred_frame_mask,
                CFGS.max_frame, B, DEVICE)

            pred_mano_rhand = get_mano_result(
                pred_motion_rhand, rhand_mano_layer, hand_motion_mask["mask_rhand"], pred_frame_mask,
                CFGS.max_frame, B, DEVICE)


            ref_motion_lhand, ref_motion_rhand = MODELS["hand_refinement_network"](
                pred_motion_lhand, pred_motion_rhand, 
                pred_mano_lhand["hand_joint"], pred_mano_rhand["hand_joint"], 
                contact_map_generator_result["ref_contact_map"].unsqueeze(dim=2), 
                ref_point_cloud_pred,
                hand_motion_mask, pred_frame_mask
            )

            ref_mano_lhand = get_mano_result(
                ref_motion_lhand,
                lhand_mano_layer,
                hand_motion_mask["mask_lhand"],
                pred_frame_mask,
                CFGS.max_frame, B, DEVICE)

            ref_mano_rhand = get_mano_result(
                ref_motion_rhand,
                rhand_mano_layer,
                hand_motion_mask["mask_rhand"],
                pred_frame_mask,
                CFGS.max_frame, B, DEVICE)

    SAVE_LOG()
        


def train_one_epoch(epoch):
    global DEVICE, MODELS, DATASET, DATALOADER, OPTIMIZERS
    global lhand_mano_layer, rhand_mano_layer

    MODELS = MODELS_SET_MODE(MODELS, train=True)
    
    loader_len = len(DATALOADER) 
    loader_iter = iter(DATALOADER)

    for batch_idx in tqdm.tqdm(range(loader_len)):

        data = next(loader_iter)
        data = params_to_device(data, DEVICE)

        B = len(data["prompt"])

        MODELS = MODELS_SET_ZERO_GRAD(MODELS)

        # Contact Map Generator results
        contact_map_generator_result = MODELS["contact_map_generator"](
            data["prompt"], 
            data["obj_verts"], 
            inference=False, 
            contact_map=data["contact_map"])
        
        contact_map_BCE_loss = binary_cross_entropy_loss(
            contact_map_generator_result["ref_contact_map"], 
            contact_map_generator_result["sample_contact_map"])
        
        contact_map_dice_loss = dice_loss(
            contact_map_generator_result["ref_contact_map"], 
            contact_map_generator_result["sample_contact_map"])
        
        Reparameterization_loss = kl_divergence_loss(
            contact_map_generator_result["mu"],
            contact_map_generator_result["log_var"])
        # Loss of Contact Map Generator
        loss_contact_map_generator = contact_map_BCE_loss + contact_map_dice_loss + Reparameterization_loss
        
        contact_map_generator_result = clone_detach_dict_tensor(contact_map_generator_result)
        
        # Predict motion frame length.
        pred_frame_len = MODELS["frame_len_predictor"](
            contact_map_generator_result["text_feature"])
        # Ground-truth motion frame length.
        true_frame_len = data["nframe"]
        # Loss of Frame Length Predictor
        loss_frame_len_predictor = l2_loss(pred_frame_len, true_frame_len.float())
        pred_frame_len = pred_frame_len.long()

        # Get left and right hand type masks base on input prompt.
        hand_motion_mask = get_hand_motion_mask(
            contact_map_generator_result["text_feature"], 
            DEVICE)
        # Get frame masks base to mask padding area.
        true_frame_mask = get_frame_mask(B, CFGS.max_frame, true_frame_len, DEVICE)
        pred_frame_mask = get_frame_mask(B, CFGS.max_frame, pred_frame_len, DEVICE)

        obj_feat = torch.cat([
            contact_map_generator_result["global_feature"], 
            contact_map_generator_result["ref_contact_map"].squeeze(dim=2), 
            contact_map_generator_result["object_scale"].unsqueeze(dim=1)], dim=1)
        
        
        # Contact Motion Generator results
        pred_motion_lhand, pred_motion_rhand, pred_motion_obj = MODELS["motion_generator"](
            obj_feat, 
            contact_map_generator_result["text_feature"], 
            CFGS.max_frame,
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
        
        ddpm_denoise_loss = \
            l2_loss(pred_motion_lhand, data["lh_motion"]) + \
            l2_loss(pred_motion_rhand, data["rh_motion"]) + \
            l2_loss(pred_motion_obj, data["obj_motion"])
        
        true_mano_lhand = get_mano_result(
            data["lh_motion"], lhand_mano_layer, hand_motion_mask["mask_lhand"], true_frame_mask,
            CFGS.max_frame, B, DEVICE)
    
        true_mano_rhand = get_mano_result(
            data["rh_motion"], rhand_mano_layer, hand_motion_mask["mask_rhand"], true_frame_mask,
            CFGS.max_frame, B, DEVICE)
        
        pred_mano_lhand = get_mano_result(
            pred_motion_lhand, lhand_mano_layer, hand_motion_mask["mask_lhand"], pred_frame_mask,
            CFGS.max_frame, B, DEVICE)

        pred_mano_rhand = get_mano_result(
            pred_motion_rhand, rhand_mano_layer, hand_motion_mask["mask_rhand"], pred_frame_mask,
            CFGS.max_frame, B, DEVICE)

        lhand_dist_map_diff = estimated_distance_maps(
            ref_point_cloud_pred, 
            ref_point_cloud, 
            pred_mano_lhand["hand_joint"], 
            true_mano_lhand["hand_joint"],
            hand_motion_mask["mask_lhand"], 
            true_frame_mask, pred_frame_mask,
            CFGS.tau)

        rhand_dist_map_diff = estimated_distance_maps(
            ref_point_cloud_pred, 
            ref_point_cloud, 
            pred_mano_rhand["hand_joint"], 
            true_mano_rhand["hand_joint"],
            hand_motion_mask["mask_rhand"], 
            true_frame_mask, pred_frame_mask,
            CFGS.tau)
        
        dist_map_loss = dm_loss(lhand_dist_map_diff, rhand_dist_map_diff)

        lhand_relative_orientation_diff_pred = relative_3d_orientation_diff(
            pred_motion_lhand, pred_motion_obj,
            hand_motion_mask["mask_lhand"], pred_frame_mask)

        lhand_relative_orientation_diff = relative_3d_orientation_diff(
            data["lh_motion"], data["obj_motion"],
            hand_motion_mask["mask_lhand"], true_frame_mask)

        rhand_relative_orientation_diff_pred = relative_3d_orientation_diff(
            pred_motion_rhand, pred_motion_obj,
            hand_motion_mask["mask_rhand"], pred_frame_mask)

        rhand_relative_orientation_diff = relative_3d_orientation_diff(
            data["rh_motion"], data["obj_motion"],
            hand_motion_mask["mask_rhand"], true_frame_mask)

        orientation_loss = orient_loss(
            lhand_relative_orientation_diff_pred, 
            lhand_relative_orientation_diff,
            rhand_relative_orientation_diff_pred, 
            rhand_relative_orientation_diff,
            hand_motion_mask["mask_lhand"], 
            hand_motion_mask["mask_rhand"]
        )

        # Loss of Motion Generator
        loss_motion_generator = ddpm_denoise_loss + dist_map_loss + orientation_loss


        pred_motion_lhand, pred_motion_rhand, pred_motion_obj = \
            pred_motion_lhand.clone().detach(), pred_motion_rhand.clone().detach(), pred_motion_obj.clone().detach()
        pred_mano_lhand = clone_detach_dict_tensor(pred_mano_lhand)
        pred_mano_rhand = clone_detach_dict_tensor(pred_mano_rhand)
        ref_point_cloud_pred = ref_point_cloud_pred.clone().detach()


        ref_motion_lhand, ref_motion_rhand = MODELS["hand_refinement_network"](
            pred_motion_lhand, pred_motion_rhand, 
            pred_mano_lhand["hand_joint"], pred_mano_rhand["hand_joint"], 
            contact_map_generator_result["ref_contact_map"].unsqueeze(dim=2), 
            ref_point_cloud_pred,
            hand_motion_mask, pred_frame_mask
        )

        ref_mano_lhand = get_mano_result(
            ref_motion_lhand,
            lhand_mano_layer,
            hand_motion_mask["mask_lhand"],
            pred_frame_mask,
            CFGS.max_frame, B, DEVICE)

        ref_mano_rhand = get_mano_result(
            ref_motion_rhand,
            rhand_mano_layer,
            hand_motion_mask["mask_rhand"],
            pred_frame_mask,
            CFGS.max_frame, B, DEVICE)

        refine_hand_motions_loss = refine_loss(
            ref_motion_lhand, data["lh_motion"], 
            ref_motion_rhand, data["rh_motion"])

        ref_penet_loss = penetrate_loss(
            ref_point_cloud_pred, 
            ref_mano_lhand["hand_verts"], 
            ref_mano_lhand["hand_faces"], 
            hand_motion_mask["mask_lhand"],
            ref_mano_rhand["hand_verts"], 
            ref_mano_rhand["hand_faces"], 
            hand_motion_mask["mask_rhand"],
            pred_frame_mask
        )

        ref_contect_loss = contect_loss(
            ref_point_cloud_pred, 
            ref_mano_lhand["hand_joint"], 
            hand_motion_mask["mask_lhand"],
            ref_mano_rhand["hand_joint"], 
            hand_motion_mask["mask_rhand"],
            pred_frame_mask,
            CFGS.tau
        )
        # Loss of Hand refiner
        loss_hand_refiner = refine_hand_motions_loss + ref_penet_loss + CFGS.contect_loss_lambda * ref_contect_loss

        loss_contact_map_generator.backward()
        loss_frame_len_predictor.backward()
        loss_motion_generator.backward()
        loss_hand_refiner.backward()

        OPTIMIZERS = OPTIMIZER_STEP(OPTIMIZERS)

        if batch_idx % 1 == 0:
            log(f"TRAIN - epoch: {epoch} - batch: {batch_idx + 1} / {loader_len} - " + 
                f"loss: {loss_contact_map_generator}; {loss_frame_len_predictor}; {loss_motion_generator}; {loss_hand_refiner}")


def train():
    global DEVICE, MODELS, OPTIMIZERS, START_EPOCH

    for epoch in tqdm.tqdm(range(START_EPOCH + 1, CFGS.epoch + 1)):
        train_one_epoch(epoch)
        SAVE_CHECKPOINT(epoch, OPTIMIZERS, MODELS)
        log(f'Epoch {epoch} done.')
        SAVE_LOG()
        pass


if __name__ == "__main__":
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # MANO layers for left and right hands.
    lhand_mano_layer = mano_layer(CFGS.mano_model_path, CFGS.batch_size, is_rhand=False).to(DEVICE)
    rhand_mano_layer = mano_layer(CFGS.mano_model_path, CFGS.batch_size, is_rhand=True).to(DEVICE)

    MODELS = {
        # Contact map generator.
        'contact_map_generator': ContactMapGenerator(CFGS, DEVICE).to(DEVICE),
        # Motion frame length predictor.
        'frame_len_predictor': FrameLenPredictor(DEVICE).to(DEVICE),
        # Motion generator.
        'motion_generator': MotionGenerator(CFGS, DEVICE).to(DEVICE),
        # Hand refinement.
        'hand_refinement_network': HandRefiner(CFGS, DEVICE).to(DEVICE),
    }

    START_EPOCH = 0

    if not CFGS.inferencing:
        log("training...")
        DATASET = Dataset(CFGS.dataset_dir, CFGS.dataset, maxfn=CFGS.max_frame, train=True)
        DATALOADER = DataLoader(DATASET, batch_size=CFGS.batch_size, shuffle=True, drop_last=True)
        OPTIMIZERS = ADD_OPTIMIZERS(MODELS)
        if CFGS.resume:
            START_EPOCH, OPTIMIZERS, MODELS = LOAD_CHECKPOINT(OPTIMIZERS, MODELS)
        train()
    else:
        log("testing...")
        DATASET = Dataset(CFGS.dataset_dir, CFGS.dataset, maxfn=CFGS.max_frame, train=False)
        DATALOADER = DataLoader(DATASET, batch_size=CFGS.batch_size, shuffle=True)
        MODELS = LOAD_WEIGHT(MODELS)
        tests()

    log("done.")
    SAVE_LOG()
        

    
