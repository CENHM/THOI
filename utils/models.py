import torch
from torch.nn import functional as F

import models.components.mano.mano.model as mano
from models.components.linear import LinearLayers
from utils.rotation import rot6d_to_axis_angle
from utils.utils import reshape_mesh_form


def mano_layer(
    model_path: str,
    is_rhand: bool,
    batch_size: int,
    ):
    """
    Return MANO model for left/right hand.
    PARAMS:
        model_path: str
            Directory that keeps MANO .pkl models.
        is_rhand: bool
            Load left or right hand model, boolean,
        batch_size: int
            Training / inferencing batch size.
    RETURN:
        MANO model: nn.Module
    """
    return mano.load(
        model_path=model_path,
        is_rhand=is_rhand,
        num_pca_comps=45,
        batch_size=batch_size,
        flat_hand_mean=False
        )


def frame_len_prediction(
    text_feature,
    max_frame,
    batch_size,
    device,
    text_feature_dim=512,
    rand_dim=64
    ):
    '''
    :RETURN
        [B, 1]
    '''

    frame_len_predictor = LinearLayers(
        in_dim=text_feature_dim+rand_dim,
        layers_out_dim=[512, 256, 128, 1], 
        activation_func='leaky_relu', 
        activation_func_param=0.02, 
        bn=False,
        sigmoid_output=True).to(device)
    z = torch.randn((batch_size, rand_dim)).to(device)
    return (frame_len_predictor(torch.cat([z, text_feature], dim=1)) * max_frame).long()


def get_mano_result(
    hand_motion,
    mano_hand_layer,
    hand_motion_mask,
    frame_padding_mask,
    frame_len,
    batch_size,
    device
    ):
    hand_motion = hand_motion.reshape(batch_size*frame_len, -1) 

    hand_motion_axis_angle = torch.cat([
        hand_motion[:, :3], 
        rot6d_to_axis_angle(
            hand_motion[:, 3:99].reshape(-1, 6)
            ).reshape(batch_size*frame_len, -1)
        ], dim=-1)

    mano_result = mano_hand_layer(
        betas=torch.rand((batch_size*frame_len, 10)).to(device)*0.1,
        global_orient=hand_motion_axis_angle[:, 3:6],
        hand_pose=hand_motion_axis_angle[:, 6:51],
        transl=hand_motion_axis_angle[:, 0:3],
        return_verts=True,
        return_tips=True
        )
    
    hand_mesh = reshape_mesh_form(mano_hand_layer.hand_meshes(mano_result), device)

    hand_verts = hand_mesh["verts"].reshape(batch_size, frame_len, 778, 3)
    hand_faces = hand_mesh["faces"].reshape(batch_size, frame_len, 1538, 3)
    hand_joint = mano_result.joints.reshape(batch_size, frame_len, 21, 3)

    hand_motion_mask = hand_motion_mask.reshape(batch_size, 1, 1, 1)
    frame_padding_mask = frame_padding_mask.reshape(batch_size, frame_len, 1, 1)

    hand_verts = hand_verts * hand_motion_mask * frame_padding_mask
    hand_faces = hand_faces * hand_motion_mask * frame_padding_mask
    hand_joint = hand_joint * hand_motion_mask * frame_padding_mask


    return {"hand_verts": hand_verts.float(), "hand_faces": hand_faces.long(), "hand_joint": hand_joint}
        