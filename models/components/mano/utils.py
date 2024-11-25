import smplx
import torch
from torch.nn import functional as F

from utils.rotation import rot6d_to_rotvec
from utils.utils import reshape_mesh_form


def mano_layer(
    model_path: str,
    batch_size: int,
    is_rhand=False,
    n_comps=45
    ):
    """
    Return MANO model for left/right hand.
    - Params:
        model_path: str
            Directory that keeps MANO .pkl models.
        is_rhand: bool
            Load left or right hand model, boolean,
        batch_size: int
            Training / inferencing batch size.
        n_comps: int
            Number of PCA components.
    - Returns:
        MANO model: nn.Module
    """
    model_type = 'MANO_RIGHT.pkl' if is_rhand else 'MANO_LEFT.pkl'

    return smplx.create(model_path=model_path,
        model_type=model_type,
        is_rhand=is_rhand,
        num_pca_comps=n_comps,
        batch_size=batch_size,
        flat_hand_mean=True)



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
        rot6d_to_rotvec(
            hand_motion[:, 3:99].reshape(-1, 6)
            ).reshape(batch_size*frame_len, -1)
        ], dim=-1)

    mano_result = mano_hand_layer(
        betas=torch.rand((batch_size*frame_len, 10)).to(device)*0.1,
        global_orient=hand_motion_axis_angle[:, 3:6],
        hand_pose=hand_motion_axis_angle[:, 6:51],
        transl=hand_motion_axis_angle[:, 0:3],
        return_verts=True,
        return_full_pose=True
        )
    
    # hand_mesh = reshape_mesh_form(mano_hand_layer.hand_meshes(mano_result), device)

    hand_verts = mano_result["vertices"].reshape(batch_size, frame_len, 778, 3)
    hand_faces = torch.from_numpy(mano_hand_layer.faces.reshape(1538, 3)).to(device)
    # edit body_models.py
    hand_joint = mano_result["joints"].reshape(batch_size, frame_len, 21, 3)

    hand_motion_mask = hand_motion_mask.reshape(batch_size, 1, 1, 1)
    frame_padding_mask = frame_padding_mask.reshape(batch_size, frame_len, 1, 1)

    hand_verts = hand_verts * hand_motion_mask * frame_padding_mask
    # hand_faces = hand_faces * hand_motion_mask * frame_padding_mask
    hand_joint = hand_joint * hand_motion_mask * frame_padding_mask


    return {"hand_verts": hand_verts.float(), "hand_faces": hand_faces, "hand_joint": hand_joint}
        