import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

from models.components.clip import Clip
from utils.rotation import rot6d_to_rotmat


def params_to_device(params, device):
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in params.items()}


def params_to_torch(params, dtype=torch.float32, **kwargs):
    if "start" in kwargs and "end" in kwargs:
        return {k: torch.from_numpy(v[kwargs['start']:kwargs['end']]).type(dtype) for k, v in params.items()}
    return {k: torch.from_numpy(v).type(dtype) for k, v in params.items()}


def get_hand_motion_mask(text_feat, device):
    '''
    Mask inputs which are not belonging to the estimated hand type.
    - Returns:
        dict, {"mask_lhand":[B, 1, 1], "mask_rhand":[B, 1, 1]}
            Mask for left and right hand.
    '''
    clip_text_encoder = Clip(device=device)
    cos_similarity_cal = nn.CosineSimilarity(dim=2, eps=1e-6)
    prompts = ["the right hand.", "the left hand.", "both hands."]
    prompts = ["A photo of " + prompt for prompt in prompts]

    prompt_features = clip_text_encoder.extract_feature(prompts).unsqueeze(0).repeat(4, 1, 1)
    text_feat_expanded = text_feat.unsqueeze(dim=1).expand(-1, 3, -1)
    cos_similarities = cos_similarity_cal(text_feat_expanded, prompt_features)
    h_star_idx = cos_similarities.argmax(dim=1)

    B = text_feat.shape[0]
    hand_motion_mask = torch.ones((B, 2, )).float().to(device)
    mask_indices = (h_star_idx == 0) | (h_star_idx == 1)
    hand_motion_mask[torch.arange(B, device=device)[mask_indices], h_star_idx[mask_indices]] = 0

    mask_lhand = hand_motion_mask[:, 0].reshape(-1, 1, 1)
    mask_rhand = hand_motion_mask[:, 1].reshape(-1, 1, 1)

    return {"mask_lhand": mask_lhand, "mask_rhand": mask_rhand}




def align_frame(x_dict, device):

    x_list = next(iter(x_dict.values()))
    # L_max = 0
    # for sample in x_list:
    #     L_curr = sample.shape[1]
    #     L_max = L_curr if L_curr > L_max else L_max
    L_max = 150

    B = len(x_list)

    x_pad_dict = {}

    for key, value in x_dict.items():
        if key == "lhand_motion" or key == "rhand_motion":
            x_final = torch.zeros((B, L_max, 99)).float().to(device)
        elif key == "obj_motion":
            x_final = torch.zeros((B, L_max, 10)).float().to(device)  
        elif key == "j_lh" or key == "j_rh":
             x_final = torch.zeros((B, L_max, 21, 3)).float().to(device)  
        for i, data in enumerate(value):
            L_curr = data.shape[1]
            if L_curr == 0:
                continue
            x_final[i, :L_curr] = data
        x_pad_dict[key] = x_final
    return x_pad_dict, L_max


def align_param_size(data_dict, max_len):
    data_aligned_dict = {}
    for key, value in data_dict.items():
        if key == "lh_motion" or key == "rh_motion":
            data_final = torch.zeros((max_len, 99)).float()
        elif key == "obj_motion":
            data_final = torch.zeros((max_len, 10)).float()
        data_final[:value.shape[0]] = value
        data_aligned_dict[key] = data_final
    return data_aligned_dict
 


def align_frame_for_component(x, x_type, frame_len, device):
    B = len(x)
    
    type_dict = {
        "lhand_motion": torch.zeros((B, frame_len, 99)).float().to(device),
        "rhand_motion": torch.zeros((B, frame_len, 99)).float().to(device),
        "obj_motion": torch.zeros((B, frame_len, 10)).float().to(device),
        "lhand_joint": torch.zeros((B, frame_len, 21, 3)).float().to(device),
        "rhand_joint": torch.zeros((B, frame_len, 21, 3)).float().to(device),
        "hand_motion_mask": torch.zeros((B, frame_len, 2)).bool().to(device),
    }

    x_final = type_dict[x_type]

    if isinstance(x, list):
        for i, data in enumerate(x):
            L_curr = data.shape[1]
            assert L_curr > 0
            x_final[i, :L_curr] = data
    else:
        x = x.unsqueeze(dim=1)
        x_final = x.repeat(1, frame_len, 1)
    
    return x_final



def frame_len_original(x_dict, device):
    '''
    :RETURN
        [B, 1]
    '''
    x = x_dict[list(x_dict.keys())[0]]
    B = len(x)
    original_frame_len = torch.full((B, ), 0, dtype=torch.int64).to(device)
    for idx, sample in enumerate(x):
        original_frame_len[idx] = sample.shape[1]
    return original_frame_len.unsqueeze(1)



def get_padding_mask( 
    batch_size, 
    frame_len, 
    true_frame_len, 
    pred_frame_len, 
    device,
    ):
    '''
    :RETURN
        [B, L, 1], [B, 2 * L, 1], [B, 3 * L + 1, 1]
    '''

    frame_padding_mask = torch.arange(frame_len).expand(batch_size, frame_len).to(device) >= true_frame_len
    frame_padding_mask = frame_padding_mask

    pred_framelen_mask = torch.arange(frame_len).expand(batch_size, frame_len).to(device) >= pred_frame_len
    pred_framelen_mask = pred_framelen_mask

    # frame_mask = frame_padding_mask | pred_framelen_mask
    frame_mask = pred_framelen_mask


    frame_mask_seq_cond = torch.full((batch_size, 3*frame_len), True).bool().to(device)
    frame_mask_seq_cond[:, 0::3] = frame_mask
    frame_mask_seq_cond[:, 1::3] = frame_mask
    frame_mask_seq_cond[:, 2::3] = frame_mask
    frame_mask_seq_cond = torch.cat((torch.full((batch_size, 1), False).bool().to(device), frame_mask_seq_cond), dim=1)
    
    frame_mask_seq = torch.full((batch_size, 2*frame_len), True).bool().to(device)
    frame_mask_seq[:, 0::2] = frame_mask
    frame_mask_seq[:, 1::2] = frame_mask

    frame_mask = torch.where(~frame_mask, 1.0, 0.0)

    return frame_mask.unsqueeze(-1), frame_mask_seq.unsqueeze(-1), frame_mask_seq_cond.unsqueeze(-1)


def get_frame_mask( 
    batch_size, 
    target_frame_len, 
    frame_len, 
    device,
    ):
    '''
    :RETURN
        [B, L, 1], [B, 2 * L, 1], [B, 3 * L + 1, 1]
    '''

    # Frame mask 
    frame_mask = \
        torch.arange(target_frame_len).expand(batch_size, target_frame_len).to(device) >= frame_len
    
    # frame_mask_seq_wo_obj = torch.full((batch_size, 2*target_frame_len), True).bool().to(device)
    # frame_mask_seq_wo_obj[:, 0::2] = frame_mask
    # frame_mask_seq_wo_obj[:, 1::2] = frame_mask

    # frame_mask_seq_w_obj = torch.full((batch_size, 3*target_frame_len), True).bool().to(device)
    # frame_mask_seq_w_obj[:, 0::3] = frame_mask
    # frame_mask_seq_w_obj[:, 1::3] = frame_mask
    # frame_mask_seq_w_obj[:, 2::3] = frame_mask
    # frame_mask_seq_cond = torch.cat((torch.full((batch_size, 1), False).bool().to(device), frame_mask_seq_cond), dim=1)
    
    frame_mask = torch.where(~frame_mask, 1.0, 0.0)
    # frame_mask_seq_wo_obj = torch.where(~frame_mask_seq_wo_obj, 1.0, 0.0)
    # frame_mask_seq_w_obj = torch.where(~frame_mask_seq_w_obj, 1.0, 0.0)


    return frame_mask.unsqueeze(-1)



def get_joint_obj_dist_map(hand_joints, point_cloud):
    return torch.cdist(hand_joints, point_cloud)


def get_deformed_obj_point_cloud(obj_motion, point_cloud):
    """
    /lib/utils/proc_output.py -> get_transformed_obj_pc
    INPUT:
        obj_motion: obj motion, [B, L, 10]
        point_cloud: object's point cloud, [B, N, 3]
    RETURN:
        d: 3D displacement between hand joints and the nearest object points in point cloud, [B, L, hand_joint, 3]
    """
    B, L = obj_motion.shape[:2]

    obj_trans = obj_motion[..., 0:3]
    obj_rot6d = obj_motion[..., 3:9]
    obj_rotmat = rot6d_to_rotmat(obj_rot6d).reshape(B, L, 3, 3)

    obj_pc_rotated = torch.einsum("btij,bkj->btki", obj_rotmat, point_cloud)
    obj_pc_transformed = obj_pc_rotated + obj_trans.unsqueeze(2)
    return obj_pc_transformed


def estimated_distance_maps(
    deformed_obj_point_cloud_pred,
    deformed_obj_point_cloud,
    pred_joint, true_joint,
    hand_mask, 
    true_frame_mask, pred_frame_mask,
    tau
    ):

    norm = deformed_obj_point_cloud_pred.shape[-3] * deformed_obj_point_cloud_pred.shape[-2] * pred_joint.shape[-2]

    pred_dist_map = get_joint_obj_dist_map(pred_joint, deformed_obj_point_cloud_pred)
    true_dist_map = get_joint_obj_dist_map(true_joint, deformed_obj_point_cloud)
    threadhold = (true_dist_map < tau)

    hand_mask = hand_mask.unsqueeze(-1)

    pred_dist_map = pred_dist_map * pred_frame_mask.unsqueeze(-1)
    true_dist_map = true_dist_map * pred_frame_mask.unsqueeze(-1)

    dist_map_diff = torch.pow((pred_dist_map - true_dist_map) * threadhold, 2) * hand_mask

    return dist_map_diff / norm


def relative_3d_orientation_diff(
    hand_motion, obj_motion, 
    hand_mask, frame_mask
    ):

    hand_3d_rotation = rot6d_to_rotmat(hand_motion[:, :, 3:9])
    obj_3d_rotation = rot6d_to_rotmat(obj_motion[:, :, 3:9])

    diff = torch.einsum("ijkm,ijmn->ijkn", [hand_3d_rotation, obj_3d_rotation])

    hand_mask = hand_mask.unsqueeze(-1)
    frame_mask = frame_mask.unsqueeze(-1)

    diff = diff * hand_mask * frame_mask
    return diff


def reshape_mesh_form(meshs, device):
    verts, faces = [], []
    for i in range(len(meshs)):
        verts.append(meshs[i].vertices)
        faces.append(meshs[i].faces)
    verts = torch.tensor(np.array(verts)).to(device)
    faces = torch.tensor(np.array(faces)).to(device)
    return {"verts": verts, "faces": faces}


def knn_points(src_xyz, trg_xyz, K=1):
    '''
    :param 
        src_xyz: [B, N1, 3] - Source Mesh vertices coordinate
        trg_xyz: [B, N2, 3] - Target Mesh vertices coordinate
        K: K-nearest neighbor
    :return
        dists: nearest distance from each target to source point
        idx: index
    '''
    dists = torch.cdist(trg_xyz, src_xyz)  # [B, N2, N1]
    min_dists, idx = torch.topk(dists, K, dim=-1, largest=False, sorted=True)
    del dists
    return min_dists, idx # [B, N2, K], [B, N2, K]


def get_nearest_neighbor(src_xyz, trg_xyz):
    '''
    :param 
        src_xyz: [B, N1, 3] - Source Mesh vertices coordinate
        trg_xyz: [B, N2, 3] - Target Mesh vertices coordinate
    :return
        dists: nearest distance from each target to source point
        idx: index 
    '''
    dists, idx = knn_points(src_xyz, trg_xyz, K=1)  # [dists, idx]
    return dists, idx


def batched_index_select(input, index, dim=1):
    '''
    :PARAMS 
        input: [B, N1, *]
        index: [B, N2]
        dim: the dim to be selected 
    :RETURN
        [B, N2, *] - selected result
    '''
    views = [input.size(0)] + [1 if i != dim else -1 for i in range(1, len(input.shape))]
    expanse = list(input.shape)
    expanse[0] = -1
    expanse[dim] = -1
    index = index.view(views).expand(expanse)
    return torch.gather(input, dim=dim, index=index)


def compute_face_norms(verts, faces):
    '''
    Compute the normal vector of the hand mesh face.
    :PARAMS 
        verts: Hand mesh vertices coordinate xyz, [B, V, 3]
        faces: Hand mesh faces (vertices index), [1, T, 3]
    :RETURN
        vert_norm_vecs: Normal vector of each vertices, [B, V, 3]
    '''
    B = verts.shape[0]
    faces = faces.repeat(B, 1, 1).long()

    v0 = verts.gather(1, faces[:, :, 0:1].expand(-1, -1, 3)).float()  # [B, F, 3] 
    v1 = verts.gather(1, faces[:, :, 1:2].expand(-1, -1, 3)).float()  # [B, F, 3] 
    v2 = verts.gather(1, faces[:, :, 2:3].expand(-1, -1, 3)).float()  # [B, F, 3] 
    
    edge1 = v1 - v0
    edge2 = v2 - v0
    
    face_norm_vecs = torch.cross(edge1, edge2, dim=-1)  # [B, F, 3]
    face_norm_vecs = face_norm_vecs / face_norm_vecs.norm(dim=-1, keepdim=True)

    return face_norm_vecs


def compute_vert_norms(verts, faces):
    '''
    Compute the normal vector of the hand mesh vertices.
    :PARAMS 
        verts: Hand mesh vertices coordinate xyz, [B, V, 3]
        faces: Hand mesh faces (vertices index), [1, T, 3]
    :RETURN
        vert_norm_vecs: Normal vector of each vertices, [B, V, 3]
    '''
    B, V, _ = verts.shape
    _, T, _ = faces.shape
    
    face_norms = compute_face_norms(verts, faces)  # [B, P, 3]
    faces = faces.long()
    vert_norm_vecs = torch.zeros(B, V, 3, device=verts.device).float()  # [B, V, 3]
    vert_norm_vecs.scatter_add_(dim=1, index=faces, src=face_norms)  # [B, V, 3]
    vert_norm_vecs = vert_norm_vecs / torch.norm(vert_norm_vecs, dim=2, keepdim=True)

    return vert_norm_vecs


def get_interior_verts(hand_vert_norms, hand_verts_xyz, obj_xyz, nearest_idx):
    '''
    :PARAMS:
        src_face_normal: surface normal of every vert in the source mesh
    :param src_face_normal: [B, 778, 3], 
    :param src_xyz: [B, 778, 3], source mesh vertices xyz
    :param trg_xyz: [B, 3000, 3], target mesh vertices xyz
    :param trg_NN_idx: [B, 3000], index of NN in source vertices from target vertices
    :return: interior [B, 3000], inter-penetrated trg vertices as 1, instead 0 (bool)
    '''
    nearest_vert2obj_xyz = batched_index_select(hand_verts_xyz, nearest_idx) 
    obj2hand_vec = nearest_vert2obj_xyz - obj_xyz

    nearest_vert2obj_norms = batched_index_select(hand_vert_norms, nearest_idx)

    # interior as true, exterior as false
    interior = (obj2hand_vec * nearest_vert2obj_norms).sum(dim=-1) > 0  
    return interior


def get_penetrate_dist(
    point_cloud, 
    hand_verts, hand_faces,
    hand_mask, frame_mask
    ):
    '''
    Get distance between hand vertices that penetrate the object surface and 
        point cloud.
    :PARAMS 
        point_cloud: Point cloud, [B, L, N, 3]
        hand_verts: Hand mesh vertices index xyz, [B, L, V, 3]
    :RETURN
        vert_norm_vecs: Normal vector of each vertices, [B, V, 3]
    '''
    B, L = point_cloud.shape[0], hand_verts.shape[1]
    N, V, T = point_cloud.shape[2], hand_verts.shape[2], hand_faces.shape[0]
    
    point_cloud = point_cloud.reshape(-1, N, 3)
    hand_verts = hand_verts.reshape(-1, V, 3)
    hand_faces = hand_faces.reshape(-1, T, 3)

    hand_normal = compute_vert_norms(hand_verts, hand_faces)
    dist, idx = get_nearest_neighbor(hand_verts, point_cloud)
    interior = get_interior_verts(hand_normal, hand_verts, point_cloud, idx).bool()

    penetrate_dist = (dist * interior.unsqueeze(-1)).sum(dim=1)
    penetrate_dist = penetrate_dist.reshape(B, L, -1) *  hand_mask * frame_mask

    return penetrate_dist


def get_close_joint_dist(
    point_cloud, 
    hand_joint,
    hand_mask, frame_mask,
    tau
    ):
    B, L = point_cloud.shape[0], hand_joint.shape[1]
    N, J = point_cloud.shape[2], hand_joint.shape[2]
    
    point_cloud = point_cloud.reshape(-1, N, 3)
    hand_joint = hand_joint.reshape(-1, J, 3)

    dist, _ = get_nearest_neighbor(point_cloud, hand_joint)
    threshold_mask = dist <= tau 
    dist = (dist * threshold_mask).sum(dim=1)
    dist = dist.reshape(B, L, -1) * hand_mask * frame_mask

    return dist.sum()

def clone_detach_dict_tensor(d: dict):
    return {k: v.clone().detach() for k, v in d.items()}



