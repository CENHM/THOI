import torch
import torch.nn as nn

from models.components.TransformerModel import PositionalEncoding, TimestepEmbedding, TransformerEncoder

from utils.arguments import CFGS
from utils.utils import rot6d_to_rotmat



class HandRefinementNetwork(nn.Module):
    def __init__(self,
            obj_dim=10,
            hand_dim=99,
            hand_joint=21,
            hand_joint_dim=3,
            n_point=CFGS.fps_npoint,
            hidden_dim=512,
        ):
        super(HandRefinementNetwork, self).__init__()

        self.dim = hidden_dim

        self.transformer_model = TransformerModel(hidden_dim=hidden_dim)

        # 99 + 3J + N + N + 3J
        cat_dim = hand_dim + 2 * hand_joint * hand_joint_dim + n_point * 2
        
        self.fc_lhand = nn.Linear(in_features=cat_dim, out_features=hidden_dim)
        self.fc_rhand = nn.Linear(in_features=cat_dim, out_features=hidden_dim)

        self.fc_out_lhand = nn.Linear(in_features=hidden_dim, out_features=hand_dim)
        self.fc_out_rhand = nn.Linear(in_features=hidden_dim, out_features=hand_dim)

    def __get_transformed_obj_pc(self, x_obj, point_cloud, dataset=CFGS.dataset):
        """
        /lib/utils/proc_output.py -> get_transformed_obj_pc
        INPUT:
            hand_joints: hand joints, [B, L, hand_joint, 3]
            point_cloud: object's point cloud, [B, L, N, 3]
        RETURN:
            d: 3D displacement between hand joints and the nearest object points in point cloud, [B, L, hand_joint, 3]
        """
        B, L = x_obj.shape[:2]

        obj_trans = x_obj[..., 0:3]
        obj_rot6d = x_obj[..., 3:9]
        obj_rotmat = rot6d_to_rotmat(obj_rot6d).reshape(B, L, 3, 3)

        obj_pc_rotated = torch.einsum("btij,bkj->btki", obj_rotmat, point_cloud)
        obj_pc_transformed = obj_pc_rotated + obj_trans.unsqueeze(2)
        return obj_pc_transformed

    def __get_attention_map(self,
            hand_joints, point_cloud         
        ):
        """
        /lib/utils/proc.py -> get_hand2obj_dist
        INPUT:
            hand_joints: hand joints, [B, L, hand_joint, 3]
            point_cloud: object's point cloud, [B, L, N, 3]
        RETURN:
            d: 3D displacement between hand joints and the nearest object points in point cloud, [B, L, hand_joint, 3]
        """
        B, L = hand_joints.shape[:2]

        hand_joints = hand_joints.view(B*L, -1, 3)
        point_cloud = point_cloud.view(B*L, -1, 3)

        hand_joints_expanded = hand_joints.unsqueeze(2) 
        point_cloud_expanded = point_cloud.unsqueeze(1) 
        
        distances = torch.sum((hand_joints_expanded - point_cloud_expanded) ** 2, dim=3)
        hand_nn_idx = torch.argmin(distances, dim=2)
        hand_nn_idx_expand = hand_nn_idx.unsqueeze(-1).expand(B*L, -1, 3) 
        obj_pc_contact = torch.gather(point_cloud, 1, hand_nn_idx_expand)

        displacement = (hand_joints - obj_pc_contact)**2
        displacement = displacement.view(B, L, -1, 3)

        att_map = torch.exp(-50 * displacement)

        return att_map


    def forward(self, 
            x_lhand, x_rhand, j_lhand, j_rhand, m_contact, x_obj, point_cloud
        ):
        """
        INPUT:
            x_lhand: hand (left ) motion, [B, L, hand_dim]
            x_rhand: hand (right) motion, [B, L, hand_dim]
            j_lhand: hand (left ) joints, [B, L, hand_joint, 3]
            j_rhand: hand (right) joints, [B, L, hand_joint, 3]
            m_contact: contact map, [B, N, 1]
            x_obj: object discription, [B, L, obj_dim]
            point_cloud: object's point cloud, [B, L, N, 3]
        RETURN:
            x_lhand: hand (left ) motion, [B, L, hand_dim]
            x_rhand: hand (right) motion, [B, L, hand_dim]
        """

        B, L = x_lhand.shape[:2]

        ref_point_cloud = self.__get_transformed_obj_pc(x_obj, point_cloud) 
        att_map_lhand = self.__get_attention_map(j_lhand, ref_point_cloud)
        att_map_rhand = self.__get_attention_map(j_rhand, ref_point_cloud)
        
        j_lhand = j_lhand.view(B, L, -1)
        j_rhand = j_rhand.view(B, L, -1)

        m_contact = m_contact.view(B, 1, -1).repeat(1, L, 1)
        # "computation of the norm is applied across the last dimension"
        ref_point_cloud = ref_point_cloud.norm(dim=-1)

        att_map_lhand = att_map_lhand.view(B, L, -1)
        att_map_rhand = att_map_rhand.view(B, L, -1) 

        x_lhand = torch.cat([x_lhand, j_lhand, m_contact, ref_point_cloud, att_map_lhand], dim=2)
        x_rhand = torch.cat([x_rhand, j_rhand, m_contact, ref_point_cloud, att_map_rhand], dim=2)

        # hand input embedding layers
        x_lhand = self.fc_lhand(x_lhand)
        x_rhand = self.fc_rhand(x_rhand)
        
        x_lhand, x_rhand = self.transformer_model(x_lhand, x_rhand)

        # hand output embedding layers
        x_lhand = self.fc_out_lhand(x_lhand)
        x_rhand = self.fc_out_rhand(x_rhand)

        return x_lhand, x_rhand
    

class TransformerModel(nn.Module):
    def __init__(self,
            hidden_dim,
        ):
        super(TransformerModel, self).__init__()

        self.frame_wise_pos_encoder = PositionalEncoding(d_model=hidden_dim, comp="hrn", encode_mode="frame-wise")
        self.agent_wise_pos_encoder = PositionalEncoding(d_model=hidden_dim, comp="hrn", encode_mode="agent-wise")

        self.encoder = TransformerEncoder(d_model=hidden_dim)

    
    def forward(self, 
            x_lhand, x_rhand
        ):
        """
        INPUT:
            x_lhand: hand (left ) motion, [B, L, hidden_dim]
            x_rhand: hand (right) motion, [B, L, hidden_dim]
        RETURN:
            x_lhand: hand (left ) motion, [B, L, hidden_dim]
            x_rhand: hand (right) motion, [B, L, hidden_dim]
        """
        B, L = x_lhand.shape[0], x_lhand.shape[1]

        x = torch.stack((x_lhand, x_rhand), dim=2)
        x = x.view(B, 2*L, -1)

        x = self.frame_wise_pos_encoder(x)
        x = self.agent_wise_pos_encoder(x)

        x = self.encoder(x)

        x_lhand = x[:, 0::2]
        x_rhand = x[:, 1::2]

        return x_lhand, x_rhand
