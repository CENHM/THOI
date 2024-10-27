import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from models.components.Clip import Clip
from models.components.PointNet import PointNet

from utils.arguments import CFGS


class ContactMapGeneration(nn.Module):
    def __init__(self, device):
        super(ContactMapGeneration, self).__init__()

        self.clip_text_encoder = Clip(device=device)
        self.pointnet = PointNet(init_k=3, device=device)
        self.contact_encoder = ContactEncoder(device=device)
        self.contact_decoder = ContactDecoder()

        self.device = device


    def __compute_mesh_scale(self, vertices):
        centroid = torch.mean(vertices, dim=1)
        centroid = centroid.unsqueeze(1)
        s_obj, _ = torch.max(torch.sqrt(torch.sum(torch.square(vertices - centroid), dim=2)), 1)
        # `s_obj` is also the maximum distance from the center of object mesh to its vertices.
        return s_obj

    def __farthest_point_sample(self, xyz, npoint=CFGS.fps_npoint):
        """
        Input:
            xyz: pointcloud data, [B, N, 3]
            npoint: number of samples
        Return:
            centroids: sampled pointcloud index, [B, npoint]
        """
        
        B, N, C = xyz.shape

        centroids = torch.zeros(B, npoint).long().to(self.device)
        distance = torch.ones(B, N) * 1e10
        distance = distance.to(self.device)
        farthest = torch.randint(0, N, (B,)).long().to(self.device)
        batch_indices = torch.arange(B).long().to(self.device)
        for i in range(npoint):
            centroids[:, i] = farthest
            centroid = xyz[batch_indices, farthest, :].reshape(B, 1, 3)
            dist = torch.sum(torch.square(xyz - centroid), -1)
            mask = dist < distance
            distance[mask] = dist[mask]
            farthest = torch.max(distance, -1)[1]
        return centroids
    
    def forward(self, text, mesh, contact_map):
        B = mesh.shape[0]

        object_scale = self.__compute_mesh_scale(mesh)

        point_cloud_idx = self.__farthest_point_sample(mesh)
        batch_point_cloud_idx = torch.arange(B).view(-1, 1).expand(-1, point_cloud_idx.shape[1])
        point_cloud = mesh[batch_point_cloud_idx, point_cloud_idx]

        point_cloud_norm = point_cloud / object_scale
        local_feature, global_feature = self.pointnet(point_cloud_norm)

        text_feature = self.clip_text_encoder.extract_feature(text)

        contact_map = contact_map[batch_point_cloud_idx, point_cloud_idx]
        if not CFGS.testing:
            point_cloud_norm_contact = torch.cat([contact_map, point_cloud_norm], dim=2)
            contact_vec = self.contact_encoder(point_cloud_norm_contact)
        else:
            contact_vec = torch.randn(B, 64)

        object_scale = object_scale.float()
        global_feature = global_feature.float()
        text_feature = text_feature.float()
        contact_vec = contact_vec.float()

        object_scale_reshape = object_scale.reshape(B, 1, 1).repeat(1, CFGS.fps_npoint, 1)
        global_feature_reshape = global_feature.unsqueeze(1).repeat(1, CFGS.fps_npoint, 1)
        text_feature_reshape = text_feature.unsqueeze(1).repeat(1, CFGS.fps_npoint, 1)
        contact_vec_reshape = contact_vec.unsqueeze(1).repeat(1, CFGS.fps_npoint, 1)

        concatenate_feature = torch.cat([global_feature_reshape, 
                                         local_feature, 
                                         text_feature_reshape, 
                                         object_scale_reshape, 
                                         contact_vec_reshape], dim=2)
        
        refine_contact_map = self.contact_decoder(concatenate_feature)

        return refine_contact_map, (global_feature, object_scale, text_feature)
    

class ContactEncoder(nn.Module):
    def __init__(self, device=None):
        super(ContactEncoder, self).__init__()

        self.pointnet_structure = PointNet(init_k=4, local_feat=False, device=device)

        self.fc1 = torch.nn.Linear(1024, 512)
        self.fc2 = torch.nn.Linear(512, 256)
        self.fc3 = torch.nn.Linear(256, 128)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)
        
    def __reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x): # B, N, D
        x = self.pointnet_structure(x)

        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)

        mean, var = torch.chunk(x, 2, dim=1)
        x = self.__reparameterize(mean, var)

        return x
        

class ContactDecoder(nn.Module):
    def __init__(self):
        super(ContactDecoder, self).__init__()

        self.fc1 = torch.nn.Linear(1665, 512)
        self.fc2 = torch.nn.Linear(512, 256)
        self.fc3 = torch.nn.Linear(256, 128)
        self.fc4 = torch.nn.Linear(128, 1)

    def forward(self, x): # B, N, D
        x = F.leaky_relu(self.fc1(x), negative_slope=0.2)
        x = F.leaky_relu(self.fc2(x), negative_slope=0.2)
        x = F.leaky_relu(self.fc3(x), negative_slope=0.2)
        x = self.fc4(x)

        return x
