import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from models.components.clip import Clip
from models.components.pointnet import PointNet
from models.components.linear import LinearLayers



class ContactMapGenerator(nn.Module):
    def __init__(self, 
        cfgs,
        device
        ):
        super(ContactMapGenerator, self).__init__()

        self.device = device
        self.npoints = cfgs.fps_npoint

        self.clip_text_encoder = Clip(device=device)
        self.pointnet = PointNet(init_k=3, device=device)
        self.contact_encoder = VAEEncoder(device)
        self.contact_decoder = LinearLayers(
            in_dim=1665, 
            layers_out_dim=[512, 256, 128, 1], 
            activation_func='leaky_relu', 
            activation_func_param=0.02, 
            bn=False
        )

    def __compute_mesh_scale(self, vertices):
        centroid = torch.mean(vertices, dim=1)
        centroid = centroid.unsqueeze(1)
        s_obj, _ = torch.max(torch.sqrt(torch.sum(torch.square(vertices - centroid), dim=2)), 1)
        # `s_obj` is also the maximum distance from the center of object mesh to its vertices.
        return s_obj

    def __farthest_point_sample(self, xyz):
        """
        PARAMS:
            xyz: pointcloud data, [B, N, 3]
        RETURN:
            centroids: sampled pointcloud index, [B, npoint]
        """
        
        B, N, C = xyz.shape

        centroids = torch.zeros(B, self.npoints).long().to(self.device)
        distance = torch.ones(B, N) * 1e10
        distance = distance.to(self.device)
        farthest = torch.randint(0, N, (B,)).long().to(self.device)
        batch_indices = torch.arange(B).long().to(self.device)
        for i in range(self.npoints):
            centroids[:, i] = farthest
            centroid = xyz[batch_indices, farthest, :].reshape(B, 1, 3)
            dist = torch.sum(torch.square(xyz - centroid), -1)
            mask = dist < distance
            distance[mask] = dist[mask]
            farthest = torch.max(distance, -1)[1]
        return centroids
    
    def forward(self, 
        prompt, mesh, 
        inference,
        contact_map=None
        ):
        B = mesh.shape[0]

        object_scale = self.__compute_mesh_scale(mesh)

        point_cloud_idx = self.__farthest_point_sample(mesh)
        batch_point_cloud_idx = torch.arange(B).view(-1, 1).expand(-1, point_cloud_idx.shape[1])
        point_cloud = mesh[batch_point_cloud_idx, point_cloud_idx]

        point_cloud_norm = point_cloud / object_scale.view(B, 1, 1)
        local_feature, global_feature = self.pointnet(point_cloud_norm)

        text_feature = self.clip_text_encoder.extract_feature(prompt).float()

        if not inference:
            sample_contact_map = contact_map[batch_point_cloud_idx, point_cloud_idx]
            point_cloud_norm_contact = torch.cat([sample_contact_map, point_cloud_norm], dim=2)
            contact_vec, mu, log_var = self.contact_encoder(point_cloud_norm_contact)
        else:
            contact_vec = torch.randn(B, 64).to(self.device)

        object_scale_reshape = object_scale.reshape(B, 1, 1).repeat(1, self.npoints, 1)
        global_feature_reshape = global_feature.unsqueeze(1).repeat(1, self.npoints, 1)
        text_feature_reshape = text_feature.unsqueeze(1).repeat(1, self.npoints, 1)
        contact_vec_reshape = contact_vec.unsqueeze(1).repeat(1, self.npoints, 1)

        concatenate_feature = torch.cat([global_feature_reshape, 
                                         local_feature, 
                                         text_feature_reshape, 
                                         object_scale_reshape, 
                                         contact_vec_reshape], dim=2)
        
        refine_contact_map = self.contact_decoder(concatenate_feature)

        return_dict = {
            "ref_contact_map": refine_contact_map,
            "global_feature": global_feature, 
            "object_scale": object_scale, 
            "text_feature": text_feature,
            "point_cloud": point_cloud
        }

        if not inference:
            return_dict["sample_contact_map"] = sample_contact_map
            return_dict["mu"] = mu
            return_dict["log_var"] = log_var

        return return_dict
        
    

class VAEEncoder(nn.Module):
    def __init__(self, 
        device
        ):
        super(VAEEncoder, self).__init__()

        self.pointnet_structure = PointNet(init_k=4, local_feat=False, device=device)

        self.mlp = LinearLayers(
            in_dim=1024, 
            layers_out_dim=[512, 256, 128], 
            activation_func='relu',
            bn=True
        )
        
    def __reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x): # [B, N, D]
        x = self.pointnet_structure(x)

        x = self.mlp(x)

        mu, log_var = torch.chunk(x, 2, dim=1)
        x = self.__reparameterize(mu, log_var)

        return x, mu, log_var
        

