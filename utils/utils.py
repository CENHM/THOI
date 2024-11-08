import torch
import torch.nn as nn
from torch.nn import functional as F

from models.components.Clip import Clip


def hand_type_selection(text_feat, device):
    clip_text_encoder = Clip(device=device)
    cos_similarity_cal = nn.CosineSimilarity(dim=2, eps=1e-6)
    prompts = ["right hand", "left hand", "both hands"]

    prompt_features = clip_text_encoder.extract_feature(prompts)
    text_feat_expanded = text_feat.expand(-1, 3, -1)
    cos_similarities = cos_similarity_cal(text_feat_expanded, prompt_features)
    h_star_idx = cos_similarities.argmax(dim=1)
    return h_star_idx


def rot6d_to_rotmat(x):
    """Convert 6D rotation representation to 3x3 rotation matrix.
    Based on Zhou et al., "On the Continuity of Rotation Representations in Neural Networks", CVPR 2019
    Input:
        (B,6) Batch of 6-D rotation representations
    Output:
        (B,3,3) Batch of corresponding rotation matrices
    """
    x = x.reshape(-1, 3, 2)
    a1 = x[:, :, 0]
    a2 = x[:, :, 1]
    b1 = F.normalize(a1)
    b2 = F.normalize(a2 - torch.einsum("bi,bi->b", b1, a2).unsqueeze(-1) * b1)
    b3 = torch.cross(b1, b2)
    return torch.stack((b1, b2, b3), dim=-1)