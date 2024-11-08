import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import tqdm

from utils.arguments import CFGS

class DiffusionModel(nn.Module):
    # https://zhuanlan.zhihu.com/p/617895786
    def __init__(self,
                 cfgs,
                 device,
                 denoise_model,
                 schedule_name="linear_beta_schedule",
                 timesteps=1000,
                 beta_start=0.0001,
                 beta_end=0.02):
        super(DiffusionModel, self).__init__()

        self.device = device
        self.denoise_model = denoise_model
        self.timesteps = timesteps

        beta_schedule_dict = {'linear_beta_schedule': self.__linear_beta_schedule,
                              'cosine_beta_schedule': self.__cosine_beta_schedule}

        if schedule_name in beta_schedule_dict:
            self.variance_schedule_func = beta_schedule_dict[schedule_name]
        else:
            raise ValueError('Function not found in dictionary')

        
        self.params = []
        # parameters for lhand, rhand and obj
        for i in range(3):
            self.params.append(self.__generate_params())


    def __generate_params(self):
        betas = self.variance_schedule_func(self.timesteps)
        # define alphas
        alphas = 1. - betas
        # \bar{alpha_t} = \prod^{t}_{i=1} alpha_i
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
        # calculations for diffusion q(x_t | x_{t-1}) and others
        # x_t  = sqrt(alphas_cumprod)*x_0 + sqrt(1 - alphas_cumprod)*z_t
        sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
        # calculations for posterior q(x_{t-1} | x_t, x_0)
        # 这里用的不是简化后的方差而是算出来的
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)


        param_dict = {
            "betas": betas, 
            "alphas": alphas, 
            "alphas_cumprod": alphas_cumprod, 
            "alphas_cumprod_prev": alphas_cumprod_prev, 
            "sqrt_recip_alphas": sqrt_recip_alphas,
            "sqrt_alphas_cumprod": sqrt_alphas_cumprod, 
            "sqrt_one_minus_alphas_cumprod": sqrt_one_minus_alphas_cumprod, 
            "posterior_variance": posterior_variance
        }

        return param_dict


    def __cosine_beta_schedule(self, timesteps, s=0.008):
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)


    def __linear_beta_schedule(self, timesteps, beta_start=0.0001, beta_end=0.02):
        return torch.linspace(beta_start, beta_end, timesteps)

    
    def __extract(self, a, t, x_shape):    
        B = t.shape[0]
        out = a.to(t.device).gather(0, t).float()
        out = out.reshape(B, *((1,) * (len(x_shape) - 1)))
        return out


    def q_sample(self, 
                 x_lhand, x_rhand, x_obj, 
                 noise_t,
                 lhand_noise, rhand_noise, obj_noise):
        # forward diffusion (using the nice property)
        # x_t  = sqrt(alphas_cumprod)*x_0 + sqrt(1 - alphas_cumprod)*z_t
        lhand_lterm = self.__extract(self.params[0]["sqrt_alphas_cumprod"], noise_t, x_lhand.shape)
        lhand_rterm = self.__extract(self.params[0]["sqrt_one_minus_alphas_cumprod"], noise_t, x_lhand.shape)
        lhand_noise = lhand_lterm * x_lhand + lhand_rterm * lhand_noise

        rhand_lterm = self.__extract(self.params[1]["sqrt_alphas_cumprod"], noise_t, x_rhand.shape)
        rhand_rterm = self.__extract(self.params[1]["sqrt_one_minus_alphas_cumprod"], noise_t, x_rhand.shape)
        rhand_noise = rhand_lterm * x_rhand + rhand_rterm * rhand_noise

        obj_lterm = self.__extract(self.params[2]["sqrt_alphas_cumprod"], noise_t, x_obj.shape)
        obj_rterm = self.__extract(self.params[2]["sqrt_one_minus_alphas_cumprod"], noise_t, x_obj.shape)
        obj_noise = obj_lterm * x_obj + obj_rterm * obj_noise

        return lhand_noise, rhand_noise, obj_noise


    def compute(self, 
                     x_lhand, x_rhand, x_obj, objs_feat, timesteps, text_feat):
        # Generate noise for x_lhand, x_rhand and x_obj
        lhand_noise = torch.randn_like(x_lhand)
        rhand_noise = torch.randn_like(x_rhand)
        obj_noise = torch.randn_like(x_obj)

        x_lhand_noisy, x_rhand_noisy, x_obj_noisy = self.q_sample(x_lhand, x_rhand, x_obj, 
                                                                  timesteps,
                                                                  lhand_noise, rhand_noise, obj_noise)
        lhand_pred, rhand_pred, obj_pred = self.denoise_model(x_lhand_noisy, x_rhand_noisy, x_obj_noisy,
                                                              objs_feat, timesteps, text_feat)
        
        return (lhand_pred, rhand_pred, obj_pred), \
               (lhand_noise, rhand_noise, obj_noise)

    @torch.no_grad()
    def p_sample(self, x, t, t_index):
        betas_t = self.__extract(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = self.__extract(
            self.sqrt_one_minus_alphas_cumprod, t, x.shape
        )
        sqrt_recip_alphas_t = self.__extract(self.sqrt_recip_alphas, t, x.shape)

        # Equation 11 in the paper
        # Use our model (noise predictor) to predict the mean
        model_mean = sqrt_recip_alphas_t * (
                x - betas_t * self.denoise_model(x, t) / sqrt_one_minus_alphas_cumprod_t
        )

        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = self.__extract(self.posterior_variance, t, x.shape)
            noise = torch.randn_like(x)
            # Algorithm 2 line 4:
            return model_mean + torch.sqrt(posterior_variance_t) * noise

    @torch.no_grad()
    def p_sample_loop(self, shape):
        device = next(self.denoise_model.parameters()).device

        b = shape[0]
        # start from pure noise (for each example in the batch)
        img = torch.randn(shape, device=device)
        imgs = []

        for i in tqdm(reversed(range(0, self.timesteps)), desc='sampling loop time step', total=self.timesteps):
            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long), i)
            imgs.append(img.cpu().numpy())
        return imgs

    @torch.no_grad()
    def sample(self, objs_feat, text_feat, hand_motion_dim, obj_motion_dim):

        B = objs_feat.shape[0]

        lhand_motion = torch.randn(
            [B, self.cfgs.max_frame, hand_motion_dim]).to(self.device)
        rhand_motion = torch.randn(
            [B, self.cfgs.max_frame, hand_motion_dim]).to(self.device)
        obj_motion = torch.randn(
            [B, self.cfgs.max_frame, obj_motion_dim]).to(self.device)

        return self.p_sample_loop(shape=(batch_size, channels, image_size, image_size))

    def forward(self, 
                x_lhand, x_rhand, x_obj, objs_feat, timesteps, text_feat, 
                training):
        if training:
            return self.compute(x_lhand, x_rhand, x_obj, objs_feat, timesteps, text_feat)
        else:
            return self.sample(objs_feat, text_feat)
