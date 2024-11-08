import tqdm
import math

import torch
import torch.nn as nn

def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return torch.tensor(betas)


class DDPM(nn.Module):
    def __init__(
            self, 
            device,
            denoise_model,
            T,
            beta_1=1e-4, 
            beta_T=0.02, 
            schedule_name="cosine", 
        ):
        super().__init__()

        
        self.denoise_model = denoise_model
        self.device = device

        self.schedule_name = schedule_name
        self.beta_1 = beta_1
        self.beta_T = beta_T
        self.T = T

        self.params = self.__generate_params()

    def __generate_params(self):
        if self.schedule_name == "cosine":
            betas = betas_for_alpha_bar(self.T, lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,)
        else:
            betas = torch.linspace(start = self.beta_1, end=self.beta_T, steps=self.T)
        alphas = 1. - betas
        # \bar{alpha_t} = \prod^{t}_{i=1} alpha_i
        alpha_bars = torch.cumprod(alphas, dim=0)
        alpha_bars_prev = torch.cat([torch.Tensor([1]).float(), alpha_bars[:-1]])

        posterior_variance = betas * (1. - alpha_bars_prev) / (1. - alpha_bars)
        
        posterior_log_variance_clip = torch.log(
            torch.hstack([posterior_variance[1], posterior_variance[1:]])
        )

        param_dict = {
            "betas": betas, 
            "alphas": alphas, 
            "alpha_bars": alpha_bars, 
            "alpha_bars_prev": alpha_bars_prev,
            "posterior_log_variance_clip": posterior_log_variance_clip
        }

        return param_dict

    def forward(
        self, 
        lhand_motion, rhand_motion, obj_motion,         
        obj_feat, text_feat,
        timesteps=None
        ):

        B = lhand_motion.shape[0]

        if timesteps == None:
            timesteps = torch.randint(0, self.T, (B, )).to(self.device)
            alpha_bars = self.params["alpha_bars"][timesteps][:, None, None]

            epsilon_lhand = torch.randn_like(lhand_motion)
            x_tilde_lhand = torch.sqrt(alpha_bars) * lhand_motion + torch.sqrt(1 - alpha_bars) * epsilon_lhand
            epsilon_rhand = torch.randn_like(rhand_motion)
            x_tilde_rhand = torch.sqrt(alpha_bars) * rhand_motion + torch.sqrt(1 - alpha_bars) * epsilon_rhand
            epsilon_obj = torch.randn_like(obj_motion)
            x_tilde_obj = torch.sqrt(alpha_bars) * obj_motion + torch.sqrt(1 - alpha_bars) * epsilon_obj
            
        else:
            timesteps = torch.Tensor(
                [timesteps for _ in range(B)]).to(self.device).long()
            x_tilde_lhand = lhand_motion
            x_tilde_rhand = rhand_motion
            x_tilde_obj = obj_motion

        pred_X0_lhand, pred_X0_rhand, pred_X0_obj = self.denoise_model(
            x_tilde_lhand, x_tilde_rhand, x_tilde_obj, 
            obj_feat, text_feat, 
            timesteps
        )

        return pred_X0_lhand, pred_X0_rhand, pred_X0_obj
    
    @torch.no_grad()
    def sampling(
        self, objs_feat, text_feat, hand_motion_dim, obj_motion_dim
        ):

        B = objs_feat.shape[0]

        lhand_motion = torch.randn(
            [B, self.cfgs.max_frame, hand_motion_dim]).to(self.device)
        rhand_motion = torch.randn(
            [B, self.cfgs.max_frame, hand_motion_dim]).to(self.device)
        obj_motion = torch.randn(
            [B, self.cfgs.max_frame, obj_motion_dim]).to(self.device)

        sample_lhand_motion, sample_rhand_motion, sample_obj_motion = self.ddpm_loop(
            lhand_motion, rhand_motion, obj_motion, objs_feat, text_feat 
        )

        return sample_lhand_motion, sample_rhand_motion, sample_obj_motion
    
    def ddpm_loop(
        self, lhand_motion, rhand_motion, obj_motion, obj_feat, text_feat  
        ):

        for t in tqdm.tqdm(
            reversed(range(self.T)), desc="ddpm denoise looping", total=self.T
            ):
            noise_lhand_motion = torch.zeros_like(lhand_motion) \
                if t == 0 else torch.randn_like(lhand_motion)
            noise_rhand_motion = torch.zeros_like(rhand_motion) \
                if t == 0 else torch.randn_like(rhand_motion)
            noise_obj_motion = torch.zeros_like(obj_motion) \
                if t == 0 else torch.randn_like(obj_motion)

            pred_X0_lhand, pred_X0_rhand, pred_X0_obj = self.forward(
                lhand_motion, rhand_motion, obj_motion, 
                obj_feat, text_feat,
                timesteps=t)
            
            beta = self.params["betas"][t]
            alpha = self.params["alphas"][t]
            alpha_bar = self.params["alpha_bars"][t]
            alpha_prev_bar = self.params["alpha_prev_bars"][t]
            log_variance = self.params["posterior_log_variance_clip"][t]

            coef_X0 = (beta * torch.sqrt(alpha_prev_bar) / (1 - alpha_bar))
            coef_noise = ((1 - alpha_prev_bar) * torch.sqrt(alpha) / (1 - alpha_bar))

            mu_xt_lhand = pred_X0_lhand * coef_X0 + sample_lhand * coef_noise
            mu_xt_rhand = pred_X0_rhand * coef_X0 + sample_rhand * coef_noise
            mu_xt_obj = pred_X0_obj * coef_X0 + sample_obj * coef_noise
            
            sample_lhand = mu_xt_lhand + torch.exp(0.5 * log_variance) * noise_lhand_motion
            sample_rhand = mu_xt_rhand + torch.exp(0.5 * log_variance) * noise_rhand_motion
            sample_obj = mu_xt_obj + torch.exp(0.5 * log_variance) * noise_obj_motion

        return sample_lhand, sample_rhand, sample_obj
