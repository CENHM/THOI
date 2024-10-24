import torch
from torch.nn import functional as F
import math

from utils.arguments import CFGS



class Diffusion:
    def __init__(
            self,
            timesteps=CFGS.timesteps
        ):

        self.timesteps = timesteps
        
        self.beta = self.__beta_scheduler()

        # \alpha_t = 1 - \beta_t
        # \hat{\alpha_t} = \prod^{t}_{i=1} \alpha_i    
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, axis=0)

        self.sqrt_alpha_hat = torch.sqrt(self.alpha_hat)
        self.sqrt_weighted_alpha_hat = torch.sqrt(1.0 - self.alpha_hat)








        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.)
        
        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = torch.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)
        
        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        # below: log calculation clipped because the posterior variance is 0 at the beginning
        # of the diffusion chain
        self.posterior_log_variance_clipped = torch.log(self.posterior_variance.clamp(min =1e-20))
        
        self.posterior_mean_coef1 = (
            self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * torch.sqrt(self.alphas)
            / (1.0 - self.alphas_cumprod)
        )

    def __linear_beta_scheduler(self, beta_s=1e-4, beta_e=0.02):
        return torch.linspace(beta_s, beta_e, self.timesteps, dtype=torch.float64)

    def __cosine_beta_scheduler(self, max_beta=0.999):
        cos_cal = lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,

        t = torch.arange(self.timesteps) / self.timesteps
        t1, t2 = t[:-1], t[1:]
        cos_t1, cos_t2 = cos_cal(t1), cos_cal(t2)
        betas = 1 - cos_t2 / cos_t1
        
        return torch.clamp(betas, max=max_beta)

    def __beta_scheduler(self):
        if CFGS.beta_schedule == "linear":
            return self.__linear_beta_scheduler()
        elif CFGS.beta_schedule == "cosine":
            return self.__cosine_beta_scheduler()
        raise ValueError(f"unknown beta schedule of: {CFGS.beta_schedule}")

    # get the param of given timestep t
    def _extract(self, a, t, x_shape):
        B = t.shape[0]
        out = a.to(t.device).gather(0, t).float()
        out = out.reshape(B, *((1,) * (len(x_shape) - 1)))
        return out
    
    # forward diffusion (using the nice property): q(x_t | x_0)
    def q_sample(self, x, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x)

        sqrt_alpha_hat_t = self._extract(self.sqrt_alpha_hat, t, x.shape)
        sqrt_weighted_alpha_hat_t = self._extract(self.sqrt_weighted_alpha_hat, t, x.shape)

        return sqrt_alpha_hat_t * x + sqrt_weighted_alpha_hat_t * noise
    
    # Get the mean and variance of q(x_t | x_0).
    def q_mean_variance(self, x_start, t):
        mean = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = self._extract(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = self._extract(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance
    
    # Compute the mean and variance of the diffusion posterior: q(x_{t-1} | x_t, x_0)
    def q_posterior_mean_variance(self, x_start, x_t, t):
        posterior_mean = (
            self._extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + self._extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = self._extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = self._extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped
    
    # compute x_0 from x_t and pred noise: the reverse of `q_sample`
    def predict_start_from_noise(self, x_t, t, noise):
        return (
            self._extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            self._extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )
    
    # compute predicted mean and variance of p(x_{t-1} | x_t)
    def p_mean_variance(self, model, x_t, t, clip_denoised=True):
        # predict noise using model
        pred_noise = model(x_t, t)
        # get the predicted x_0: different from the algorithm2 in the paper
        x_recon = self.predict_start_from_noise(x_t, t, pred_noise)
        if clip_denoised:
            x_recon = torch.clamp(x_recon, min=-1., max=1.)
        model_mean, posterior_variance, posterior_log_variance = \
                    self.q_posterior_mean_variance(x_recon, x_t, t)
        return model_mean, posterior_variance, posterior_log_variance
        
    # denoise_step: sample x_{t-1} from x_t and pred_noise
    @torch.no_grad()
    def p_sample(self, model, x_t, t, clip_denoised=True):
        # predict mean and variance
        model_mean, _, model_log_variance = self.p_mean_variance(model, x_t, t,
                                                    clip_denoised=clip_denoised)
        noise = torch.randn_like(x_t)
        # no noise when t == 0
        nonzero_mask = ((t != 0).float().view(-1, *([1] * (len(x_t.shape) - 1))))
        # compute x_{t-1}
        pred_img = model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise
        return pred_img
    
    # denoise: reverse diffusion
    @torch.no_grad()
    def p_sample_loop(self, model, shape):
        batch_size = shape[0]
        device = next(model.parameters()).device
        # start from pure noise (for each example in the batch)
        img = torch.randn(shape, device=device)
        imgs = []
        for i in tqdm(reversed(range(0, timesteps)), desc='sampling loop time step', total=timesteps):
            img = self.p_sample(model, img, torch.full((batch_size,), i, device=device, dtype=torch.long))
            imgs.append(img.cpu().numpy())
        return imgs
    
    # sample new images
    @torch.no_grad()
    def sample(self, model, image_size, batch_size=8, channels=3):
        return self.p_sample_loop(model, shape=(batch_size, channels, image_size, image_size))
    
    # compute train losses
    def train_losses(self, model, x, t):
        # generate random noise
        noise = torch.randn_like(x)
        # get x_t
        x_noisy = self.q_sample(x, t, noise=noise)
        predicted_noise = model(x_noisy, t)
        loss = F.mse_loss(noise, predicted_noise)
        return loss