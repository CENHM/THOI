from utils.utils import sqrtm
import torch


class Metric:
    def __init__(self) -> None:
        pass

    def calculate(self):
        pass
        


class FrechetDistance(Metric):
    def __init__(self) -> None:
        super().__init__()

    def __calculate_mu(self, x: torch.Tensor):
        return x.mean(dim=0)
    
    def __calculate_sigma(self, x: torch.Tensor):
        N = x.shape[0]
        return torch.cov(x.reshape(N, -1).T)

    def calculate_frechet_distance(self, mu1, sigma1, mu2, sigma2, eps=1e-6):
        """Numpy implementation of the Frechet Distance.
        The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
        and X_2 ~ N(mu_2, C_2) is
                d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

        Stable version by Dougal J. Sutherland.

        Params:
        -- mu1   : Numpy array containing the activations of a layer of the
                inception net (like returned by the function 'get_predictions')
                for generated samples.
        -- mu2   : The sample mean over activations, precalculated on an
                representative data set.
        -- sigma1: The covariance matrix over activations for generated samples.
        -- sigma2: The covariance matrix over activations, precalculated on an
                representative data set.

        Returns:
        --   : The Frechet Distance.
        """

        diff = mu1 - mu2

        covmean = torch.matmul(sigma1, sigma2)

        try:
            covmean = sqrtm(covmean)
        except Exception as e:
            print(f"Error computing sqrtm: {e}")
            offset = torch.eye(sigma1.shape[0], device=sigma1.device) * eps
            covmean = sqrtm(torch.matmul(sigma1 + offset, sigma2 + offset))

        tr_covmean = torch.trace(covmean)

        fid = torch.sum(diff ** 2) + torch.trace(sigma1) + torch.trace(sigma2) - 2 * tr_covmean
        return fid.item()
    
    def calculate(self, pred, true):
        mu_pred = self.__calculate_mu(pred)
        mu_true = self.__calculate_mu(true)

        sigma_pred = self.__calculate_sigma(pred)
        sigma_true = self.__calculate_sigma(true)

        return self.calculate_frechet_distance(mu_pred, sigma_pred, mu_true, sigma_true)
    
