import torch
from models.components.PointNet import PointNet

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = PointNet().to(DEVICE)
normal_repr = torch.Tensor.__repr__
torch.Tensor.__repr__ = lambda self: f"{self.shape}_{normal_repr(self)}"

i = torch.randn((3, 8, 3)).float().to(DEVICE)


x = model(i)

pass