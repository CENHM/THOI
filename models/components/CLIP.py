import torch
import clip
from PIL import Image


class Clip:
    def __init__(self, device):
        self.device = device
        self.model, _ = clip.load("ViT-B/32", device=device, jit=False)

    def __tokenize(self, text):
        return torch.cat([clip.tokenize(t, truncate=True) for t in text]).float().to(self.device)
    
    def extract_feature(self, text: list):
        self.model.eval()
        with torch.no_grad():
            return self.model.encode_text(self.__tokenize(text))


# normal_repr = torch.Tensor.__repr__
# torch.Tensor.__repr__ = lambda self: f"{self.shape}_{normal_repr(self)}"
# device = "cuda" if torch.cuda.is_available() else "cpu"
# model, preprocess = clip.load("ViT-B/32", device=device, jit=False)

# text = torch.cat([clip.tokenize(f"Pour milk with the right hand."), clip.tokenize(f"Pour milk with the left hand.")]).to(device)

# with torch.no_grad():
#     # image_features = model.encode_image(image)
#     text_features = model.encode_text(text)
    
    # logits_per_image, logits_per_text = model(image, text)
    # probs = logits_per_image.softmax(dim=-1).cpu().numpy()




pass
# print("Label probs:", probs)  # prints: [[0.9927937  0.00421068 0.00299572]]