import torch
import clip


class Clip:
    def __init__(self, device=None):
        self.device = device
        self.model, _ = clip.load("ViT-B/32", device=device, jit=False)

    def __tokenize(self, text):
        return torch.cat([clip.tokenize(t, truncate=True) for t in text]).long().to(self.device)
    
    def extract_feature(self, text: list):
        self.model.eval()
        with torch.no_grad():
            return self.model.encode_text(self.__tokenize(text))
