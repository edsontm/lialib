import torch

class ArcFace(torch.Module):
    def __init__(self,backbone):
        self.backbone = backbone
        
    def add_dml_head(self):
        pass

