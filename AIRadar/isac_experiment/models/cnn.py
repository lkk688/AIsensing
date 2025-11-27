import torch
import torch.nn as nn

class CommDemapperCNN(nn.Module):
    """
    A simple CNN for demapping communication symbols from grid features.
    """
    def __init__(self, in_ch=2, width=32, depth=3, out_ch=2):
        super().__init__()
        layers = [nn.Conv2d(in_ch, width, 3, padding=1), nn.ReLU()]
        for _ in range(depth-1):
            layers += [nn.Conv2d(width, width, 3, padding=1), nn.ReLU()]
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Conv2d(width, out_ch, 1)  # 2 bits/logits per grid element
        
    def forward(self, x):  # x: (B,C,H,W)
        h = self.backbone(x)
        y = self.head(h)   # (B,2,H,W)
        return y
