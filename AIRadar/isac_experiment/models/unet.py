import torch
import torch.nn as nn

class UNetLite(nn.Module):
    """
    A lightweight U-Net architecture for Radar Range-Doppler denoising/detection.
    """
    def __init__(self, in_ch=1, ch=32):
        super().__init__()
        # Encoder
        self.e1 = nn.Sequential(nn.Conv2d(in_ch,ch,3,padding=1), nn.ReLU(), nn.Conv2d(ch,ch,3,padding=1), nn.ReLU())
        self.p1 = nn.MaxPool2d(2)
        self.e2 = nn.Sequential(nn.Conv2d(ch,2*ch,3,padding=1), nn.ReLU(), nn.Conv2d(2*ch,2*ch,3,padding=1), nn.ReLU())
        self.p2 = nn.MaxPool2d(2)
        
        # Bridge
        self.b  = nn.Sequential(nn.Conv2d(2*ch,4*ch,3,padding=1), nn.ReLU(), nn.Conv2d(4*ch,4*ch,3,padding=1), nn.ReLU())
        
        # Decoder
        self.u2 = nn.ConvTranspose2d(4*ch,2*ch,2,stride=2)
        self.d2 = nn.Sequential(nn.Conv2d(4*ch,2*ch,3,padding=1), nn.ReLU(), nn.Conv2d(2*ch,2*ch,3,padding=1), nn.ReLU())
        self.u1 = nn.ConvTranspose2d(2*ch,ch,2,stride=2)
        self.d1 = nn.Sequential(nn.Conv2d(2*ch,ch,3,padding=1), nn.ReLU(), nn.Conv2d(ch,ch,3,padding=1), nn.ReLU())
        
        # Output
        self.out = nn.Conv2d(ch,1,1)
        
    def forward(self,x):
        e1 = self.e1(x)            # HxW
        e2 = self.e2(self.p1(e1))  # H/2 x W/2
        b  = self.b(self.p2(e2))   # H/4 x W/4
        
        d2 = self.d2(torch.cat([self.u2(b), e2], dim=1))
        d1 = self.d1(torch.cat([self.u1(d2), e1], dim=1))
        return self.out(d1)        # logits
